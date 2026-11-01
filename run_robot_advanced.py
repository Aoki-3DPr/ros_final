"""
指示(ミッション)受付層つき 自律探索コントローラ

  ros2 topic pub --once /mission_cmd std_msgs/msg/String "{data: '3分間探索して、その後スタートに戻って'}"
  ros2 topic pub --once /mission_cmd std_msgs/msg/String "{data: '{\"task\":\"goto\",\"params\":{\"x\":2.0,\"y\":1.0}}'}"
  ros2 topic echo /mission_status

入力: /mission_cmd (std_msgs/String)  自然言語 または JSON
出力: /mission_status (std_msgs/String)  受理結果・完了通知・報告

自然言語はGeminiでコマンドJSONに変換する(失敗時は簡易ルールで解釈)。
JSONはそのまま受け付ける。

対応タスク (LiDAR + オドメトリのみで実現できるもの)
  explore        全域探索 (既定)
  timed_explore  時間制限付き探索 (終了後 stop / return_home)
  return_home    来た道を逆にたどってスタート地点へ帰還
  goto           座標へ移動 (スタート基準 / frame:"robot" で現在位置基準)
  wait           その場で待機
  stop           その場で停止し指示待ち
  report         走行距離・探索面積・行き止まり候補などを報告 (タスクは変えない)
方針(policy): 左右前後の優先 / 立入禁止領域(円) の追加・解除

llm_hybrid_controller.py と同じフォルダに置くこと。
"""
import json
import math
import queue
import os
import re
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

import rclpy
from std_msgs.msg import String
from google.genai import types

from llm_hybrid_controller import LLMHybridController, normalize_angle, clamp


class GlobalMap:
    """スキャンを蓄積したオドメトリ座標系の占有格子 (0.1m)。
    LLMに「全体としてどこまで探索したか」の事実(フロンティア)を渡すために使う。
    各セルは対数オッズ風の値: 空きに観測されるほど負、壁に観測されるほど正。"""
    RES = 0.1
    L_FREE, L_OCC, CLAMP = -0.4, 0.9, 3.0
    INFLATE = 2            # 到達判定用の膨張 [セル] (=0.2m)
    MIN_FRONTIER = 4       # これ未満(0.4m未満)の境界はロボットが通れないので無視

    def __init__(self):
        self.lock = threading.Lock()
        self.val = {}

    def cell(self, x, y):
        return (round(x / self.RES), round(y / self.RES))

    def odom(self, c):
        return (c[0] * self.RES, c[1] * self.RES)

    def update(self, s):
        x, y, yaw, rmax = s.x, s.y, s.yaw, s.rmax
        free, hit = set(), set()
        for ang, r in s.scan_points:
            ca, sa = math.cos(yaw + ang), math.sin(yaw + ang)
            for k in range(int(r / self.RES)):
                d = k * self.RES
                free.add(self.cell(x + d * ca, y + d * sa))
            if r < rmax - 1e-3:                       # 反射あり = 壁
                hit.add(self.cell(x + r * ca, y + r * sa))
        free -= hit
        with self.lock:
            for c in free:
                self.val[c] = max(self.val.get(c, 0.0) + self.L_FREE, -self.CLAMP)
            for c in hit:
                self.val[c] = min(self.val.get(c, 0.0) + self.L_OCC, self.CLAMP)

    def analyze(self, rx, ry):
        """ロボット位置から既知の空き領域だけで到達できるフロンティアを返す"""
        with self.lock:
            val = dict(self.val)
        occ = {c for c, v in val.items() if v > 0.5}
        free = {c for c, v in val.items() if v < -0.3}
        I = self.INFLATE
        blocked = set()
        for (ox, oy) in occ:
            for dx in range(-I, I + 1):
                for dy in range(-I, I + 1):
                    blocked.add((ox + dx, oy + dy))
        start = self.cell(rx, ry)
        parent = {start: None}
        dist = {start: 0}
        dq = deque([start])
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        while dq:
            cur = dq.popleft()
            for dx, dy in dirs:
                n = (cur[0] + dx, cur[1] + dy)
                if n in parent or n not in free or n in blocked:
                    continue
                parent[n] = cur
                dist[n] = dist[cur] + 1
                dq.append(n)

        def unknown(c):
            return c not in free and c not in occ
        fr = {c for c in parent
              if any(unknown((c[0] + dx, c[1] + dy)) for dx, dy in dirs[:4])}
        clusters, seen = [], set()
        for c in fr:
            if c in seen:
                continue
            comp, st = [], [c]
            seen.add(c)
            while st:
                u = st.pop()
                comp.append(u)
                for dx, dy in dirs:
                    n = (u[0] + dx, u[1] + dy)
                    if n in fr and n not in seen:
                        seen.add(n)
                        st.append(n)
            if len(comp) >= self.MIN_FRONTIER:
                mx = sum(u[0] for u in comp) / len(comp)
                my = sum(u[1] for u in comp) / len(comp)
                rep = min(comp, key=lambda u: (u[0] - mx) ** 2 + (u[1] - my) ** 2)
                clusters.append({'cell': rep, 'odom': self.odom(rep), 'size': len(comp),
                                 'dist': dist[rep] * self.RES})
        clusters.sort(key=lambda f: f['dist'])
        return SimpleNamespace(frontiers=clusters, parent=parent,
                               area=len(free) * self.RES ** 2)

    def waypoint(self, an, target_cell, lookahead=1.2):
        """既知の空き領域を通る最短経路上で、ロボットから lookahead 先の点(odom座標)"""
        if target_cell not in an.parent:
            return None
        path, c = [], target_cell
        while c is not None:
            path.append(c)
            c = an.parent[c]
        path.reverse()
        k = min(len(path) - 1, max(1, int(lookahead / self.RES)))
        return self.odom(path[k])


class LLMMissionController(LLMHybridController):
    ARRIVE_TOL = 0.25          # goto の到達判定 [m]
    HOME_TOL = 0.30            # 帰還の到達判定 [m]
    TRAIL_STEP = 0.3           # 帰り道(パンくず)の記録間隔 [m]
    TRAIL_REACH = 1.4          # 帰還時に「次の中継点」として見る範囲 [m]
    GOTO_TIMEOUT = 600.0       # goto の打ち切り時間 [s]
    DEAD_END_CLEAR = 0.8       # 行き止まり判定: 「空いている」とみなす距離 [m]
    TASKS = ('explore', 'timed_explore', 'return_home', 'goto', 'wait', 'stop', 'report', 'none')

    def __init__(self):
        # 基底クラスのスレッドが動き出す前に状態を用意しておく
        self.m_lock = threading.Lock()
        self.task_id = 0
        self.task = {'type': 'explore', 'id': 0, 'deadline': None, 'then': 'stop',
                     'target': None, 'until': None}
        self.policy = {'prefer': 'none', 'avoid': []}    # avoid: [(odom_x, odom_y, r)]
        self.home = None                                 # (x, y, yaw) 起動時の姿勢
        self.trail = []
        self.record_trail = True
        self.distance = 0.0
        self._last_xy = None
        self.dead_ends = []
        self.t_start = time.time()
        self.cmd_queue = queue.Queue()
        self.status_pub = None
        self.gmap = GlobalMap()
        self.last_frontiers = []

        super().__init__()

        self.status_pub = self.create_publisher(String, 'mission_status', 10)
        self.create_subscription(String, 'mission_cmd', self.cmd_callback, 10)
        threading.Thread(target=self.mission_loop, daemon=True).start()
        threading.Thread(target=self.gmap_loop, daemon=True).start()
        # 端末からの対話入力 (MISSION_STDIN=0 で無効 / =1 で非ttyでも有効)
        use_stdin = os.environ.get('MISSION_STDIN', 'auto')
        if use_stdin == '1' or (use_stdin == 'auto' and sys.stdin is not None and sys.stdin.isatty()):
            threading.Thread(target=self.stdin_loop, daemon=True).start()
        self.say("指示待ち受け開始: /mission_cmd と端末入力 (既定タスク: 全域探索)")

    # ==================================================================
    # 入出力
    # ==================================================================
    def cmd_callback(self, msg):
        self.cmd_queue.put(msg.data)

    HELP = (
        "【端末から指示を入力できます】(Enterで送信)\n"
        "  例: 探索して / 3分間探索して、その後スタートに戻って / 前方2m左1mに行って\n"
        "      スタートに戻って / 5秒待って / 止まって / 報告して / 右側を優先して\n"
        "      左の通路には入らないで / 立入禁止を解除して\n"
        "  JSON直接入力も可: {\"task\":\"goto\",\"params\":{\"x\":2,\"y\":1}}\n"
        "  help: この表示   status: 現在の状態   quit: 終了")

    def stdin_loop(self):
        """実行中に端末へ打った指示をそのまま受け付ける"""
        print(self.HELP, flush=True)
        while self.is_running:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if line == '':                 # EOF (パイプ終端/端末クローズ)
                return
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if low in ('help', '?', 'ヘルプ'):
                print(self.HELP, flush=True)
            elif low in ('status', '状態'):
                self.say(self.status_line())
            elif low in ('quit', 'exit', '終了'):
                self.say("終了します")
                self.is_running = False
                try:
                    self.halt()
                finally:
                    rclpy.shutdown()
                return
            else:
                self.cmd_queue.put(line)

    def say(self, text):
        self.get_logger().info(text)
        if self.status_pub is not None:
            self.status_pub.publish(String(data=text))

    # ==================================================================
    # 座標 / センサの拡張
    # ==================================================================
    def home_to_odom(self, p):
        hx, hy, hyaw = self.home
        c, s = math.cos(hyaw), math.sin(hyaw)
        return (hx + c * p[0] - s * p[1], hy + s * p[0] + c * p[1])

    def odom_to_home(self, p):
        hx, hy, hyaw = self.home
        c, s = math.cos(hyaw), math.sin(hyaw)
        dx, dy = p[0] - hx, p[1] - hy
        return (c * dx + s * dy, -s * dx + c * dy)

    def point_to_odom(self, snap, x, y, frame):
        if frame == 'robot':
            return self.to_global(snap, (x, y))
        return self.home_to_odom((x, y))

    def odom_callback(self, msg):
        super().odom_callback(msg)
        x, y, yaw = self.get_pose()
        with self.m_lock:
            if self.home is None:
                self.home = (x, y, yaw)
                self.trail = [(x, y)]
                self._last_xy = (x, y)
                return
            self.distance += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
            self._last_xy = (x, y)
            if self.record_trail and (not self.trail or math.hypot(
                    x - self.trail[-1][0], y - self.trail[-1][1]) >= self.TRAIL_STEP):
                self.trail.append((x, y))

    def snapshot(self, with_visited=False):
        """立入禁止領域を「壁」として障害物集合に混ぜる (グリッド層だけに効く)"""
        s = super().snapshot(with_visited)
        with self.m_lock:
            avoid = list(self.policy['avoid'])
            has_home = self.home is not None
        if avoid and has_home:
            extra = set()
            H = self.MAP_HALF
            for ax, ay, r in avoid:
                lx, ly = self.to_local(s, (ax, ay))
                ci, cj, rc = lx / self.CELL, ly / self.CELL, r / self.CELL
                if math.hypot(ci, cj) - rc > H + 2:
                    continue
                for dx in range(-int(rc) - 1, int(rc) + 2):
                    for dy in range(-int(rc) - 1, int(rc) + 2):
                        cell = (round(ci) + dx, round(cj) + dy)
                        if (abs(cell[0]) <= H and abs(cell[1]) <= H and
                                math.hypot(cell[0] - ci, cell[1] - cj) <= rc):
                            extra.add(cell)
            s.obstacles = set(s.obstacles) | extra
        return s

    # ==================================================================
    # タスク管理
    # ==================================================================
    def set_task(self, ttype, expect_id=None, **fields):
        with self.m_lock:
            if expect_id is not None and self.task['id'] != expect_id:
                return False
            self.task_id += 1
            self.task = {'type': ttype, 'id': self.task_id, 'deadline': None,
                         'then': 'stop', 'target': None, 'until': None}
            self.task.update(fields)
            self.record_trail = (ttype != 'return_home')   # 帰還中はパンくずを固定
        self.abort_goal()
        return True

    def abort_goal(self):
        """現在の目標を破棄して再計画させる (走行中でも即座に切り替わる)"""
        with self.lock:
            self.goal = None
            self.goal_id += 1
            self.plan_requested = True
        self.need_plan.set()

    def complete(self, tid, msg):
        with self.m_lock:
            if self.task['id'] != tid:
                return
        self.say(msg)
        self.set_task('idle', expect_id=tid)

    def status_line(self):
        snap = self.snapshot()
        with self.m_lock:
            ttype, dist = self.task['type'], self.distance
            home_ok = self.home is not None
        pos = self.odom_to_home((snap.x, snap.y)) if home_ok else (0.0, 0.0)
        return f"task={ttype}, 位置(スタート基準)=({pos[0]:.1f},{pos[1]:.1f}), 走行{dist:.1f}m"

    # ==================================================================
    # 指示の解釈
    # ==================================================================
    def interpret(self, text):
        s = text.strip()
        if s.startswith('{'):
            try:
                return json.loads(s)
            except Exception:
                pass
        try:
            return self.interpret_llm(s)
        except Exception as e:
            self.get_logger().warn(f"LLMでの指示解釈に失敗: {e} → ルールで解釈")
            return self.interpret_rules(s)

    def interpret_llm(self, text):
        prompt = f"""
あなたはロボットへの指示を、実行可能なコマンドJSONに変換するインタプリタです。
座標系: スタート地点が原点、X軸が前方、Y軸が左(単位m)。frame="robot" を付けると現在のロボット位置基準。

【現在の状態】{self.status_line()}
【ユーザーの指示】{text}

選べる task:
- "explore": 全域探索
- "timed_explore": 時間制限付き探索 (params.duration_s 必須, params.then は "stop" か "return_home")
- "return_home": 来た道を戻ってスタート地点へ帰還
- "goto": 座標へ移動 (params.x, params.y, 任意で params.frame)
- "wait": その場で待機 (params.duration_s 必須)
- "stop": その場で停止して指示待ち
- "report": 状況の報告 (走行距離・探索面積・行き止まり候補)
- "none": タスクは変えない(方針だけ変える場合や、理解できない場合)

policy (任意):
- "prefer": "left"|"right"|"front"|"back"|"none"|"keep"
- "avoid_regions": [{{"x":..,"y":..,"r":半径m,"frame":"home"|"robot"}}]  立入禁止の円
- "clear_avoid": true で立入禁止をすべて解除
「左の通路に入らないで」のような指示は、frame:"robot" で左前方の位置(例 x=1.0,y=1.0,r=0.7)を禁止領域にする。

出力はJSONのみ。例:
{{"task":"goto","params":{{"x":2.0,"y":1.0}},"policy":{{}},"reply":"X=2.0,Y=1.0へ向かいます"}}
"""
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.0))
        data = json.loads(response.text.strip())
        return data[0] if isinstance(data, list) else data

    @staticmethod
    def interpret_rules(s):
        low = s.lower()
        m = re.search(r'(\d+(?:\.\d+)?)\s*(分|min|秒|sec)', s)
        dur = None
        if m:
            dur = float(m.group(1)) * (60.0 if m.group(2) in ('分', 'min') else 1.0)
        if re.search(r'停止|止まっ|止まれ|ストップ|stop', low):
            return {'task': 'stop', 'reply': '停止します'}
        if re.search(r'戻|帰|return|home', low):
            return {'task': 'return_home', 'reply': 'スタート地点へ戻ります'}
        if re.search(r'待', s) and dur:
            return {'task': 'wait', 'params': {'duration_s': dur}, 'reply': f'{dur:.0f}秒待機します'}
        if re.search(r'報告|レポート|report', low):
            return {'task': 'report'}
        if re.search(r'探索|explore', low):
            if dur:
                return {'task': 'timed_explore', 'params': {'duration_s': dur},
                        'reply': f'{dur:.0f}秒間探索します'}
            return {'task': 'explore', 'reply': '探索します'}
        return {'task': 'none', 'reply': f'指示を解釈できませんでした: {s}'}

    # ==================================================================
    # コマンドの適用
    # ==================================================================
    def handle_command(self, text):
        t0 = time.time()
        while self.home is None and time.time() - t0 < 5.0:
            time.sleep(0.1)
        if self.home is None:
            self.say("オドメトリ未受信のため指示を受け付けられません")
            return
        self.apply_command(self.interpret(text))

    def apply_command(self, cmd):
        task = str(cmd.get('task', 'none'))
        params = cmd.get('params') or {}
        pol = cmd.get('policy') or {}
        reply = cmd.get('reply')
        if task not in self.TASKS:
            self.say(f"未対応のタスクです: {task}")
            return

        # --- 方針の変更 ---------------------------------------------
        snap = self.snapshot()
        changed = False
        with self.m_lock:
            if pol.get('clear_avoid'):
                self.policy['avoid'] = []
                changed = True
            pref = pol.get('prefer', 'keep')
            if pref in ('left', 'right', 'front', 'back', 'none') and pref != self.policy['prefer']:
                self.policy['prefer'] = pref
                changed = True
            for r in pol.get('avoid_regions') or []:
                try:
                    gx, gy = self.point_to_odom(snap, float(r['x']), float(r['y']),
                                                r.get('frame', 'home'))
                    self.policy['avoid'].append((gx, gy, clamp(float(r.get('r', 0.5)), 0.2, 2.0)))
                    changed = True
                except Exception:
                    self.say(f"立入禁止領域の指定が不正です: {r}")

        # --- タスクの変更 -------------------------------------------
        try:
            if task == 'explore':
                self.set_task('explore')
            elif task == 'timed_explore':
                d = float(params['duration_s'])
                then = params.get('then', 'stop')
                self.set_task('explore', deadline=time.time() + d,
                              then=then if then in ('stop', 'return_home') else 'stop')
            elif task == 'return_home':
                self.set_task('return_home')
            elif task == 'goto':
                target = self.point_to_odom(snap, float(params['x']), float(params['y']),
                                            params.get('frame', 'home'))
                self.set_task('goto', target=target, deadline=time.time() + self.GOTO_TIMEOUT)
            elif task == 'wait':
                self.set_task('wait', until=time.time() + float(params['duration_s']))
            elif task == 'stop':
                self.set_task('idle')
            elif task == 'report':
                self.say(self.build_report())
                return
            elif changed:
                self.abort_goal()          # 方針だけ変わった場合も計画し直す
        except (KeyError, TypeError, ValueError) as e:
            self.say(f"指示のパラメータが不足/不正です ({task}): {e}")
            return
        self.say(reply or f"受理: {task} 方針={self.policy['prefer']} 立入禁止={len(self.policy['avoid'])}件")

    def build_report(self):
        snap = self.snapshot(with_visited=True)
        hp = self.odom_to_home((snap.x, snap.y))
        with self.m_lock:
            ttype, dist = self.task['type'], self.distance
            de = [self.odom_to_home(p) for p in self.dead_ends]
        area = len(snap.visited) * 0.25
        an = self.gmap.analyze(snap.x, snap.y)
        de_txt = ", ".join(f"({x:.1f},{y:.1f})" for x, y in de) if de else "なし"
        return (f"[報告] タスク={ttype} / 現在位置(スタート基準) X={hp[0]:.1f} Y={hp[1]:.1f} / "
                f"走行距離 {dist:.1f}m / 探索面積 約{area:.1f}㎡ / 経過 {time.time() - self.t_start:.0f}秒 / "
                f"直近の失敗 {len(self.stuck_log)}件 / 行き止まり候補 {len(de)}件: {de_txt} / "
                f"地図: 既知の空き約{an.area:.1f}㎡, 到達可能な未探索境界 {len(an.frontiers)}件")

    # ==================================================================
    # ミッションスレッド (指示受付 + タイマ + 到達判定)
    # ==================================================================
    def mission_loop(self):
        while self.is_running:
            try:
                text = self.cmd_queue.get(timeout=0.2)
            except queue.Empty:
                text = None
            try:
                if text:
                    self.handle_command(text)
                if self.home is not None:
                    self.check_task()
            except Exception as e:
                self.get_logger().error(f"ミッション処理エラー: {e}")

    def gmap_loop(self):
        while self.is_running:
            time.sleep(0.5)
            try:
                with self.lock:
                    ready = self.scan_count > 0 and self.odom_count > 0
                if ready:
                    self.gmap.update(super().snapshot())
            except Exception as e:
                self.get_logger().error(f"地図更新エラー: {e}")

    def check_task(self):
        now = time.time()
        with self.m_lock:
            task = dict(self.task)
        ttype, tid = task['type'], task['id']
        snap = self.snapshot()

        if ttype == 'explore' and task['deadline'] and now > task['deadline']:
            if task['then'] == 'return_home':
                self.say("時間切れ: スタート地点へ帰還します")
                self.set_task('return_home', expect_id=tid)
            else:
                self.complete(tid, "時間切れ: 探索を終了しました")
            return
        if ttype == 'wait' and now >= task['until']:
            self.complete(tid, "待機終了")
            return
        if ttype == 'goto':
            tl = self.to_local(snap, task['target'])
            if math.hypot(tl[0], tl[1]) < self.ARRIVE_TOL:
                self.complete(tid, "目標座標に到達しました")
                return
            if now > task['deadline']:
                self.complete(tid, "goto: タイムアウトで中止しました")
                return
        if ttype == 'return_home':
            hx, hy, _ = self.home
            if math.hypot(snap.x - hx, snap.y - hy) < self.HOME_TOL:
                self.complete(tid, "スタート地点に帰還しました")
                return
        if ttype in ('explore', 'goto'):
            self.update_dead_ends(snap)

    def update_dead_ends(self, snap):
        """行き止まり候補(ヒューリスティック): 近距離で空いている方向が
        ひとまとまり(90度以内)の1方向だけ = 来た道以外が塞がっている"""
        clear = []
        for k in range(12):
            c = normalize_angle(math.radians(k * 30))
            vals = [r for a, r in snap.scan_points
                    if abs(normalize_angle(a - c)) <= math.radians(15)]
            clear.append(min(vals) if vals else 0.0)
        opened = [v >= self.DEAD_END_CLEAR for v in clear]
        n = sum(opened)
        groups = sum(1 for k in range(12) if opened[k] and not opened[k - 1])
        if n == 0 or n > 3 or groups != 1:
            return
        p = (snap.x, snap.y)
        with self.m_lock:
            if any(math.hypot(p[0] - q[0], p[1] - q[1]) < 0.7 for q in self.dead_ends):
                return
            self.dead_ends.append(p)
        self.get_logger().info(f"行き止まり候補を記録: ({p[0]:.1f},{p[1]:.1f})")

    # ==================================================================
    # プランナー (タスクごとに目標の決め方を切り替える)
    # ==================================================================
    def mission_text(self):
        with self.m_lock:
            task, pol = dict(self.task), dict(self.policy)
        t = "迷路内の未踏エリアをすべて探索する"
        if task['deadline'] and task['type'] == 'explore':
            t += f"(残り時間 {max(0.0, task['deadline'] - time.time()):.0f}秒。時間が来たら探索を終える)"
        names = {'left': '左側', 'right': '右側', 'front': '前方', 'back': '後方'}
        if pol['prefer'] in names:
            t += f"。{names[pol['prefer']]}の未踏エリアを優先して選ぶ"
        if pol['avoid']:
            t += "。立入禁止領域は地図上で障害物として反映済み"
        return t

    def choose_fallback(self, unvisited, visited):
        pool = unvisited if unvisited else visited
        pref = self.policy['prefer']

        def key(p):
            x, y, s = p
            main = {'left': y, 'right': -y, 'back': -x}.get(pref, x)
            sub = x if pref in ('left', 'right', 'back') else -abs(y)
            return (main, sub, s)
        best = max(pool, key=key)
        return best[0], best[1]

    def pick_return_goal(self, snap):
        """来た道(パンくず)のうち、見える範囲で最もスタートに近い点を次の中継点にする"""
        hx, hy, _ = self.home
        reachable = self.last_reachable
        with self.m_lock:
            trail = list(self.trail)
        for i, p in enumerate(trail):
            lx, ly = self.to_local(snap, p)
            if (math.hypot(lx, ly) <= self.TRAIL_REACH and
                    (round(lx / self.CELL), round(ly / self.CELL)) in reachable):
                with self.m_lock:
                    self.trail = self.trail[:i]      # この点以降は消化済み
                return p
        return (hx, hy)

    def global_waypoint(self, snap, target_odom):
        an = self.gmap.analyze(snap.x, snap.y)
        return self.gmap.waypoint(an, self.gmap.cell(*target_odom))

    def frontier_text(self, snap):
        an = self.gmap.analyze(snap.x, snap.y)
        frs = an.frontiers[:6]
        self.last_frontiers = frs
        lines = [f"既知の空き領域: 約{an.area:.1f}㎡ / 到達可能な未探索境界(フロンティア): {len(an.frontiers)}件"]
        for i, f in enumerate(frs, 1):
            hp = self.odom_to_home(f['odom'])
            lines.append(f"  F{i}: スタート基準(X={hp[0]:.1f}, Y={hp[1]:.1f}) 経路距離約{f['dist']:.1f}m 幅{f['size'] * 0.1:.1f}m")
        if len(an.frontiers) > len(frs):
            lines.append(f"  ...ほか{len(an.frontiers) - len(frs)}件")
        return an, frs, "\n".join(lines)

    def ask_llm(self, grid_context, unvisited, visited, global_text=""):
        extra = ""
        if 'history' in self.requested_info:
            extra += "\n【直近の移動履歴】\n" + "\n".join(self.history) + "\n"
        if 'stuck_log' in self.requested_info:
            extra += "\n【直近の失敗ログ】\n" + ("\n".join(self.stuck_log) or "なし") + "\n"
        prompt = f"""
あなたは自律移動ロボットの探索エージェント(大脳)です。ロボットは止まらずに動き続けており、
あなたの目標は走行中に先読みで更新されます。

【ミッション】{self.mission}

【前回のあなたのメモ(thought_scratchpad)】
{self.scratchpad}

【直近の失敗回数】{len(self.stuck_log)}
{extra}
【全体の探索状況 (走行中に蓄積した地図から算出した事実)】
{global_text}

【周囲(3.6m四方)のマップ】
{grid_context}

【未踏マス】 (x, y, 安全面積スコア): {list(unvisited)}
【探索済みマス】 (x, y, 安全面積スコア): {list(visited)}

【あなたが決めること】
1. 探索が完了したか。完了と判断したら is_completed を true にする。
   システムは検証しません。上の「全体の探索状況」を根拠に自分で判断してください
   (到達可能なフロンティアが残っているなら、未探索の場所があるということです)。
2. 完了でなければ次の行き先を1つ選ぶ。
   (a) 近くの安全なマス: target_x, target_y (上のリストの値をそのまま)
   (b) 遠くの未探索境界: target_frontier に F の番号(整数)
   近くに未踏マスがあれば(a)、近くに無くフロンティアが残っていれば(b)が基本です。
3. thought_scratchpad に「現在の仮説・次にやる予定」を短く(200字以内)書く。次のあなたに引き継がれる。
4. 次回さらに情報が必要なら requested_next_info に "history" / "stuck_log" を入れる(不要なら空配列)。
5. 出力はJSONのみ。

フォーマット例:
{{"thought_scratchpad": "右の広い空間を探索中。", "requested_next_info": [],
  "is_completed": false, "target_frontier": null, "target_x": 1.0, "target_y": 0.5}}
"""
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.0))
        data = json.loads(response.text.strip())
        return data[0] if isinstance(data, list) else data

    def plan_once(self):
        with self.m_lock:
            task = dict(self.task)
        mid, ttype = task['id'], task['type']
        if ttype in ('idle', 'wait'):
            with self.lock:
                self.plan_requested = False
            return

        with self.lock:
            idle = self.goal is None
        if idle:
            self.wait_fresh_scan()
        snap = self.snapshot(with_visited=True)
        grid_context, unvisited, visited = self.create_local_grid(snap)
        print("\n" + grid_context)

        goal_global = None
        if self.force_escape or (not unvisited and not visited):
            tx, ty = self.get_escape_target(snap)
            self.force_escape = False
            self.get_logger().warn(f"脱出移動: X={tx}m, Y={ty}m")
            goal_global = self.to_global(snap, (tx, ty))

        elif ttype == 'return_home':
            goal_global = self.pick_return_goal(snap)

        elif ttype == 'goto':
            target = task['target']
            tl = self.to_local(snap, target)
            d = math.hypot(tl[0], tl[1])
            cell = (round(tl[0] / self.CELL), round(tl[1] / self.CELL))
            if d <= 1.5 and cell in self.last_reachable:
                goal_global = target                       # 見えている範囲なら直接向かう
            elif (wp := self.global_waypoint(snap, target)) is not None:
                goal_global = wp                           # 既知地図上の経路の途中点へ
            else:                                          # 未知の所は「目標に近い・未踏」を貪欲に選ぶ
                cands = [(p, False) for p in unvisited] + [(p, True) for p in visited]
                best = None
                for p, was_visited in cands:
                    g = self.to_global(snap, (p[0], p[1]))
                    score = math.hypot(g[0] - target[0], g[1] - target[1]) + (0.6 if was_visited else 0.0)
                    if best is None or score < best[0]:
                        best = (score, g)
                goal_global = best[1]

        else:                                              # explore
            self.mission = self.mission_text()
            an, frs, global_text = self.frontier_text(snap)

            def frontier_goal(k):
                f = frs[k]
                wp = self.gmap.waypoint(an, f['cell'], 1.2)
                return wp if wp is not None else f['odom']

            try:
                data = self.ask_llm(grid_context, unvisited, visited, global_text)
                self.scratchpad = str(data.get('thought_scratchpad', self.scratchpad))[:400]
                req = data.get('requested_next_info', [])
                self.requested_info = [req] if isinstance(req, str) else list(req)

                # 完了の判断はLLMに任せる (Python側では上書きしない)
                if data.get('is_completed') is True:
                    self.complete(mid, f"★探索完了(LLM判断)★ 残フロンティア{len(an.frontiers)}件 / "
                                       f"未踏マス{len(unvisited)}件 / メモ: {self.scratchpad}")
                    return

                fid = data.get('target_frontier')
                if fid is not None and 1 <= int(fid) <= len(frs):
                    goal_global = frontier_goal(int(fid) - 1)
                    self.get_logger().info(f"LLMがフロンティアF{int(fid)}を選択")
                else:
                    tc = data.get('target_coordinate', data)
                    lx, ly = float(tc['target_x']), float(tc['target_y'])
                    candidates = unvisited + visited
                    nearest = min(candidates, key=lambda p: math.hypot(p[0] - lx, p[1] - ly))
                    if math.hypot(nearest[0] - lx, nearest[1] - ly) > 0.25:
                        raise ValueError(f"候補外の座標: ({lx}, {ly})")
                    goal_global = self.to_global(snap, (nearest[0], nearest[1]))
            except Exception as e:
                self.get_logger().error(f"推論エラー: {e} → ルールベースで代替")
                if frs and not unvisited:
                    goal_global = frontier_goal(0)         # 近くに未踏が無ければ最寄りのフロンティア
                else:
                    tx, ty = self.choose_fallback(unvisited, visited)
                    goal_global = self.to_global(snap, (tx, ty))

        self.history.append(
            f"({snap.x:.1f},{snap.y:.1f})から[{ttype}]目標odom({goal_global[0]:.1f},{goal_global[1]:.1f})を選択")
        self.get_logger().info(
            f"[{ttype}] 目標 odom({goal_global[0]:.2f}, {goal_global[1]:.2f})")

        with self.m_lock:
            stale = self.task['id'] != mid                 # 推論中に指示が変わった
        with self.lock:
            if not stale:
                self.goal = goal_global
                self.goal_id += 1
            self.plan_requested = False
        if stale:
            self.request_plan()


def main(args=None):
    rclpy.init(args=args)
    node = LLMMissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.is_running = False
        try:
            node.halt()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
