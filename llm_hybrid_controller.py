"""
LLM × ハイブリッド2段構造 自律探索コントローラ (ROS2)

  大脳   : Gemini (フロンティア選択 / スクラッチパッド / 完了判定 / 情報要求)
  経路層 : 相対ローリンググリッド + 障害物膨張 + ダイクストラ(壁から離れる重み付き)
           → 直線化したウェイポイント列 (グリッド空間)
  制御層 : DWA (Dynamic Window Approach) で連続空間の滑らかな軌道を生成 (20Hz)

LLMの推論中もロボットは止まらず、ゴール手前(PREPLAN_DIST)で次の目標を先読みして
切り替える。目標はオドメトリ座標系で保持するので、推論遅延で座標がズレない。
"""
import heapq
import json
import math
import os
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from google import genai
from google.genai import types


def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LLMHybridController(Node):
    # ---- グリッド / 経路層 ------------------------------------------------
    CELL = 0.1                 # 細かいグリッド解像度 [m]
    MAP_HALF = 18              # ローカルマップ半幅 [セル] (=1.8m)
    INFLATE_R = 0.20           # 障害物膨張半径 [m]
    EXEMPT_R = 0.20            # ロボット中心付近は塞がない半径 [m]
    MIN_AREA = 6               # 0.5m角(25マス)中の最低到達可能マス数
    NEAR_PENALTY = 0.5         # 壁際セルの追加コスト (壁から離れた経路を好む)
    COARSE = [1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5]
    # ---- DWA / 制御層 -----------------------------------------------------
    CTRL_DT = 0.05             # 制御周期 [s] (20Hz)
    V_MAX = 0.20               # [m/s]
    W_MAX = 1.2                # [rad/s]
    A_V = 0.3                  # 加速度上限 [m/s^2]  (ウィンドウ幅 = A * WINDOW_DT)
    A_W = 1.5                  # 角加速度上限 [rad/s^2]
    WINDOW_DT = 0.1
    SIM_DT = 0.1               # 軌道シミュレーション刻み
    PRED_T = 1.6               # 予測時間 [s]
    NV, NW = 6, 17             # 速度サンプル数
    ROBOT_R = 0.11             # 衝突判定半径 [m]
    W_HEAD, W_PROG, W_CLEAR, W_VEL = 1.0, 1.5, 0.4, 0.3
    LOOKAHEAD = 0.4            # 経路上の目標点(carrot)の先読み距離 [m]
    GOAL_TOL = 0.15
    PREPLAN_DIST = 0.6         # ゴールまでこの距離になったら次の目標を先読み
    REPLAN_PERIOD = 0.5        # ローカル経路の再計画周期 [s]
    GOAL_TIMEOUT = 40.0
    STUCK_TIME = 5.0
    # --------------------------------------------------------------------

    def __init__(self):
        super().__init__('llm_hybrid_controller')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', self.odom_callback, qos_profile_sensor_data)

        self.lock = threading.Lock()
        self.is_running = True
        self.mission_done = False
        self.force_escape = False

        # センサ状態
        self.current_x = self.current_y = self.current_yaw = 0.0
        self.odom_count = 0
        self.visited_cells = set()
        self.local_obstacles = set()
        self.scan_xy = np.zeros((0, 2))
        self.scan_points = []
        self.scan_count = 0
        self.scan_rmax = 10.0

        # 目標 (オドメトリ座標系)
        self.goal = None
        self.goal_id = 0
        self.need_plan = threading.Event()
        self.plan_requested = True
        self.need_plan.set()

        # 制御層の状態 (実行スレッドのみが触る)
        self.ctl = SimpleNamespace(gid=None, path=None, last_plan=0.0, t0=0.0,
                                   stuck_ref=None, v=0.0, w=0.0)

        # LLMエージェントの状態
        self.mission = os.environ.get('MISSION', '迷路内の未踏エリアをすべて探索する')
        self.scratchpad = "(初回) まだ何も探索していない。"
        self.requested_info = []
        self.history = deque(maxlen=8)
        self.stuck_log = deque(maxlen=5)

        try:
            self.client = genai.Client()
        except Exception as e:
            self.get_logger().error(f"Gemini Clientの初期化失敗: {e}")
            raise
        self.model_name = os.environ.get('GEMINI_MODEL', 'gemini-3.1-flash-lite')
        self.get_logger().info("Hybrid LLM Controller (BFS/Dijkstra path + DWA) Started.")

        threading.Thread(target=self.execution_loop, daemon=True).start()
        threading.Thread(target=self.planner_loop, daemon=True).start()

    # ==================================================================
    # センサ
    # ==================================================================
    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        with self.lock:
            self.current_x, self.current_y, self.current_yaw = x, y, yaw
            self.odom_count += 1
            self.visited_cells.add((round(x * 2) / 2, round(y * 2) / 2))

    def scan_callback(self, msg):
        rmax = msg.range_max if math.isfinite(msg.range_max) else 10.0
        rmin = max(msg.range_min, 0.05)
        points, xy, obstacles = [], [], set()
        for i, r in enumerate(msg.ranges):
            if math.isnan(r):
                continue
            ang = normalize_angle(msg.angle_min + i * msg.angle_increment)
            if math.isinf(r) or r >= rmax:
                points.append((ang, rmax))
                continue
            if r < rmin:
                continue
            points.append((ang, r))
            if r <= 3.5:
                lx, ly = r * math.cos(ang), r * math.sin(ang)
                xy.append((lx, ly))
                obstacles.add((round(lx / self.CELL), round(ly / self.CELL)))
        arr = np.array(xy, dtype=float).reshape(-1, 2)
        with self.lock:
            self.scan_points = points
            self.scan_xy = arr
            self.local_obstacles = obstacles
            self.scan_count += 1
            self.scan_rmax = rmax

    def snapshot(self, with_visited=False):
        """姿勢とスキャンを同一時点でまとめて取得 (座標系のズレ防止)"""
        with self.lock:
            return SimpleNamespace(
                x=self.current_x, y=self.current_y, yaw=self.current_yaw,
                obstacles=self.local_obstacles, scan_xy=self.scan_xy,
                scan_points=self.scan_points, rmax=self.scan_rmax,
                visited=set(self.visited_cells) if with_visited else None)

    def wait_fresh_scan(self, n=2, timeout=1.5):
        with self.lock:
            start = self.scan_count
        t0 = time.time()
        while self.is_running and time.time() - t0 < timeout:
            with self.lock:
                if self.scan_count >= start + n:
                    return
            time.sleep(0.05)

    # 座標変換 -----------------------------------------------------------
    @staticmethod
    def to_local(s, g):
        dx, dy = g[0] - s.x, g[1] - s.y
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        return (c * dx + sn * dy, -sn * dx + c * dy)

    @staticmethod
    def to_global(s, p):
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        return (s.x + c * p[0] - sn * p[1], s.y + sn * p[0] + c * p[1])

    # ==================================================================
    # 経路層: グリッド (膨張 → BFS到達判定 → ダイクストラ経路)
    # ==================================================================
    def build_inflated(self, obstacles):
        r_cells = int(self.INFLATE_R / self.CELL + 1e-9)
        offsets = [(dx, dy)
                   for dx in range(-r_cells, r_cells + 1)
                   for dy in range(-r_cells, r_cells + 1)
                   if math.hypot(dx, dy) * self.CELL <= self.INFLATE_R + 1e-9]
        exempt = self.EXEMPT_R / self.CELL
        inflated = set()
        for ox, oy in obstacles:
            for dx, dy in offsets:
                c = (ox + dx, oy + dy)
                if math.hypot(c[0], c[1]) > exempt:   # 中心付近は塞がない (窒息防止)
                    inflated.add(c)
        return inflated

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def bfs_reachable(self, inflated):
        H = self.MAP_HALF
        reachable = {(0, 0)}
        dq = deque([(0, 0)])
        while dq:
            cx, cy = dq.popleft()
            for dx, dy in self._DIRS:
                n = (cx + dx, cy + dy)
                if abs(n[0]) > H or abs(n[1]) > H or n in reachable or n in inflated:
                    continue
                if dx != 0 and dy != 0 and ((cx + dx, cy) in inflated or (cx, cy + dy) in inflated):
                    continue
                reachable.add(n)
                dq.append(n)
        return reachable

    def create_local_grid(self, snap):
        inflated = self.build_inflated(snap.obstacles)
        reachable = self.bfs_reachable(inflated)
        self.last_reachable = reachable        # サブクラス(ミッション層)が到達判定に使う

        unvisited, visited_list = [], []
        grid_str = "【360度 フロンティア探索マップ】\n"
        grid_str += "※ X(+)前方, Y(+)左。(数値)は25マス中の安全面積スコア。\n\n"
        c, s = math.cos(snap.yaw), math.sin(snap.yaw)
        for cx in self.COARSE:
            row = f"X={cx:>4.1f}m | "
            for cy in self.COARSE:
                if cx == 0.0 and cy == 0.0:
                    row += "   [R:現在地]   "
                    continue
                ci, cj = round(cx / self.CELL), round(cy / self.CELL)
                count = sum(1 for dx in range(-2, 3) for dy in range(-2, 3)
                            if (ci + dx, cj + dy) in reachable)
                if count >= self.MIN_AREA:
                    gx = snap.x + cx * c - cy * s
                    gy = snap.y + cx * s + cy * c
                    if (round(gx * 2) / 2, round(gy * 2) / 2) in snap.visited:
                        row += f"[Y={cy:>4.1f}済({count:>2})]"
                        visited_list.append((cx, cy, count))
                    else:
                        row += f"[Y={cy:>4.1f}未({count:>2})]"
                        unvisited.append((cx, cy, count))
                else:
                    row += f"  [障害物({count:>2})] "
            grid_str += row + "\n"
        return grid_str, unvisited, visited_list

    def plan_path(self, snap, goal_local):
        """ロボット座標系で goal_local へのウェイポイント列を返す。
        到達不能なら到達可能セルのうち目標に最も近い点までの経路を返す。"""
        inflated = self.build_inflated(snap.obstacles)
        near = set()
        for (x, y) in inflated:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    near.add((x + dx, y + dy))
        near -= inflated

        H = self.MAP_HALF
        gx, gy = goal_local[0] / self.CELL, goal_local[1] / self.CELL
        m = max(abs(gx), abs(gy))
        if m > H - 1:                      # マップ外のゴールは境界へ射影
            gx, gy = gx * (H - 1) / m, gy * (H - 1) / m
        goal = (round(gx), round(gy))

        dist = {(0, 0): 0.0}
        parent = {}
        heap = [(0.0, 0, (0, 0))]
        cnt = 0
        while heap:
            d, _, cur = heapq.heappop(heap)
            if d > dist.get(cur, math.inf):
                continue
            if cur == goal:
                break
            for dx, dy in self._DIRS:
                n = (cur[0] + dx, cur[1] + dy)
                if abs(n[0]) > H or abs(n[1]) > H or n in inflated:
                    continue
                if dx != 0 and dy != 0 and ((cur[0] + dx, cur[1]) in inflated or
                                            (cur[0], cur[1] + dy) in inflated):
                    continue
                step = 1.4142 if (dx and dy) else 1.0
                nd = d + step * (1.0 + (self.NEAR_PENALTY if n in near else 0.0))
                if nd < dist.get(n, math.inf):
                    dist[n] = nd
                    parent[n] = cur
                    cnt += 1
                    heapq.heappush(heap, (nd, cnt, n))

        end = goal if goal in dist else min(dist, key=lambda c: math.hypot(c[0] - gx, c[1] - gy))
        cells = [end]
        while cells[-1] in parent:
            cells.append(parent[cells[-1]])
        cells.reverse()

        # 見通しの良い区間をまとめて直線化 (ウェイポイント化)
        def los(a, b):
            n = max(1, int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2))
            for k in range(n + 1):
                t = k / n
                c = (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t))
                if c in inflated:
                    return False
            return True

        pruned = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not los(cells[i], cells[j]):
                j -= 1
            pruned.append(cells[j])
            i = j
        return [(c[0] * self.CELL, c[1] * self.CELL) for c in pruned]

    # ==================================================================
    # 制御層: carrot (経路上の先読み点) + DWA
    # ==================================================================
    @staticmethod
    def carrot_local(path_local, lookahead):
        """経路(ロボット座標系)上で、最近点から lookahead だけ先の点"""
        if len(path_local) == 1:
            return path_local[0]
        best = (math.inf, 0, 0.0)
        for i in range(len(path_local) - 1):
            ax, ay = path_local[i]
            bx, by = path_local[i + 1]
            abx, aby = bx - ax, by - ay
            l2 = abx * abx + aby * aby
            t = 0.0 if l2 < 1e-9 else clamp(-(ax * abx + ay * aby) / l2, 0.0, 1.0)
            d = math.hypot(ax + t * abx, ay + t * aby)
            if d < best[0]:
                best = (d, i, t)
        _, i, t = best
        ax, ay = path_local[i]
        bx, by = path_local[i + 1]
        cur = (ax + t * (bx - ax), ay + t * (by - ay))
        rem = lookahead
        while True:
            bx, by = path_local[i + 1]
            seg = math.hypot(bx - cur[0], by - cur[1])
            if seg >= rem and seg > 1e-9:
                k = rem / seg
                return (cur[0] + (bx - cur[0]) * k, cur[1] + (by - cur[1]) * k)
            rem -= seg
            cur = (bx, by)
            i += 1
            if i >= len(path_local) - 1:
                return path_local[-1]

    def dwa(self, snap, carrot, v_cur, w_cur):
        """ダイナミックウィンドウ内の(v,w)を軌道ロールアウトで評価し最良を返す。
        有効な軌道が無ければ None。"""
        wdt = self.WINDOW_DT
        v_lo = max(0.0, v_cur - self.A_V * wdt)
        v_hi = min(self.V_MAX, v_cur + self.A_V * wdt)
        w_lo = max(-self.W_MAX, w_cur - self.A_W * wdt)
        w_hi = min(self.W_MAX, w_cur + self.A_W * wdt)
        V, W = np.meshgrid(np.linspace(v_lo, v_hi, self.NV),
                           np.linspace(w_lo, w_hi, self.NW))
        V, W = V.ravel(), W.ravel()
        n = V.size

        steps = max(1, int(self.PRED_T / self.SIM_DT))
        x = np.zeros(n); y = np.zeros(n); th = np.zeros(n)
        X = np.empty((n, steps)); Y = np.empty((n, steps))
        for k in range(steps):
            th = th + W * self.SIM_DT
            x = x + V * np.cos(th) * self.SIM_DT
            y = y + V * np.sin(th) * self.SIM_DT
            X[:, k], Y[:, k] = x, y

        obs = snap.scan_xy
        if obs.size:
            obs = obs[np.hypot(obs[:, 0], obs[:, 1]) < 2.5]
        if len(obs) > 300:
            obs = obs[::int(math.ceil(len(obs) / 300))]
        if len(obs):
            d2 = (X[:, :, None] - obs[None, None, :, 0]) ** 2 + \
                 (Y[:, :, None] - obs[None, None, :, 1]) ** 2
            dmin = np.sqrt(d2.min(axis=(1, 2)))
        else:
            dmin = np.full(n, 3.0)

        # 衝突しない & 手前で止まれる速度のみ有効
        valid = (dmin > self.ROBOT_R) & \
                (V <= np.sqrt(2.0 * self.A_V * np.maximum(dmin - self.ROBOT_R, 0.0)) + 1e-9)
        if not valid.any():
            return None

        cx, cy = carrot
        ang = np.arctan2(cy - y, cx - x)
        err = np.abs(np.arctan2(np.sin(ang - th), np.cos(ang - th)))
        heading = 1.0 - err / math.pi
        d0 = math.hypot(cx, cy)
        df = np.hypot(cx - x, cy - y)
        progress = np.clip((d0 - df) / (self.V_MAX * self.PRED_T), -1.0, 1.0)
        clear = np.minimum(dmin, 1.0)
        score = (self.W_HEAD * heading + self.W_PROG * progress +
                 self.W_CLEAR * clear + self.W_VEL * V / self.V_MAX)
        score[~valid] = -np.inf
        k = int(np.argmax(score))
        return float(V[k]), float(W[k])

    # ==================================================================
    # 実行スレッド (20Hz)
    # ==================================================================
    def halt(self):
        self.publisher_.publish(Twist())
        self.ctl.v = self.ctl.w = 0.0

    def request_plan(self):
        with self.lock:
            if self.plan_requested:
                return
            self.plan_requested = True
        self.need_plan.set()

    def finish_goal(self, gid, note='到達'):
        with self.lock:
            if self.goal_id != gid:
                return
            self.goal = None
        self.halt()
        self.history.append(f"目標{gid}: {note}")
        self.request_plan()

    def fail(self, gid, reason):
        with self.lock:
            if self.goal_id != gid:
                return
            self.goal = None
        self.halt()
        self.get_logger().warn(f"【失敗】目標{gid}: {reason} → 脱出モードへ")
        self.stuck_log.append(f"目標{gid}: {reason}")
        self.history.append(f"目標{gid}: 失敗({reason})")
        self.force_escape = True
        self.recover()
        self.request_plan()

    def recover(self):
        """最も空いている方向へその場で向き直る (円形ロボットなので安全)"""
        snap = self.snapshot()
        ex, ey = self.get_escape_target(snap)
        self.turn_to(snap.yaw + math.atan2(ey, ex))

    def turn_to(self, target_yaw):
        _, _, yaw = self.get_pose()
        timeout = abs(normalize_angle(target_yaw - yaw)) / 0.15 + 1.0
        t0 = time.time()
        while self.is_running and time.time() - t0 < timeout:
            _, _, yaw = self.get_pose()
            err = normalize_angle(target_yaw - yaw)
            if abs(err) < 0.05:
                break
            w = clamp(1.5 * err, -0.5, 0.5)
            if abs(w) < 0.15:
                w = math.copysign(0.15, err)
            tw = Twist()
            tw.angular.z = w
            self.publisher_.publish(tw)
            time.sleep(0.05)
        self.halt()
        time.sleep(0.15)

    def get_pose(self):
        with self.lock:
            return self.current_x, self.current_y, self.current_yaw

    def control_tick(self, now):
        c = self.ctl
        with self.lock:
            goal, gid = self.goal, self.goal_id
        if goal is None:
            if c.v != 0.0 or c.w != 0.0:
                self.halt()
            c.path = None
            return

        snap = self.snapshot()
        if gid != c.gid:                       # 新しい目標に切り替え (滑らかに継続)
            c.gid, c.path, c.t0 = gid, None, now
            c.stuck_ref = (now, snap.x, snap.y, snap.yaw)

        gl = self.to_local(snap, goal)
        rem = math.hypot(gl[0], gl[1])
        if rem < self.GOAL_TOL:
            self.finish_goal(gid)
            return
        if rem < self.PREPLAN_DIST:
            self.request_plan()                # 止まる前に次を考えさせる
        if now - c.t0 > self.GOAL_TIMEOUT:
            self.fail(gid, 'タイムアウト')
            return

        t_ref, x_ref, y_ref, yaw_ref = c.stuck_ref
        if now - t_ref > self.STUCK_TIME:
            moved = math.hypot(snap.x - x_ref, snap.y - y_ref)
            turned = abs(normalize_angle(snap.yaw - yaw_ref))
            if moved < 0.08 and turned < 0.3:
                self.fail(gid, 'スタック')
                return
            c.stuck_ref = (now, snap.x, snap.y, snap.yaw)

        # ローカル経路: 周期的に最新のLiDARで再計画
        if c.path is None or now - c.last_plan > self.REPLAN_PERIOD:
            pts = self.plan_path(snap, gl)
            c.last_plan = now
            length = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                         for i in range(len(pts) - 1))
            if length < 0.1:                   # これ以上進めない
                if rem < 0.4:
                    self.finish_goal(gid, '到達(壁際で打ち切り)')
                else:
                    self.fail(gid, '経路なし')
                return
            c.path = [self.to_global(snap, p) for p in pts]

        path_local = [self.to_local(snap, p) for p in c.path]
        carrot = self.carrot_local(path_local, self.LOOKAHEAD)
        cmd = self.dwa(snap, carrot, c.v, c.w)
        if cmd is None:
            self.fail(gid, '有効な軌道なし')
            return
        tw = Twist()
        tw.linear.x, tw.angular.z = cmd
        self.publisher_.publish(tw)
        c.v, c.w = cmd

    def execution_loop(self):
        while self.is_running:
            t0 = time.time()
            try:
                self.control_tick(t0)
            except Exception as e:
                self.get_logger().error(f"制御エラー: {e}")
                try:
                    self.halt()
                except Exception:
                    pass
            time.sleep(max(0.0, self.CTRL_DT - (time.time() - t0)))

    # ==================================================================
    # 大脳: LLMプランナー (スクラッチパッド / 完了判定 / 情報要求)
    # ==================================================================
    def get_escape_target(self, snap):
        best_dist, best_ang = 0.0, 0.0
        for k in range(12):
            center = normalize_angle(math.radians(k * 30))
            vals = [r for a, r in snap.scan_points
                    if abs(normalize_angle(a - center)) <= math.radians(10)]
            clearance = min(vals) if vals else 0.0
            if clearance > best_dist:
                best_dist, best_ang = clearance, center
        d = clamp(best_dist - 0.25, 0.2, 0.5)
        return round(d * math.cos(best_ang), 2), round(d * math.sin(best_ang), 2)

    @staticmethod
    def fallback_choice(unvisited, visited):
        pool = unvisited if unvisited else visited
        best = max(pool, key=lambda p: (p[0], -abs(p[1]), p[2]))
        return best[0], best[1]

    def ask_llm(self, grid_context, unvisited, visited):
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
{grid_context}

【優先して狙うべき [未] (未踏) 座標リスト】 (x, y, 安全面積スコア)
{list(unvisited)}

【すでに通った [済] (探索済) 座標リスト】 (x, y, 安全面積スコア)
{list(visited)}

【ルール】
1. 【未踏】に前方(Xが正)のマスがあれば最優先。スコアが大きいほど安全。
2. 【未踏】が空/後方のみなら【探索済】から前方または左右の安全なマスを選ぶ。
3. なるべく遠く(1.0m〜1.5m)を選ぶ。必ずリスト内の (x, y) をそのまま選ぶ。
4. 【未踏】リストが空で、ミッションが達成されたと判断したら is_completed を true にする。
5. thought_scratchpad に「現在の仮説・次にやる予定」を短く(200字以内)書く。次のあなたに引き継がれる。
6. 次回さらに情報が必要なら requested_next_info に "history" / "stuck_log" を入れる(不要なら空配列)。
7. 出力はJSONのみ。

フォーマット例:
{{"thought_scratchpad": "右の広い空間を探索中。次はX=1.5の未踏を攻める。",
  "requested_next_info": [],
  "target_x": 1.0, "target_y": 0.5,
  "is_completed": false}}
"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.0),
        )
        data = json.loads(response.text.strip())
        if isinstance(data, list):
            data = data[0]
        return data

    def planner_loop(self):
        time.sleep(1.0)
        while self.is_running and not self.mission_done:
            if not self.need_plan.wait(0.2):
                continue
            self.need_plan.clear()
            with self.lock:
                ready = self.scan_count > 0 and self.odom_count > 0
            if not ready:
                time.sleep(0.5)
                self.need_plan.set()
                continue
            try:
                self.plan_once()
            except Exception as e:
                self.get_logger().error(f"プランナーエラー: {e}")
                with self.lock:
                    self.plan_requested = False
                time.sleep(1.0)
                self.request_plan()

    def plan_once(self):
        with self.lock:
            idle = self.goal is None
        if idle:                                   # 停止直後は古いスキャンを避ける
            self.wait_fresh_scan()
        snap = self.snapshot(with_visited=True)
        grid_context, unvisited, visited = self.create_local_grid(snap)
        print("\n" + grid_context)

        if self.force_escape or (not unvisited and not visited):
            tx, ty = self.get_escape_target(snap)
            self.force_escape = False
            self.get_logger().warn(f"脱出移動: X={tx}m, Y={ty}m")
        else:
            candidates = unvisited + visited
            try:
                data = self.ask_llm(grid_context, unvisited, visited)
                tc = data.get('target_coordinate', data)
                lx, ly = float(tc['target_x']), float(tc['target_y'])

                self.scratchpad = str(data.get('thought_scratchpad', self.scratchpad))[:400]
                req = data.get('requested_next_info', [])
                self.requested_info = [req] if isinstance(req, str) else list(req)

                # 完了判定は Python 側でも検証 (未踏が残っているのに完了させない)
                if data.get('is_completed') is True and not unvisited:
                    self.get_logger().info(f"★ミッション完了★ メモ: {self.scratchpad}")
                    self.mission_done = True
                    with self.lock:
                        self.goal = None
                        self.plan_requested = False
                    self.halt()
                    return

                nearest = min(candidates, key=lambda p: math.hypot(p[0] - lx, p[1] - ly))
                if math.hypot(nearest[0] - lx, nearest[1] - ly) > 0.25:
                    raise ValueError(f"候補外の座標: ({lx}, {ly})")
                tx, ty = nearest[0], nearest[1]
            except Exception as e:
                self.get_logger().error(f"推論エラー: {e} → ルールベースで代替")
                tx, ty = self.fallback_choice(unvisited, visited)

        goal_global = self.to_global(snap, (tx, ty))   # 推論中に動いても座標は不変
        self.get_logger().info(
            f"決定した目標: ローカル(X={tx}, Y={ty}) → odom({goal_global[0]:.2f}, {goal_global[1]:.2f})")
        self.history.append(
            f"({snap.x:.1f},{snap.y:.1f})から目標(X={tx:+.1f},Y={ty:+.1f})を選択")
        with self.lock:
            self.goal = goal_global
            self.goal_id += 1
            self.plan_requested = False


def main(args=None):
    rclpy.init(args=args)
    node = LLMHybridController()
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
