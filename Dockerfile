FROM osrf/ros:humble-desktop

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir google-genai

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

COPY . /workspace/

CMD ["bash"]