FROM ubuntu:22.04

# Install libraries from apt
RUN apt-get update && apt-get -y install software-properties-common ca-certificates
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && apt-get update 
RUN apt-get -y install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf \
    build-essential curl gcc-9 g++-9 pkg-config psmisc unzip \
    python3 python3-pip python-is-python3 wget git vim
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
              --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
              --slave /usr/bin/gcov gcov /usr/bin/gcov-9

# Install requirements
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt && rm requirements.txt

# Install bazel
RUN git clone https://github.com/IntelliSys-Lab/Nitro-ray.git ~/ray
RUN cd ~/ray/ci/env && ./install-bazel.sh

# Install ray
RUN cd ~/ray/python && pip3 install .

# Install mujoco
RUN cd ~/ && wget "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz" && \
    tar -xvf mujoco210-linux-x86_64.tar.gz && \
    mkdir ~/.mujoco && \
    mv ./mujoco210 ~/.mujoco/mujoco210 && \
    rm mujoco210-linux-x86_64.tar.gz
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin" >> ~/.bashrc

# Set head node address
ENV RAY_HEAD_PORT="6379"

# Start from entrypoint
COPY entrypoint.sh ~/entrypoint.sh
ENTRYPOINT ["~/entrypoint.sh"]
