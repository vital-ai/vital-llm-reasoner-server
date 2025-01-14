FROM python:3.11-slim
LABEL maintainer="marc@vital.ai"
LABEL description="Vital LLM Reasoner Server"

# all implementing code should be in vital-llm-reasoner
# other than perhaps thin wrapper for the vllm server

# This is set up to compile for Mac OS for local testing
# but should be compiled in linux and nvidia environment for actual use

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    ninja-build \
    clang \
    git \
    wget \
    software-properties-common \
    libomp-dev \
    python3-dev \
    libopenblas-dev \
    && apt-get clean

# this was to get a newer cmake
RUN echo "deb http://deb.debian.org/debian bookworm-backports main" >> /etc/apt/sources.list && \
    apt-get update && apt-get -t bookworm-backports install -y cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN cmake --version

RUN pip install --no-cache-dir --upgrade pip

# trying g++ instead of clang
RUN apt-get install -y gcc g++ && \
    export CC=gcc CXX=g++

# these seemed to need to be installed for vllm to compile
RUN pip install --no-cache-dir numpy

# RUN apt-get update && apt-get install -y libomp-dev
# RUN apt-get update && apt-get install -y libnuma-dev

# these seem to be ignored
ENV VLLM_USE_NUMA=0
ENV VLLM_USE_BF16=0
ENV VLLM_USE_CPU_EXTENSION=0
ENV VLLM_USE_OPENMP=0

ENV VLLM_TARGET_DEVICE=cpu

# ENV CC=clang
# ENV CXX=clang++
# ENV CFLAGS="-fopenmp -I/usr/lib/llvm-12/include"
# ENV LDFLAGS="-fopenmp -L/usr/lib/llvm-12/lib"

# ENV CXXFLAGS="-march=armv8-a"
# ENV CXXFLAGS="${CXXFLAGS} -std=c++14"

# RUN pip install --use-deprecated=legacy-resolver --no-cache-dir vllm==0.6.6
# RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y libnuma-dev

RUN python3.11 -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu && \
    python3.11 -c "import torch; print(torch.__version__)" && \
    git clone https://github.com/vllm-project/vllm.git /opt/vllm && \
    cd /opt/vllm && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_CXX_FLAGS="-std=c++17" \
        -DVLLM_USE_NUMA=OFF \
        -DVLLM_USE_BF16=OFF \
        -DVLLM_USE_CPU_EXTENSION=OFF \
        -DVLLM_USE_OPENMP=OFF \
        -DVLLM_TARGET_DEVICE=cpu \
        -DVLLM_PYTHON_EXECUTABLE=$(which python3.11) && \
    cmake --build . --target install -j$(nproc) && \
    cd /opt/vllm && \
    pip install .

# RUN pip install "vital-llm-reasoner[vllm]>=0.0.2" vllm==0.6.6.post2.dev214+g87054a57.cpu

RUN VLLM_VERSION=$(python3.11 -m pip show vllm | grep '^Version:' | awk '{print $2}') && \
    echo "Installed vllm version: $VLLM_VERSION" && \
    python3.11 -m pip install "vital-llm-reasoner[vllm]>=0.0.2" "vllm==$VLLM_VERSION"

# RUN pip install scikit-build-core

# RUN pip install scikit-learn

# ENV CFLAGS="-mcpu=generic"
# ENV CXXFLAGS="-mcpu=generic"
# RUN pip install llama-cpp-python --no-build-isolation

# ENV GGML_CPU_BACKEND=OFF
# RUN pip install llama-cpp-python --no-build-isolation

# Clone llama-cpp-python
# RUN git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git /tmp/llama-cpp-python

# Build with optimizations disabled
# RUN CMAKE_ARGS="-DGGML_CPU_BACKEND=OFF -DGGML_DISABLE_SVE=ON" \
#    pip install /tmp/llama-cpp-python

# llama.cpp was removed
# the below worked to install llama.cpp
# native OFF was important
# ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_BACKEND=ON -DGGML_DISABLE_SVE=ON -DGGML_DISABLE_I8MM=ON -DCMAKE_C_FLAGS=-mcpu=cortex-a72 -DCMAKE_CXX_FLAGS=-mcpu=cortex-a72"
# ENV CFLAGS="-mcpu=cortex-a72"
# ENV CXXFLAGS="-mcpu=cortex-a72"

# Clone the llama-cpp-python repository
# WORKDIR /tmp
# RUN git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git

# Install the package
# WORKDIR /tmp/llama-cpp-python
# RUN pip install scikit-build cmake ninja
# RUN CFLAGS="-mcpu=cortex-a72" CXXFLAGS="-mcpu=cortex-a72" pip install .

# Clean up
# RUN rm -rf /tmp/llama-cpp-python

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8033

CMD ["uvicorn", "vital_llm_reasoner_server.main:app", "--host", "0.0.0.0", "--port", "8033"]
