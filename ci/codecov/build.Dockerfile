FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /root

ARG MPICH_VERSION=3.3.2

ENV DEBIAN_FRONTEND noninteractive
ENV FORCE_UNSAFE_CONFIGURE 1
ENV MPICH_VERSION ${MPICH_VERSION}

# Install basic tools
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    python3 \
    git tar wget curl \
    gcc-8 g++-8 make && \
    update-alternatives \
        --install /usr/bin/gcc gcc /usr/bin/gcc-8 60 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-8 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-8 && \
    update-alternatives --config gcc && \
    rm -rf /var/lib/apt/lists/*

RUN cd /usr/local/bin && \
    curl -Ls https://codecov.io/bash > codecov.sh && \
    echo "89c658e261d5f25533598a222fd96cf17a5fa0eb3772f2defac754d9970b2ec8 codecov.sh" | sha256sum --check --quiet && \
    chmod +x codecov.sh

RUN wget -q "https://github.com/linux-test-project/lcov/archive/v1.15.tar.gz" && \
    echo "d88b0718f59815862785ac379aed56974b9edd8037567347ae70081cd4a3542a v1.15.tar.gz" | sha256sum --check --quiet && \
    tar -xzf v1.15.tar.gz && \
    cd lcov-1.15 && \
    make install -j$(nproc) && \
    rm -rf lcov-1.15 v1.15.tar.gz

# Install MPICH ABI compatible with Cray's lib on Piz Daint
RUN wget -q https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz -O mpich.tar.gz && \
    echo "4bfaf8837a54771d3e4922c84071ef80ffebddbb6971a006038d91ee7ef959b9 mpich.tar.gz" | sha256sum --check --quiet && \
    tar -xzf mpich.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure --disable-fortran && \
    make install -j$(nproc) && \
    rm -rf mpich.tar.gz mpich-${MPICH_VERSION}

# Install cmake
RUN wget -q "https://github.com/Kitware/CMake/releases/download/v3.12.4/cmake-3.12.4-Linux-x86_64.tar.gz" -O cmake.tar.gz && \
    echo "486edd6710b5250946b4b199406ccbf8f567ef0e23cfe38f7938b8c78a2ffa5f cmake.tar.gz" | sha256sum --check --quiet && \
    tar --strip-components=1 -xzf cmake.tar.gz -C /usr/local && \
    rm -rf cmake.tar.gz

# Install bundle tooling for creating small Docker images
RUN wget -q https://github.com/haampie/libtree/releases/download/v1.2.0/libtree_x86_64.tar.gz && \
    echo "4316a52aed7c8d2f7d2736c935bbda952204be92e56948110a143283764c427c libtree_x86_64.tar.gz" | sha256sum --check --quiet && \
    tar -xzf libtree_x86_64.tar.gz && \
    rm libtree_x86_64.tar.gz && \
    ln -s /root/libtree/libtree /usr/local/bin/libtree
