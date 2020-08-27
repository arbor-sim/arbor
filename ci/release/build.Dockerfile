FROM nvidia/cuda:10.1-devel-ubuntu18.04

WORKDIR /root

ARG MPICH_VERSION=3.3.2

ENV DEBIAN_FRONTEND noninteractive
ENV FORCE_UNSAFE_CONFIGURE 1
ENV MPICH_VERSION ${MPICH_VERSION}

# Install basic tools
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    build-essential \
    python3 \
    git tar wget curl && \
    rm -rf /var/lib/apt/lists/*

# Install cmake
RUN wget -qO- "https://github.com/Kitware/CMake/releases/download/v3.12.4/cmake-3.12.4-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

# Install MPICH ABI compatible with Cray's lib on Piz Daint
RUN wget -q https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar -xzf mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure --disable-fortran && \
    make install -j$(nproc) && \
    rm -rf mpich-${MPICH_VERSION}.tar.gz mpich-${MPICH_VERSION}

# Install bundle tooling for creating small Docker images
RUN wget -q https://github.com/haampie/libtree/releases/download/v1.2.0/libtree_x86_64.tar.gz && \
    tar -xzf libtree_x86_64.tar.gz && \
    rm libtree_x86_64.tar.gz && \
    ln -s /root/libtree/libtree /usr/local/bin/libtree
