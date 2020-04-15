# Multistage build: here we import the current source code
# into build environment image, build the project, bundle it
# and then extract it into a small image that just contains
# the binaries we need to run

ARG BUILD_ENV

FROM $BUILD_ENV as builder

# Build arbor
COPY . /arbor

# Build and bundle binaries
RUN mkdir /arbor/build && cd /arbor/build && \
    CC=mpicc CXX=mpicxx cmake .. \
      -DARB_VECTORIZE=ON \
      -DARB_ARCH=broadwell \
      -DARB_WITH_PYTHON=OFF \
      -DARB_WITH_MPI=ON \
      -DARB_GPU=cuda-nvcc \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr && \
    make -j$(nproc) tests && \
    /root/libtree/libtree --chrpath --strip \
      -d /root/arbor.bundle \
      /arbor/build/bin/modcc \
      /arbor/build/bin/unit \
      /arbor/build/bin/unit-local \
      /arbor/build/bin/unit-modcc \
      /arbor/build/bin/unit-mpi && \
    rm -rf /arbor

FROM ubuntu:18.04

# This is the only thing necessary really from nvidia/cuda's ubuntu18.04 runtime image
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"

COPY --from=builder /root/arbor.bundle /root/arbor.bundle

# Make it easy to call our binaries.
ENV PATH="/root/arbor.bundle/usr/bin:$PATH"

RUN echo "/root/arbor.bundle/usr/lib/" > /etc/ld.so.conf.d/arbor.conf && ldconfig

WORKDIR /root/arbor.bundle/usr/bin

