ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG NUM_PROCS
ARG MICRO_ARCH
ARG GPU=none
ARG GPU_ARCH=60
COPY . /arbor.src

RUN mkdir -p /arbor.src/build \
  && cd /arbor.src/build \
  && cmake .. \
     -GNinja \
     -DCMAKE_INSTALL_PREFIX=/arbor.install \
     -DCMAKE_BUILD_TYPE=Release \
     -DARB_ARCH=$MICRO_ARCH \
     -DARB_WITH_ASSERTIONS=ON \
     -DARB_WITH_PROFILING=ON \
     -DARB_VECTORIZE=ON \
     -DARB_WITH_PYTHON=ON \
     -DARB_USE_HWLOC=ON \
     -DARB_WITH_MPI=ON \
     -DARB_GPU=$GPU\
     -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH \
     -DARB_USE_GPU_RNG=ON \
  && ninja -j${NUM_PROCS} tests examples pyarb \
  && ninja install
