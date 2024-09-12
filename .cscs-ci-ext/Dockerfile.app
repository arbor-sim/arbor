ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY . /arbor.src

ARG NUM_PROCS
ARG CXX_FLAGS=""
ARG GPU=none
ARG GPU_ARCH=60

RUN echo ${CXX_FLAGS}

RUN mkdir -p /arbor.src/build \
  && cd /arbor.src/build \
  && cmake .. \
     -GNinja \
     -DCMAKE_INSTALL_PREFIX=/arbor.install \
     -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_TESTING=ON \
     -DARB_ARCH=none \
     -DARB_CXX_FLAGS_TARGET="${CXX_FLAGS}" \
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
