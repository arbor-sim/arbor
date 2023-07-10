ARG BASE_IMG
FROM $BASE_IMG

ARG NUM_PROCS

COPY . /arbor.src

RUN mkdir -p /arbor.src/build \
  && cd /arbor.src/build \
  && cmake .. \
     -GNinja \
     -DCMAKE_INSTALL_PREFIX=/arbor.install \
     -DCMAKE_BUILD_TYPE=Release \
     -DARB_WITH_ASSERTIONS=ON \
     -DARB_WITH_PROFILING=ON \
     -DARB_VECTORIZE=ON \
     -DARB_WITH_PYTHON=ON \
     -DARB_USE_HWLOC=ON \
     -DARB_WITH_MPI=ON \
     -DARB_GPU=cuda\
     -DCMAKE_CUDA_ARCHITECTURES=60
  && ninja -j${NUM_PROCS} tests examples pyarb \
  && ninja install

