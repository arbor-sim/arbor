ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG NUM_PROCS

COPY . /arbor.src

RUN mkdir -p /arbor.src/build \
  && cd /arbor.src/build \
  && cmake .. \
     -DGNinja \
     -DCMAKE_BUILD_TYPE=Release \
     -DARB_WITH_ASSERTIONS=ON \
     -DARB_VECTORIZE=ON \
  && ninja -j${NUM_PROCS} tests examples pyarb
    
     
