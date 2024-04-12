# Multistage build: here we import the current source code
# into build environment image, build the project, bundle it
# and then extract it into a small image that just contains
# the binaries we need to run

ARG BUILD_ENV

ARG SOURCE_DIR=/arbor
ARG BUILD_DIR=/arbor-build
ARG BUNDLE_DIR=/root/arbor.bundle

FROM $BUILD_ENV as builder

ARG SOURCE_DIR
ARG BUILD_DIR
ARG BUNDLE_DIR

# Build arbor
COPY . $SOURCE_DIR

# Build and bundle binaries
RUN mkdir ${BUILD_DIR} && cd ${BUILD_DIR} && \
    CC=mpicc CXX=mpicxx cmake ${SOURCE_DIR} \
      -DARB_VECTORIZE=ON \
      -DARB_ARCH=broadwell \
      -DARB_WITH_PYTHON=OFF \
      -DARB_WITH_MPI=ON \
      -DARB_GPU=cuda \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr && \
    make -j$(nproc) tests && \
    libtree --chrpath --strip \
      -d ${BUNDLE_DIR} \
      ${BUILD_DIR}/bin/modcc \
      ${BUILD_DIR}/bin/unit \
      ${BUILD_DIR}/bin/unit-local \
      ${BUILD_DIR}/bin/unit-modcc \
      ${BUILD_DIR}/bin/unit-mpi && \
    rm -rf ${BUILD_DIR}

# Only keep the sources for tests, not the git history
RUN rm -rf ${SOURCE_DIR}/.git

FROM ubuntu:18.04

ARG SOURCE_DIR
ARG BUNDLE_DIR

# This is the only thing necessary really from nvidia/cuda's ubuntu18.04 runtime image
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"

COPY --from=builder ${BUNDLE_DIR} ${BUNDLE_DIR}
COPY --from=builder ${SOURCE_DIR} ${SOURCE_DIR}

# Make it easy to call our binaries.
ENV PATH="${BUNDLE_DIR}/usr/bin:$PATH"

# Automatically print stacktraces on segfault
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libSegFault.so

RUN echo "${BUNDLE_DIR}/usr/lib/" > /etc/ld.so.conf.d/arbor.conf && ldconfig

WORKDIR ${BUNDLE_DIR}/usr/bin

