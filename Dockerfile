FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -qq \
      build-essential cmake libomp-dev libopenmpi-dev openmpi-bin lcov git\
   && rm -rf /var/lib/apt/lists/*

WORKDIR /appdocker build -t nbody-dev .
