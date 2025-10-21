 
# From Jacobus G.M. van der Linden “STreeD”
# https://github.com/AlgTUDelft/pystreed 


FROM gcc:9.4

# https://github.com/Rikorose/gcc-cmake/blob/master/Dockerfile
ARG CMAKE_VERSION=3.23.1
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh

ENV PATH="/usr/bin/cmake/bin:${PATH}"

RUN mkdir /SORTD
WORKDIR /SORTD
ADD data /SORTD/data/
ADD include /SORTD/include/
ADD src /SORTD/src/
ADD test /SORTD/test/
ADD CMakeLists.txt /SORTD/

RUN mkdir build
WORKDIR /SORTD/build 
RUN cmake ..
RUN cmake --build .
RUN ctest
WORKDIR /SORTD 


