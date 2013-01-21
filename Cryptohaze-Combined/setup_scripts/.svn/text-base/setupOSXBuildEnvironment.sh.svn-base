#!/bin/bash

CUDA_DOWNLOAD_URL=http://developer.download.nvidia.com/compute/cuda/5_0/rel-update-1/installers/cuda_5.0.36_macos.pkg
CUDA_PKG_NAME=cuda_5.0.36_macos.pkg

CMAKE_DOWNLOAD_URL=http://www.cmake.org/files/v2.8/cmake-2.8.10.2-Darwin64-universal.dmg
CMAKE_IMAGE_NAME=cmake-2.8.10.2-Darwin64-universal

BOOST_DOWNLOAD_URL=http://downloads.sourceforge.net/project/boost/boost/1.52.0/boost_1_52_0.tar.bz2
BOOST_DIRECTORY_NAME=boost_1_52_0

PROTOBUF_DOWNLOAD_URL=http://protobuf.googlecode.com/files/protobuf-2.4.1.tar.bz2
PROTOBUF_DIRECTORY_NAME=protobuf-2.4.1

# Store the current directory
BUILD_BASE_DIRECTORY=`pwd`

# Create downloads directory for things being installed
mkdir downloads

# Check for dependencies
# CUDA first.
if [ ! -f /usr/local/cuda/bin/nvcc ]; then
  echo "CUDA toolkit is not installed.  Downloading, please install."
  cd downloads
  curl -O -L "$CUDA_DOWNLOAD_URL"
  open -W $CUDA_PKG_NAME
  if [ ! -f /usr/local/cuda/bin/nvcc ]; then
    echo "CUDA toolkit still not installed.  Sorry."
    DEP_ERRORS=1
  fi
  cd ..
fi

# cmake next
if [ ! -f /usr/bin/cmake ]; then
  echo "CMAKE binaries are not installed.  Downloading, please install from the dmg."
  echo "YOU MUST INSTALL THE COMMAND LINE TOOLS TOO!"
  cd downloads
  curl -O -L "$CMAKE_DOWNLOAD_URL"
  open -W "$CMAKE_IMAGE_NAME.dmg"
  open -W /Volumes/$CMAKE_IMAGE_NAME/$CMAKE_IMAGE_NAME.pkg
  if [ ! -f /usr/bin/cmake ]; then
    echo "CMAKE still not installed.  Sorry."
    DEP_ERRORS=1
  fi
fi

if [ "$DEP_ERRORS" == 1 ]; then
echo "Dependencies not installed.  Exiting."
exit 1
fi



# Download boost - use latest URL.
cd downloads
curl -L -o boost_latest.tar.bz2 "$BOOST_DOWNLOAD_URL"

# Go back to root and unzip boost
cd ..
echo "Unpacking boost..."
tar -xjf downloads/boost_latest.tar.bz2

# Change to directory and start the build
cd $BOOST_DIRECTORY_NAME
./bootstrap.sh
./b2 install --prefix="$BUILD_BASE_DIRECTORY" --without-mpi -j4
cd ..

# Boost, in theory, is installed.
if [ ! -f lib/libboost_thread.dylib ]; then
echo "Boost did not install properly - please look for errors."
exit 1
fi

# Grab the protobuf libraries and install them.
cd downloads
curl -L -o protobuf_latest.tar.bz2 "$PROTOBUF_DOWNLOAD_URL"
cd ..
echo "Unpacking protobuf..."
tar -xjf downloads/protobuf_latest.tar.bz2

# Unzip & build protobuf
cd $PROTOBUF_DIRECTORY_NAME
./configure --prefix "$BUILD_BASE_DIRECTORY"
make && make install
cd ..

# Check for protobuf
if [ ! -f lib/libprotobuf.dylib ]; then
echo "Protobuf did not install properly - please look for errors."
exit 1
fi

# Get the source tree
svn co https://cryptohaze.svn.sourceforge.net/svnroot/cryptohaze/Cryptohaze-Combined Cryptohaze-Dev

# Build argtable2
cd Cryptohaze-Dev/dependencies/argtable2-13/
./configure --prefix="$BUILD_BASE_DIRECTORY"
make && make install
cd ../../..

# Set up cmake
cd Cryptohaze-Dev/build
cmake -DBOOST_ROOT=$BUILD_BASE_DIRECTORY/$BOOST_DIRECTORY_NAME -DPROTOBUF_INCLUDE_DIR=$BUILD_BASE_DIRECTORY/include/ -DPROTOBUF_LIBRARY=$BUILD_BASE_DIRECTORY/lib/libprotobuf.a ..

echo "Configuration complete."

echo "Attempting to build..."
cd Cryptohaze-Dev/build
make -j4 install

if [ ! -f bin/New-Multiforcer ];then
echo "Build failed! Look for errors..."
else
echo "Build appear to have succeeded.  Binaries in Cryptohaze-Dev/build/bin"
fi
