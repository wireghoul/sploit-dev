#!/bin/bash

# Set to the number of CPU cores to use for builds.
CPU_THREADS=4

CUDA_DOWNLOAD_URL=http://developer.download.nvidia.com/compute/cuda/5_0/rel-update-1/installers/cuda_5.0.35_linux_64_ubuntu11.10-1.run
CUDA_PKG_NAME=cuda_5.0.35_linux_64_ubuntu11.10-1.run

BOOST_DOWNLOAD_URL=http://downloads.sourceforge.net/project/boost/boost/1.52.0/boost_1_52_0.tar.bz2
BOOST_DOWNLOAD_FILE=boost_1_52_0.tar.bz2
BOOST_DIRECTORY_NAME=boost_1_52_0


PROTOBUF_DOWNLOAD_URL=http://protobuf.googlecode.com/files/protobuf-2.4.1.tar.bz2
PROTOBUF_DOWNLOAD_FILE=protobuf-2.4.1.tar.bz2
PROTOBUF_DIRECTORY_NAME=protobuf-2.4.1

AMD_DRIVER_DOWNLOAD_URL=http://www2.ati.com/drivers/linux/amd-driver-installer-12-8-x86.x86_64.zip
AMD_DRIVER_FILE=amd-driver-installer-12-8-x86.x86_64.zip
AMD_DRIVER_UNZIPPED=amd-driver-installer-8.982-x86.x86_64.run

# Store the current directory
BUILD_BASE_DIRECTORY=`pwd`

# Create downloads directory for things being installed
mkdir downloads

# cmake next
if [ ! -f /usr/bin/g++ ]; then
  echo "Installing basic build environment and G++."
  sudo apt-get -y install build-essential subversion libcurl4-openssl-dev libncurses5-dev unzip libargtable2-dev debhelper dh-modaliases execstack dkms
  if [ ! -f /usr/bin/g++ ]; then
    echo "G++ still not installed.  Sorry."
    DEP_ERRORS=1
  fi
fi

# cmake next
if [ ! -f /usr/bin/cmake ]; then
  echo "Installing cmake as root - enter password"
  sudo apt-get -y install cmake
  if [ ! -f /usr/bin/cmake ]; then
    echo "CMAKE still not installed.  Sorry."
    DEP_ERRORS=1
  fi
fi

# Check for dependencies
# CUDA first.
if [ ! -f /usr/local/cuda/bin/nvcc ]; then
  echo "CUDA toolkit is not installed.  Downloading, will attempt to install."
  cd downloads
  if [ ! -f $CUDA_PKG_NAME ]; then
    wget "$CUDA_DOWNLOAD_URL"
  fi
  chmod 755 $CUDA_PKG_NAME
  echo "About to install the CUDA toolkit.  You do NOT need to install the driver"
  echo "unless you have a CUDA capable card.  This will run as root with sudo."
  echo "You can ignore warnings about no driver unless you have an nVidia card."
  # Install the toolkit only
  echo
  sudo ./$CUDA_PKG_NAME -toolkit -silent
  # Attempt to add the needed path configs & such.
  echo >> ~/.bashrc
  echo "export CUDA_HOME=/usr/local/cuda-5.0" >> ~/.bashrc
  echo "PATH=${CUDA_HOME}/bin:${PATH}" >> ~/.bashrc
  echo "export PATH" >> ~/.bashrc
  source ~/.bashrc
  sudo bash -c 'echo /usr/local/cuda-5.0/lib64 >> /etc/ld.so.conf'
  sudo ldconfig
  if [ ! -f /usr/local/cuda/bin/nvcc ]; then
    echo "CUDA toolkit still not installed.  Sorry."
    DEP_ERRORS=1
  fi
  cd ..
fi

# If libOpenCL is not installed, go at the AMD drivers.
if [ ! -f /usr/lib/libOpenCL.so ]; then
  cd downloads
  if [ ! -f $AMD_DRIVER_FILE ]; then
    wget "$AMD_DRIVER_DOWNLOAD_URL"
  fi
  unzip $AMD_DRIVER_FILE
  echo "Creating AMD driver packages and installing..."
  chmod 755 $AMD_DRIVER_UNZIPPED
  sudo ./$AMD_DRIVER_UNZIPPED --buildpkg
  sudo dpkg -i fglrx*.deb
  sudo apt-get -f -y install
  cd ..
fi

if [ "$DEP_ERRORS" == 1 ]; then
echo "Dependencies not installed.  Exiting."
exit 1
fi


if [ ! -f lib/libboost_thread.a ]; then
  # Download boost - use latest URL.
  cd downloads
  wget "$BOOST_DOWNLOAD_URL"

  # Go back to root and unzip boost
  cd ..
  echo "Unpacking boost..."
  tar -xjf downloads/$BOOST_DOWNLOAD_FILE

  # Change to directory and start the build
  cd $BOOST_DIRECTORY_NAME
  ./bootstrap.sh
  ./b2 install --prefix="$BUILD_BASE_DIRECTORY" --without-mpi -j$CPU_THREADS
  cd ..

  # Boost, in theory, is installed.
  if [ ! -f lib/libboost_thread.a ]; then
    echo "Boost did not install properly - please look for errors."
    exit 1
  fi
fi

# Grab the protobuf libraries and install them.
if [ ! -f lib/libprotobuf.a ]; then
  cd downloads
  wget "$PROTOBUF_DOWNLOAD_URL"
  cd ..
  echo "Unpacking protobuf..."
  tar -xjf downloads/$PROTOBUF_DOWNLOAD_FILE

  # Unzip & build protobuf
  cd $PROTOBUF_DIRECTORY_NAME
  ./configure --prefix "$BUILD_BASE_DIRECTORY"
  make -j$CPU_THREADS && make install
  cd ..
fi

# Check for protobuf
if [ ! -f lib/libprotobuf.a ]; then
  echo "Protobuf did not install properly - please look for errors."
  exit 1
fi

# Get the source tree
svn co https://cryptohaze.svn.sourceforge.net/svnroot/cryptohaze/Cryptohaze-Combined Cryptohaze-Dev

# Set up cmake
cd Cryptohaze-Dev/build
# Remove any old cached stuff if the setup failed, otherwise it will keep failing.
rm CMakeCache.txt
cmake -DBOOST_ROOT=$BUILD_BASE_DIRECTORY/$BOOST_DIRECTORY_NAME -DPROTOBUF_INCLUDE_DIR=$BUILD_BASE_DIRECTORY/include/ -DPROTOBUF_LIBRARY=$BUILD_BASE_DIRECTORY/lib/libprotobuf.a ..

echo "Configuration complete."

echo "Attempting to build..."
make -j$CPU_THREADS install

if [ ! -f bin/New-Multiforcer ];then
echo "Build failed! Look for errors..."
else
echo "Build appear to have succeeded.  Binaries in Cryptohaze-Dev/build/bin"
fi
