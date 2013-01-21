#!/bin/bash

rm -rf binaries

make grtcrack staticboost=1

rm -rf ./build_linux

mkdir ./build_linux
mkdir ./build_linux/GRTCrack-Linux
mkdir ./build_linux/GRTCrack-Linux/kernels
mkdir ./build_linux/GRTCrack-Linux/lib64
mkdir ./build_linux/GRTCrack-Linux/test_hashes

cp ./binaries/* ./build_linux/GRTCrack-Linux
cp ./binaries/kernels/* ./build_linux/GRTCrack-Linux/kernels
cp ./shared/test_hashes/* ./build_linux/GRTCrack-Linux/test_hashes

# Strip debugging symbols
strip ./build_linux/GRTCrack-Linux/*

# Move libraries over
cp /usr/local/cuda/lib64/libcudart.so.4 ./build_linux/GRTCrack-Linux/lib64

cd ./build_linux
tar -cvjf ./GRTCrack-Linux.tar.bz2 ./GRTCrack-Linux
cd ..
