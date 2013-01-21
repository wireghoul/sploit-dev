#!/bin/bash

rm -rf binaries

make grt staticboost=1
make multiforcer staticboost=1
make binaries/New-Multiforcer staticboost=1

rm -rf ./build_linux

mkdir ./build_linux
mkdir ./build_linux/Cryptohaze-Linux
mkdir ./build_linux/Cryptohaze-Linux/charsets
mkdir ./build_linux/Cryptohaze-Linux/charsets/ip_addresses
mkdir ./build_linux/Cryptohaze-Linux/kernels
mkdir ./build_linux/Cryptohaze-Linux/lib64
mkdir ./build_linux/Cryptohaze-Linux/test_hashes

cp ./binaries/* ./build_linux/Cryptohaze-Linux
cp ./shared/charsets/* ./build_linux/Cryptohaze-Linux/charsets
cp ./shared/charsets/ip_addresses/* ./build_linux/Cryptohaze-Linux/charsets/ip_addresses
cp ./binaries/kernels/* ./build_linux/Cryptohaze-Linux/kernels
cp ./shared/test_hashes/* ./build_linux/Cryptohaze-Linux/test_hashes

# Strip debugging symbols
strip ./build_linux/Cryptohaze-Linux/*

# Move libraries over
cp /usr/local/cuda/lib64/libcudart.so.4 ./build_linux/Cryptohaze-Linux/lib64

cd ./build_linux
tar -cvjf ./Cryptohaze-Linux.tar.bz2 ./Cryptohaze-Linux
cd ..
