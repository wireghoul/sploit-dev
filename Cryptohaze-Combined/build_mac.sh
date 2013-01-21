#!/bin/bash

rm -rf binaries

make grt staticboost=1
make multiforcer staticboost=1

rm -rf ./build_mac

mkdir ./build_mac
mkdir ./build_mac/Cryptohaze-Mac
mkdir ./build_mac/Cryptohaze-Mac/charsets
mkdir ./build_mac/Cryptohaze-Mac/charsets/ip_addresses
mkdir ./build_mac/Cryptohaze-Mac/kernels
mkdir ./build_mac/Cryptohaze-Mac/lib64
mkdir ./build_mac/Cryptohaze-Mac/test_hashes

cp ./binaries/* ./build_mac/Cryptohaze-Mac
cp ./shared/charsets/* ./build_mac/Cryptohaze-Mac/charsets
cp ./shared/charsets/ip_addresses/* ./build_mac/Cryptohaze-Mac/charsets/ip_addresses
cp ./binaries/kernels/* ./build_mac/Cryptohaze-Mac/kernels
cp ./shared/test_hashes/* ./build_mac/Cryptohaze-Mac/test_hashes

# Strip debugging symbols
strip ./build_mac/Cryptohaze-Mac/*

# Move libraries over
cp /usr/local/cuda/lib/libcudart.dylib ./build_mac/Cryptohaze-Mac/lib64
cp /usr/local/cuda/lib/libtlshook.dylib ./build_mac/Cryptohaze-Mac/lib64
cp /usr/local/cuda/lib/libcuda.dylib ./build_mac/Cryptohaze-Mac/lib64

cp setenv_mac.sh ./build_mac/Cryptohaze-Mac/
chmod 755 ./build_mac/Cryptohaze-Mac/setenv_mac.sh


for file in ./build_mac/Cryptohaze-Mac/*; do
    echo "$file"
    install_name_tool -add_rpath ./lib64 "$file"
done

cd ./build_mac
tar -cvjf ./Cryptohaze-Mac.tar.bz2 ./Cryptohaze-Mac
cd ..
