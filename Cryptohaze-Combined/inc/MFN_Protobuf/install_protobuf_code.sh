#!/bin/bash

# Installs the protobuf code into the proper places with the correct source
# modification to find the header.  RUN IT FROM THIS DIRECTORY.  Seriously.

echo "Creating CHHashFileVPlain C++..."
protoc --cpp_out=. CHHashFileVPlain.proto
mv CHHashFileVPlain.pb.h ../CH_HashFiles/CHHashFileVPlain.pb.h
sed -e 's/#include "/#include "CH_HashFiles\//' CHHashFileVPlain.pb.cc > ../../src/CH_HashFiles/CHHashFileVPlain.pb.cpp
rm CHHashFileVPlain.pb.cc

echo "Creating CHHashFileVSalted C++..."
protoc --cpp_out=. CHHashFileVSalted.proto
mv CHHashFileVSalted.pb.h ../CH_HashFiles/CHHashFileVSalted.pb.h
sed -e 's/#include "/#include "CH_HashFiles\//' CHHashFileVSalted.pb.cc > ../../src/CH_HashFiles/CHHashFileVSalted.pb.cpp
rm CHHashFileVSalted.pb.cc

echo "Creating MFNCharsetNew C++..."
protoc --cpp_out=. MFNCharsetNew.proto
mv MFNCharsetNew.pb.h ../MFN_Common/MFNCharsetNew.pb.h
sed -e 's/#include "/#include "MFN_Common\//' MFNCharsetNew.pb.cc > ../../src/MFN_Common/MFNCharsetNew.pb.cpp
rm MFNCharsetNew.pb.cc

echo "Creating MFNNetworkRPC C++..."
protoc --cpp_out=. MFNNetworkRPC.proto
mv MFNNetworkRPC.pb.h ../MFN_Common/MFNNetworkRPC.pb.h
sed -e 's/#include "/#include "MFN_Common\//' MFNNetworkRPC.pb.cc > ../../src/MFN_Common/MFNNetworkRPC.pb.cpp
rm MFNNetworkRPC.pb.cc

echo "Creating MFNWorkunit C++..."
protoc --cpp_out=. MFNWorkunit.proto
mv MFNWorkunit.pb.h ../MFN_Common/MFNWorkunit.pb.h
sed -e 's/#include "/#include "MFN_Common\//' MFNWorkunit.pb.cc > ../../src/MFN_Common/MFNWorkunit.pb.cpp
rm MFNWorkunit.pb.cc
