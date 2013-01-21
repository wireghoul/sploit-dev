#!/bin/bash

# This file builds a number of test files and verifies the MD5 hashes
# are correct.  This is for verifying build correctness.

# Call from the binaries directory.

# Test the three table formats.
./GRTGen-OpenCL -h NTLM -c ../shared/charsets/charsetall -l 6 -i 0 --numchains 10000 --chainlength 1000 --seed 5000 --tableversion 1 $@
./GRTGen-OpenCL -h NTLM -c ../shared/charsets/charsetall -l 6 -i 0 --numchains 10000 --chainlength 1000 --seed 5000 --tableversion 2 $@
./GRTGen-OpenCL -h NTLM -c ../shared/charsets/charsetall -l 6 -i 0 --numchains 10000 --chainlength 1000 --seed 5000 --tableversion 3 $@

./GRTGen-OpenCL -h MD5 -c ../shared/charsets/charsetall -l 6 -i 0 --numchains 10000 --chainlength 1000 --seed 5000 --tableversion 1 $@
./GRTGen-OpenCL -h MD5 -c ../shared/charsets/charsetall -l 6 -i 0 --numchains 10000 --chainlength 1000 --seed 5000 --tableversion 2 $@
./GRTGen-OpenCL -h MD5 -c ../shared/charsets/charsetall -l 6 -i 0 --numchains 10000 --chainlength 1000 --seed 5000 --tableversion 3 $@

# Test some various seeds & parameters
./GRTGen-OpenCL -h MD5 -c ../shared/charsets/charsetall -l 7 -i 0 --numchains 10000 --chainlength 1000 --seed 12345 --tableversion 2 $@
./GRTGen-OpenCL -h MD5 -c ../shared/charsets/charsetall -l 7 -i 100000 --numchains 10000 --chainlength 1000 --seed 12345 --tableversion 2 $@
./GRTGen-OpenCL -h MD5 -c ../shared/charsets/charsetall -l 7 -i 200000 --numchains 10000 --chainlength 1000 --seed 12345 --tableversion 2 $@



./GRTGen-OpenCL -h MD5 -c ../shared/charsets/charsetall -l 8 -i 0 --numchains 10000 --chainlength 1000 --seed 45454 --tableversion 2 $@
./GRTGen-OpenCL -h MD5 -c ../shared/charsets/charsetall -l 8 -i 100000 --numchains 10000 --chainlength 1000 --seed 45454 --tableversion 2 $@
./GRTGen-OpenCL -h MD5 -c ../shared/charsets/charsetall -l 8 -i 200000 --numchains 10000 --chainlength 1000 --seed 45454 --tableversion 2 $@

# Test generating multiple tables
./GRTGen-OpenCL -h MD5 -c ../shared/charsets/charsetall -l 8 -i 0 --numchains 10000 --chainlength 1000 --numtables 10 --seed 98765 --tableversion 2 $@

# Test generating multiple files in V3
./GRTGen-OpenCL -h MD5 -c ../shared/charsets/charsetall -l 8 -i 0 --numchains 10000 --chainlength 1000 --numtables 10 --seed 98765 --tableversion 3 $@

# Now verify.
echo "Testing files: They should all match perfectly!  Mismatches ARE A BUG!"

md5sum --quiet -c ../shared/grt_verification/test_grt_files.md5

echo Verification completed.

