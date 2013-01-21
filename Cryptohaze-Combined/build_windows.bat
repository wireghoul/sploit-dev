
rmdir /s build_windows/Cryptohaze-Windows

mkdir build_windows\Cryptohaze-Windows
mkdir build_windows\Cryptohaze-Windows\charsets
mkdir build_windows\Cryptohaze-Windows\charsets\ip_addresses
mkdir build_windows\Cryptohaze-Windows\kernels
mkdir build_windows\Cryptohaze-Windows\test_hashes

copy binaries\* build_windows\Cryptohaze-Windows
copy shared\charsets\* build_windows\Cryptohaze-Windows\charsets
copy shared\charsets\ip_addresses\* build_windows\Cryptohaze-Windows\charsets\ip_addresses
copy binaries\kernels\* build_windows\Cryptohaze-Windows\kernels
copy shared\test_hashes\* build_windows\Cryptohaze-Windows\test_hashes

