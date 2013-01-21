/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
Copyright (C) 2011  Bitweasil (http://www.cryptohaze.com/)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/


#include "Multiforcer_Common/CHCommon.h"

extern struct global_commands global_interface;


typedef uint32_t UINT4;
__device__ __constant__ char deviceCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN];
__device__ __constant__ __align__(16) unsigned char charsetLengths[16];
__device__ __constant__ unsigned char constantBitmap[8192]; // for lookups


#include "Multiforcer_CUDA_device/CUDAcommon.h"
#include "CUDA_Common/CUDAMD4.h"


#define MD4_CUDA_KERNEL_CREATE(length) \
__global__ void CUDA_MD4_Search_##length (unsigned char *OutputPassword, unsigned char *success,  \
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, \
				unsigned int count, unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { \
  const int pass_length = length; \
  uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
  uint32_t a,b,c,d; \
  uint32_t thread_index = blockIdx.x*blockDim.x + threadIdx.x; \
  uint32_t *DEVICE_Hashes_32 = (uint32_t *)DEVICE_Hashes; \
  unsigned char p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15; \
  UINT4 password_count = 0; \
  __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN]; \
  __shared__ __align__(16) unsigned char sharedBitmap[8192]; \
  __shared__ unsigned char sharedLengths[16]; \
  copyCharsetAndBitmap(sharedCharset, sharedBitmap, sharedLengths, charsetLen, pass_length); \
  loadStartPositions(pass_length, thread_index, DEVICE_Start_Positions,  \
		   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15); \
  while (password_count < count) { \
  initMD(pass_length, sharedCharset, \
  	p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
	b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);	 \
  CUDA_MD4(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d); \
  checkHashMulti(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords, \
		DEVICE_Hashes_32, success, OutputPassword, \
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
		a, b, c, d, b0, b1, b2, b3, b4, b5); \
  password_count++; \
  incrementCounters##length##Multi(); \
  } \
}


MD4_CUDA_KERNEL_CREATE(1)
MD4_CUDA_KERNEL_CREATE(2)
MD4_CUDA_KERNEL_CREATE(3)
MD4_CUDA_KERNEL_CREATE(4)
MD4_CUDA_KERNEL_CREATE(5)
MD4_CUDA_KERNEL_CREATE(6)
MD4_CUDA_KERNEL_CREATE(7)
MD4_CUDA_KERNEL_CREATE(8)
MD4_CUDA_KERNEL_CREATE(9)
MD4_CUDA_KERNEL_CREATE(10)
MD4_CUDA_KERNEL_CREATE(11)
MD4_CUDA_KERNEL_CREATE(12)
MD4_CUDA_KERNEL_CREATE(13)
MD4_CUDA_KERNEL_CREATE(14)
MD4_CUDA_KERNEL_CREATE(15)
MD4_CUDA_KERNEL_CREATE(16)

// Copy the shared variables to the host
extern "C" void copyMD4DataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId) {
    //printf("Thread %d in CHHashTypeMD5.cu, copyMD5DataToCharset()\n", threadId);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceCharset, hostCharset, (MAX_CHARSET_LENGTH * charsetLength)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(constantBitmap, hostSharedBitmap, 8192));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(charsetLengths, hostCharsetLengths, 16));
}

extern "C" void Launch_CUDA_MD4_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
						unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions, uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes, unsigned char *DEVICE_Bitmap) {
    if (passlength == 1) {
	  CUDA_MD4_Search_1 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 2) {
	  CUDA_MD4_Search_2 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 3) {
	  CUDA_MD4_Search_3 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 4) {
	  CUDA_MD4_Search_4 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 5) {
	  CUDA_MD4_Search_5 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 6) {
	  CUDA_MD4_Search_6 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 7) {
	  CUDA_MD4_Search_7 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 8) {
	  CUDA_MD4_Search_8 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 9) {
	  CUDA_MD4_Search_9 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 10) {
	  CUDA_MD4_Search_10 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 11) {
	  CUDA_MD4_Search_11 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 12) {
	  CUDA_MD4_Search_12 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 13) {
	  CUDA_MD4_Search_13 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 14) {
	  CUDA_MD4_Search_14 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 15) {
	  CUDA_MD4_Search_15 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 16) {
	  CUDA_MD4_Search_16 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	}

	cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
      {
        sprintf(global_interface.exit_message, "Cuda error: %s.\n", cudaGetErrorString( err) );
        global_interface.exit = 1;
        return;
      }
}