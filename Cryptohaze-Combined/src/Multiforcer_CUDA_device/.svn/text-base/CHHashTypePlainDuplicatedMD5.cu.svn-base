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
__device__ __constant__ __align__(16) unsigned char charsetLengths[MAX_PASSWORD_LEN];
__device__ __constant__ unsigned char constantBitmap[8192]; // for lookups


#include "Multiforcer_CUDA_device/CUDAcommon.h"
#include "CUDA_Common/CUDAMD5.h"


/*
#define MD5_CUDA_KERNEL_CREATE(length) \
__global__ void CUDA_MD5_Search_##length (unsigned char *OutputPassword, unsigned char *success, \
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, unsigned int count,  \
				unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { \
  const int pass_length = length; \
  uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
  uint32_t a,b,c,d; \
  uint32_t thread_index = blockIdx.x*blockDim.x + threadIdx.x; \
  uint32_t *DEVICE_Hashes_32 = (uint32_t *)DEVICE_Hashes; \
  unsigned char p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15; \
  UINT4 password_count = 0; \
  __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN]; \
  __shared__ __align__(16) unsigned char sharedBitmap[8192]; \
  __shared__ __align__(8)  unsigned char sharedLengths[MAX_PASSWORD_LEN]; \
  copyCharsetAndBitmap(sharedCharset, sharedBitmap, sharedLengths, charsetLen, pass_length); \
  loadStartPositions(pass_length, thread_index, DEVICE_Start_Positions,  \
		   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15); \
  while (password_count < count) { \
  initMD(pass_length, sharedCharset, \
  	p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
	b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);	 \
  CUDA_MD5(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d); \
  checkHashMulti(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords, \
		DEVICE_Hashes_32, success, OutputPassword, \
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
		a, b, c, d, b0, b1, b2, b3, b4, b5); \
  password_count++; \
  incrementCounters##length##Multi(); \
  } \
}*/



// This is actually just as fast for small sizes, so we use it.
// Compiler optimizations FTW!
#define DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(length) \
__global__ void CUDA_DuplicatedMD5_Search_##length (unsigned char *OutputPassword, unsigned char *success, \
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, unsigned int count,  \
				unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { \
  const int pass_length = length; \
  uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
  uint32_t a,b,c,d; \
  uint32_t thread_index = blockIdx.x*blockDim.x + threadIdx.x; \
  uint32_t *DEVICE_Hashes_32 = (uint32_t *)DEVICE_Hashes; \
  unsigned char p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, \
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43, \
           p44, p45, p46, p47; \
  UINT4 password_count = 0; \
  __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN]; \
  __shared__ __align__(16) unsigned char sharedBitmap[8192]; \
  __shared__ __align__(8)  unsigned char sharedLengths[MAX_PASSWORD_LEN]; \
  copyCharsetAndBitmap(sharedCharset, sharedBitmap, sharedLengths, charsetLen, pass_length); \
  loadStartPositionsLong(pass_length, thread_index, DEVICE_Start_Positions,  \
		   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
                   p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, \
                   p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43, \
                   p44, p45, p46, p47); \
  while (password_count < count) { \
  clearB0toB15(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  LoadPasswordAtPositionLong(pass_length, 0, sharedCharset, \
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
        p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, \
        p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43, \
        p44, p45, p46, p47, \
        b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  LoadPasswordAtPositionLong(pass_length, pass_length, sharedCharset, \
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
        p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, \
        p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43, \
        p44, p45, p46, p47, \
        b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  CUDA_GENERIC_MD5(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, \
        a, b, c, d, pass_length * 2); \
  checkDuplicatedHashMultiLong(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords, \
		DEVICE_Hashes_32, success, OutputPassword, \
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
                p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, \
                p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43, \
                p44, p45, p46, p47, \
		a, b, c, d, b0, b1, b2, b3, b4, b5); \
  password_count++; \
  incrementCounters##length##Multi(); \
  } \
}

DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(1)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(2)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(3)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(4)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(5)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(6)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(7)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(8)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(9)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(10)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(11)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(12)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(13)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(14)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(15)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(16)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(17)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(18)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(19)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(20)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(21)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(22)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(23)
DUPLICATEDMD5_CUDA_KERNEL_CREATE_LONG(24)

// Copy the shared variables to the host
extern "C" void copyDuplicatedMD5DataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId) {
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceCharset, hostCharset, (MAX_CHARSET_LENGTH * charsetLength)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(constantBitmap, hostSharedBitmap, 8192));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(charsetLengths, hostCharsetLengths, MAX_PASSWORD_LEN));
}

extern "C" void Launch_CUDA_DuplicatedMD5_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
						unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions, uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes, unsigned char *DEVICE_Bitmap) {

    if (passlength == 1) {
	  CUDA_DuplicatedMD5_Search_1 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 2) {
	  CUDA_DuplicatedMD5_Search_2 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 3) {
	  CUDA_DuplicatedMD5_Search_3 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 4) {
	  CUDA_DuplicatedMD5_Search_4 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 5) {
	  CUDA_DuplicatedMD5_Search_5 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 6) {
	  CUDA_DuplicatedMD5_Search_6 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 7) {
	  CUDA_DuplicatedMD5_Search_7 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 8) {
	  CUDA_DuplicatedMD5_Search_8 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 9) {
	  CUDA_DuplicatedMD5_Search_9 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 10) {
	  CUDA_DuplicatedMD5_Search_10 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 11) {
	  CUDA_DuplicatedMD5_Search_11 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 12) {
	  CUDA_DuplicatedMD5_Search_12 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 13) {
	  CUDA_DuplicatedMD5_Search_13 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 14) {
	  CUDA_DuplicatedMD5_Search_14 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 15) {
	  CUDA_DuplicatedMD5_Search_15 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 16) {
	  CUDA_DuplicatedMD5_Search_16 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 17) {
          CUDA_DuplicatedMD5_Search_17 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 18) {
          CUDA_DuplicatedMD5_Search_18 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 19) {
          CUDA_DuplicatedMD5_Search_19 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 20) {
          CUDA_DuplicatedMD5_Search_20 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 21) {
          CUDA_DuplicatedMD5_Search_21 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 22) {
          CUDA_DuplicatedMD5_Search_22 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 23) {
          CUDA_DuplicatedMD5_Search_23 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 24) {
          CUDA_DuplicatedMD5_Search_24 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else {
            sprintf(global_interface.exit_message, "Duplicated MD5 length >24 not currently supported!\n");
            global_interface.exit = 1;
            return;
        }

        cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
      {
        sprintf(global_interface.exit_message, "Cuda error: %s.\n", cudaGetErrorString( err) );
        global_interface.exit = 1;
        return;
      }
}