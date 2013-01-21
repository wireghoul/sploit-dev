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

#include "CUDA_Common/CUDASHA1.h"



#define CUDA_SHA1_KERNEL_CREATE(length) \
__global__ void CUDA_SHA1_Search_##length (unsigned char *OutputPassword, unsigned char *success, \
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, unsigned int count, unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { \
  const int pass_length = length; \
  uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
  uint32_t a,b,c,d, e; \
  uint32_t thread_index = blockIdx.x*blockDim.x + threadIdx.x; \
  uint32_t *DEVICE_Hashes_32 = (uint32_t *)DEVICE_Hashes; \
  unsigned char p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15; \
  UINT4 password_count = 0; \
  __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN]; \
  __shared__ __align__(16) unsigned char sharedBitmap[8192]; \
  __shared__ __align__(8) unsigned char sharedLengths[16];  \
  copyCharsetAndBitmap(sharedCharset, sharedBitmap, sharedLengths, charsetLen, pass_length); \
  loadStartPositions(pass_length, thread_index, DEVICE_Start_Positions,  \
		   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15); \
  while (password_count < count) { \
  initSHA1(pass_length, sharedCharset, \
  	p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
	b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);	 \
  SHA_TRANSFORM_SMALL(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  checkHashMultiSHA1(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords, \
		DEVICE_Hashes_32, success, OutputPassword, \
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
		b0, b1, b2, b3, b4, a, b, c, d, e, b5); \
  password_count++; \
  incrementCounters##length##Multi(); \
  } \
}



#define CUDA_SHA1_KERNEL_CREATELONG(length) \
__global__ void CUDA_SHA1_Search_##length (unsigned char *OutputPassword, unsigned char *success, \
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, unsigned int count, unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { \
  const int pass_length = length; \
  uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
  uint32_t a,b,c,d, e; \
  uint32_t thread_index = blockIdx.x*blockDim.x + threadIdx.x; \
  uint32_t *DEVICE_Hashes_32 = (uint32_t *)DEVICE_Hashes; \
  unsigned char p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, \
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43, \
           p44, p45, p46, p47; \
  UINT4 password_count = 0; \
  __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN]; \
  __shared__ __align__(16) unsigned char sharedBitmap[8192]; \
  __shared__ __align__(8) unsigned char sharedLengths[MAX_PASSWORD_LEN];  \
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
  b15 = ((pass_length * 8) & 0xff) << 24 | (((pass_length * 8) >> 8) & 0xff) << 16; \
  SetCharacterAtPosition(0x80, pass_length, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 ); \
  SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  checkHashMultiSHA1Long(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords, \
		DEVICE_Hashes_32, success, OutputPassword, \
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
                p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, \
                p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43, \
                p44, p45, p46, p47, \
		b0, b1, b2, b3, b4, a, b, c, d, e, b5); \
  password_count++; \
  incrementCounters##length##Multi(); \
  } \
}


CUDA_SHA1_KERNEL_CREATE(1)
CUDA_SHA1_KERNEL_CREATE(2)
CUDA_SHA1_KERNEL_CREATE(3)
CUDA_SHA1_KERNEL_CREATE(4)
CUDA_SHA1_KERNEL_CREATE(5)
CUDA_SHA1_KERNEL_CREATE(6)
CUDA_SHA1_KERNEL_CREATE(7)
CUDA_SHA1_KERNEL_CREATE(8)
CUDA_SHA1_KERNEL_CREATE(9)
CUDA_SHA1_KERNEL_CREATE(10)
// Can be short above this...
CUDA_SHA1_KERNEL_CREATELONG(11)
CUDA_SHA1_KERNEL_CREATELONG(12)
CUDA_SHA1_KERNEL_CREATELONG(13)
CUDA_SHA1_KERNEL_CREATELONG(14)
CUDA_SHA1_KERNEL_CREATELONG(15)
CUDA_SHA1_KERNEL_CREATELONG(16)
CUDA_SHA1_KERNEL_CREATELONG(17)
CUDA_SHA1_KERNEL_CREATELONG(18)
CUDA_SHA1_KERNEL_CREATELONG(19)
CUDA_SHA1_KERNEL_CREATELONG(20)
CUDA_SHA1_KERNEL_CREATELONG(21)
CUDA_SHA1_KERNEL_CREATELONG(22)
CUDA_SHA1_KERNEL_CREATELONG(23)
CUDA_SHA1_KERNEL_CREATELONG(24)
CUDA_SHA1_KERNEL_CREATELONG(25)
CUDA_SHA1_KERNEL_CREATELONG(26)
CUDA_SHA1_KERNEL_CREATELONG(27)
CUDA_SHA1_KERNEL_CREATELONG(28)
CUDA_SHA1_KERNEL_CREATELONG(29)
CUDA_SHA1_KERNEL_CREATELONG(30)
CUDA_SHA1_KERNEL_CREATELONG(31)
CUDA_SHA1_KERNEL_CREATELONG(32)
CUDA_SHA1_KERNEL_CREATELONG(33)
CUDA_SHA1_KERNEL_CREATELONG(34)
CUDA_SHA1_KERNEL_CREATELONG(35)
CUDA_SHA1_KERNEL_CREATELONG(36)
CUDA_SHA1_KERNEL_CREATELONG(37)
CUDA_SHA1_KERNEL_CREATELONG(38)
CUDA_SHA1_KERNEL_CREATELONG(39)
CUDA_SHA1_KERNEL_CREATELONG(40)
CUDA_SHA1_KERNEL_CREATELONG(41)
CUDA_SHA1_KERNEL_CREATELONG(42)
CUDA_SHA1_KERNEL_CREATELONG(43)
CUDA_SHA1_KERNEL_CREATELONG(44)
CUDA_SHA1_KERNEL_CREATELONG(45)
CUDA_SHA1_KERNEL_CREATELONG(46)
CUDA_SHA1_KERNEL_CREATELONG(47)
CUDA_SHA1_KERNEL_CREATELONG(48)

// Copy the shared variables to the host
extern "C" void copySHA1DataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId) {
    //printf("Thread %d in CHHashTypeNTLM.cu, copyNTLMDataToCharset()\n", threadId);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceCharset, hostCharset, (MAX_CHARSET_LENGTH * charsetLength)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(constantBitmap, hostSharedBitmap, 8192));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(charsetLengths, hostCharsetLengths, 16));
}



extern "C" void Launch_CUDA_SHA1_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
        unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions, uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes, unsigned char *DEVICE_Bitmap) {
    if (passlength == 1) {
	  CUDA_SHA1_Search_1 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 2) {
	  CUDA_SHA1_Search_2 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 3) {
	  CUDA_SHA1_Search_3 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 4) {
	  CUDA_SHA1_Search_4 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 5) {
	  CUDA_SHA1_Search_5 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 6) {
	  CUDA_SHA1_Search_6 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 7) {
	  CUDA_SHA1_Search_7 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 8) {
	  CUDA_SHA1_Search_8 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 9) {
	  CUDA_SHA1_Search_9 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 10) {
	  CUDA_SHA1_Search_10 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 11) {
	  CUDA_SHA1_Search_11 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 12) {
	  CUDA_SHA1_Search_12 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 13) {
	  CUDA_SHA1_Search_13 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 14) {
	  CUDA_SHA1_Search_14 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 15) {
	  CUDA_SHA1_Search_15 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 16) {
	  CUDA_SHA1_Search_16 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 17) {
          CUDA_SHA1_Search_17 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 18) {
          CUDA_SHA1_Search_18 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 19) {
          CUDA_SHA1_Search_19 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 20) {
          CUDA_SHA1_Search_20 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 21) {
          CUDA_SHA1_Search_21 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 22) {
          CUDA_SHA1_Search_22 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 23) {
          CUDA_SHA1_Search_23 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 24) {
          CUDA_SHA1_Search_24 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 25) {
          CUDA_SHA1_Search_25 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 26) {
          CUDA_SHA1_Search_26 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 27) {
          CUDA_SHA1_Search_27 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 28) {
          CUDA_SHA1_Search_28 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 29) {
          CUDA_SHA1_Search_29 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 30) {
          CUDA_SHA1_Search_30 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 31) {
          CUDA_SHA1_Search_31 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 32) {
          CUDA_SHA1_Search_32 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 33) {
          CUDA_SHA1_Search_33 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 34) {
          CUDA_SHA1_Search_34 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 35) {
          CUDA_SHA1_Search_35 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 36) {
          CUDA_SHA1_Search_36 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 37) {
          CUDA_SHA1_Search_37 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 38) {
          CUDA_SHA1_Search_38 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 39) {
          CUDA_SHA1_Search_39 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 40) {
          CUDA_SHA1_Search_40 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 41) {
          CUDA_SHA1_Search_41 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 42) {
          CUDA_SHA1_Search_42 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 43) {
          CUDA_SHA1_Search_43 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 44) {
          CUDA_SHA1_Search_44 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 45) {
          CUDA_SHA1_Search_45 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 46) {
          CUDA_SHA1_Search_46 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 47) {
          CUDA_SHA1_Search_47 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 48) {
          CUDA_SHA1_Search_48 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else {
            sprintf(global_interface.exit_message, "SHA1 length >48 not currently supported!\n");
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