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
#include "CUDA_Common/CUDALM.h"


__device__ inline void checkHashMultiLM(int pass_length, unsigned char *sharedBitmap, unsigned char *DEVICE_HashTable, uint32_t numberOfPasswords,
		uint32_t *DEVICE_Hashes_32, unsigned char *success, unsigned char *OutputPassword,
		unsigned char p0, unsigned char p1, unsigned char p2, unsigned char p3,
		unsigned char p4, unsigned char p5, unsigned char p6,
		UINT4 a, UINT4 b) {
  uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp;
  if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) {
  if ((!DEVICE_HashTable) || (DEVICE_HashTable && (DEVICE_HashTable[a >> 3] >> (a & 0x00000007)) & 0x00000001)) {
  // Init binary search through global password space
  search_high = numberOfPasswords;
  search_low = 0;
  search_index = 0;
  while (search_low < search_high) {
    // Midpoint between search_high and search_low
    search_index = search_low + (search_high - search_low) / 2;
    // reorder from low endian to big endian for searching, as hashes are sorted by byte.
    temp = DEVICE_Hashes_32[2 * search_index];
    hash_order_mem = (temp & 0xff) << 24 | ((temp >> 8) & 0xff) << 16 | ((temp >> 16) & 0xff) << 8 | ((temp >> 24) & 0xff);
    hash_order_a = (a & 0xff) << 24 | ((a >> 8) & 0xff) << 16 | ((a >> 16) & 0xff) << 8 | ((a >> 24) & 0xff);

    // Adjust search_high & search_low to work through space
    if (hash_order_mem < hash_order_a) {
      search_low = search_index + 1;
    } else  {
      search_high = search_index;
    }
    if ((hash_order_a == hash_order_mem) && (search_low < numberOfPasswords)) {
      // Break out of the search loop - search_index is on a value
      break;
    }
  }

  // Yes - it's a goto.  And it speeds up performance significantly (15%).
  // It stays.  These values are already loaded.  If they are not the same,
  // there is NO point to touching global memory again.
  if (hash_order_a != hash_order_mem) {
      goto next;
  }
  // We've broken out of the loop, search_index should be on a matching value
  // Loop while the search index is the same - linear search through this to find all possible
  // matching passwords.
  // We first need to move backwards to the beginning, as we may be in the middle of a set of matching hashes.
  // If we are index 0, do NOT subtract, as we will wrap and this goes poorly.

  while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 2])) {
    search_index--;
  }

  while ((a == DEVICE_Hashes_32[search_index * 2])) {
  /*if (a == DEVICE_Hashes_32[search_index * 4])*/ {
    if (b == DEVICE_Hashes_32[search_index * 2 + 1]) {
        if (pass_length >= 1) OutputPassword[search_index * MAX_PASSWORD_LEN + 0] = deviceCharset[p0];
        if (pass_length >= 2) OutputPassword[search_index * MAX_PASSWORD_LEN + 1] = deviceCharset[p1 + (MAX_CHARSET_LENGTH * 1)];
        if (pass_length >= 3) OutputPassword[search_index * MAX_PASSWORD_LEN + 2] = deviceCharset[p2 + (MAX_CHARSET_LENGTH * 2)];
        if (pass_length >= 4) OutputPassword[search_index * MAX_PASSWORD_LEN + 3] = deviceCharset[p3 + (MAX_CHARSET_LENGTH * 3)];
        if (pass_length >= 5) OutputPassword[search_index * MAX_PASSWORD_LEN + 4] = deviceCharset[p4 + (MAX_CHARSET_LENGTH * 4)];
        if (pass_length >= 6) OutputPassword[search_index * MAX_PASSWORD_LEN + 5] = deviceCharset[p5 + (MAX_CHARSET_LENGTH * 5)];
        if (pass_length >= 7) OutputPassword[search_index * MAX_PASSWORD_LEN + 6] = deviceCharset[p6 + (MAX_CHARSET_LENGTH * 6)];
        success[search_index] = (unsigned char) 1;
    }
  }
  search_index++;
  }
  }
  }
  // This is where the goto goes.  Notice the skipping of all the global memory access.
  next:
  return;
}




#define LM_CUDA_KERNEL_CREATE(length) \
__global__ void CUDA_LM_Search_##length (unsigned char *OutputPassword, unsigned char *success, \
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, unsigned int count,  \
				unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { \
  const int pass_length = length; \
  uint32_t b0, b1, bz; \
  uint32_t a,b;  \
  uint32_t thread_index = blockIdx.x*blockDim.x + threadIdx.x;  \
  uint32_t *DEVICE_Hashes_32 = (uint32_t *)DEVICE_Hashes;  \
  unsigned char p0, p1, p2, p3, p4, p5, p6, p7, pz; \
  UINT4 password_count = 0;  \
  __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * 7]; \
  __shared__ __align__(16) unsigned char sharedBitmap[8192];  \
  __shared__ __align__(8)  unsigned char sharedLengths[MAX_PASSWORD_LEN];  \
  __shared__ uint32_t shared_des_skb[8][64]; \
  __shared__ uint32_t shared_des_SPtrans[8][64]; \
  if (threadIdx.x == 0) { \
    for (a = 0; a < 8; a++) { \
        for (b = 0; b < 64; b++) { \
            shared_des_skb[a][b] = des_skb[a][b]; \
            shared_des_SPtrans[a][b] = des_SPtrans[a][b]; \
        } \
    } \
    } \
    syncthreads(); \
  copyCharsetAndBitmap(sharedCharset, sharedBitmap, sharedLengths, charsetLen, pass_length);  \
  loadStartPositions(pass_length, thread_index, DEVICE_Start_Positions,   \
		   p0, p1, p2, p3, p4, p5, p6, p7, pz, pz, pz, pz, pz, pz, pz, pz); \
  while (password_count < count) { \
      b0 = 0x00000000; \
      b1 = 0x00000000; \
      LoadPasswordAtPosition(pass_length, 0, sharedCharset, \
          p0, p1, p2, p3, p4, p5, p6, p7, pz, pz, pz, pz, pz, pz, pz, pz, \
          b0, b1, bz, bz, bz, bz, bz, bz, bz, bz, bz, bz, bz, bz, bz, bz); \
      cudaLM(b0, b1, a, b, shared_des_skb, shared_des_SPtrans); \
      checkHashMultiLM(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords, \
		DEVICE_Hashes_32, success, OutputPassword,  \
		p0, p1, p2, p3, p4, p5, p6, a, b); \
    password_count++; \
    incrementCounters##length##Multi(); \
  }  \
}


LM_CUDA_KERNEL_CREATE(1)
LM_CUDA_KERNEL_CREATE(2)
LM_CUDA_KERNEL_CREATE(3)
LM_CUDA_KERNEL_CREATE(4)
LM_CUDA_KERNEL_CREATE(5)
LM_CUDA_KERNEL_CREATE(6)
LM_CUDA_KERNEL_CREATE(7)

// Copy the shared variables to the host
extern "C" void copyLMDataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId) {
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceCharset, hostCharset, (MAX_CHARSET_LENGTH * charsetLength)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(constantBitmap, hostSharedBitmap, 8192));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(charsetLengths, hostCharsetLengths, MAX_PASSWORD_LEN));
}

extern "C" void Launch_CUDA_LM_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
						unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions, uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes, unsigned char *DEVICE_Bitmap) {

    if (passlength == 1) {
	  CUDA_LM_Search_1 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 2) {
	  CUDA_LM_Search_2 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 3) {
	  CUDA_LM_Search_3 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 4) {
	  CUDA_LM_Search_4 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 5) {
	  CUDA_LM_Search_5 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 6) {
	  CUDA_LM_Search_6 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 7) {
	  CUDA_LM_Search_7 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else {
            sprintf(global_interface.exit_message, "LM length >7 is meaningless!\n");
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