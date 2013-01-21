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

/* MySQL323 CUDA functions for cryptohaze.com Multiforcer */

typedef uint32_t UINT4;


/*
#define MYSQL323_CUDA_KERNEL_CREATE(length) \
__global__ void CUDA_NTLM_Search_##length (unsigned char *OutputPassword, unsigned char *success,  \
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions,  \
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
  __shared__ __align__(8) unsigned char sharedLengths[16]; \
  copyCharsetAndBitmap(sharedCharset, sharedBitmap, sharedLengths, charsetLen, pass_length); \
  loadStartPositions(pass_length, thread_index, DEVICE_Start_Positions,  \
		   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15); \
  while (password_count < count) { \
  initNTLM(pass_length, sharedCharset, \
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


MYSQL323_CUDA_KERNEL_CREATE(1)
MYSQL323_CUDA_KERNEL_CREATE(2)
MYSQL323_CUDA_KERNEL_CREATE(3)
MYSQL323_CUDA_KERNEL_CREATE(4)
MYSQL323_CUDA_KERNEL_CREATE(5)
MYSQL323_CUDA_KERNEL_CREATE(6)
MYSQL323_CUDA_KERNEL_CREATE(7)
MYSQL323_CUDA_KERNEL_CREATE(8)
MYSQL323_CUDA_KERNEL_CREATE(9)
MYSQL323_CUDA_KERNEL_CREATE(10)
MYSQL323_CUDA_KERNEL_CREATE(11)
MYSQL323_CUDA_KERNEL_CREATE(12)
MYSQL323_CUDA_KERNEL_CREATE(13)
MYSQL323_CUDA_KERNEL_CREATE(14)
MYSQL323_CUDA_KERNEL_CREATE(15)
MYSQL323_CUDA_KERNEL_CREATE(16)
 *
 * */


__device__ inline void checkHashMySQL323(int pass_length, unsigned char *sharedBitmap, 
                unsigned char *DEVICE_HashTable, uint32_t numberOfPasswords,
		uint32_t *DEVICE_Hashes_32, unsigned char *success, unsigned char *OutputPassword,
		unsigned char p0, unsigned char p1, unsigned char p2, unsigned char p3,
		unsigned char p4, unsigned char p5, unsigned char p6, unsigned char p7,
		unsigned char p8, unsigned char p9, unsigned char p10, unsigned char p11,
		unsigned char p12, unsigned char p13, unsigned char p14, unsigned char p15,
		UINT4 a, UINT4 b) {
    
  uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp;

  // Endian swap a for this hash.

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
    // Endian issues...
    //hash_order_a = a;
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

//printf("Hash order a: %08x   Mem order a: %08x\n", hash_order_a, DEVICE_Hashes_32[search_index * 2]);
  while ((a == DEVICE_Hashes_32[search_index * 2])) {
      //printf("Hash order a: %08x   Mem order a: %08x\n", hash_order_a, DEVICE_Hashes_32[search_index * 2]);
  if (a == DEVICE_Hashes_32[search_index * 2]) {
      //printf("A hit\n");
    if (b == DEVICE_Hashes_32[search_index * 2 + 1]) {
        //printf("B hit\n");
      {
        {
        if (pass_length >= 1) OutputPassword[search_index * MAX_PASSWORD_LEN + 0] = deviceCharset[p0];
        if (pass_length >= 2) OutputPassword[search_index * MAX_PASSWORD_LEN + 1] = deviceCharset[p1 + (MAX_CHARSET_LENGTH * 1)];
        if (pass_length >= 3) OutputPassword[search_index * MAX_PASSWORD_LEN + 2] = deviceCharset[p2 + (MAX_CHARSET_LENGTH * 2)];
        if (pass_length >= 4) OutputPassword[search_index * MAX_PASSWORD_LEN + 3] = deviceCharset[p3 + (MAX_CHARSET_LENGTH * 3)];
        if (pass_length >= 5) OutputPassword[search_index * MAX_PASSWORD_LEN + 4] = deviceCharset[p4 + (MAX_CHARSET_LENGTH * 4)];
        if (pass_length >= 6) OutputPassword[search_index * MAX_PASSWORD_LEN + 5] = deviceCharset[p5 + (MAX_CHARSET_LENGTH * 5)];
        if (pass_length >= 7) OutputPassword[search_index * MAX_PASSWORD_LEN + 6] = deviceCharset[p6 + (MAX_CHARSET_LENGTH * 6)];
        if (pass_length >= 8) OutputPassword[search_index * MAX_PASSWORD_LEN + 7] = deviceCharset[p7 + (MAX_CHARSET_LENGTH * 7)];
        if (pass_length >= 9) OutputPassword[search_index * MAX_PASSWORD_LEN + 8] = deviceCharset[p8 + (MAX_CHARSET_LENGTH * 8)];
        if (pass_length >= 10) OutputPassword[search_index * MAX_PASSWORD_LEN + 9] = deviceCharset[p9 + (MAX_CHARSET_LENGTH * 9)];
        if (pass_length >= 11) OutputPassword[search_index * MAX_PASSWORD_LEN + 10] = deviceCharset[p10 + (MAX_CHARSET_LENGTH * 10)];
        if (pass_length >= 12) OutputPassword[search_index * MAX_PASSWORD_LEN + 11] = deviceCharset[p11 + (MAX_CHARSET_LENGTH * 11)];
        if (pass_length >= 13) OutputPassword[search_index * MAX_PASSWORD_LEN + 12] = deviceCharset[p12 + (MAX_CHARSET_LENGTH * 12)];
        if (pass_length >= 14) OutputPassword[search_index * MAX_PASSWORD_LEN + 13] = deviceCharset[p13 + (MAX_CHARSET_LENGTH * 13)];
        if (pass_length >= 15) OutputPassword[search_index * MAX_PASSWORD_LEN + 14] = deviceCharset[p14 + (MAX_CHARSET_LENGTH * 14)];
        if (pass_length >= 16) OutputPassword[search_index * MAX_PASSWORD_LEN + 15] = deviceCharset[p15 + (MAX_CHARSET_LENGTH * 15)];
        success[search_index] = (unsigned char) 1;
        /*printf("FOUND PASSWORD:");
        for (int i = 0; i < pass_length; i++) {
            printf("%c", OutputPassword[search_index * MAX_PASSWORD_LEN + i]);
        }
        printf("\n");*/
        }
      }
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


__global__ void CUDA_MYSQL323_Search_3 (unsigned char *OutputPassword, unsigned char *success,  
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions,  
				unsigned int count, unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { 
  const int pass_length = 3;
  uint32_t add, nr, nr2, tmp, out1, out2;

  uint32_t thread_index = blockIdx.x*blockDim.x + threadIdx.x; 
  uint32_t *DEVICE_Hashes_32 = (uint32_t *)DEVICE_Hashes; 
  unsigned char p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15; 
  UINT4 password_count = 0;
  __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN]; 
  __shared__ __align__(16) unsigned char sharedBitmap[8192]; 
  __shared__ __align__(8) unsigned char sharedLengths[16]; 
  copyCharsetAndBitmap(sharedCharset, sharedBitmap, sharedLengths, charsetLen, pass_length); 
  loadStartPositions(pass_length, thread_index, DEVICE_Start_Positions,  
		   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15); 
  while (password_count < count) {
      // Init constants
      nr = (uint32_t)1345345333;
      add = 7;
      nr2 = 0x12345671;

      tmp  = (uint32_t) (uint8_t) sharedCharset[p0];
      nr  ^= (((nr & 63) + add) * tmp) + (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;
      tmp  = (uint32_t) (uint8_t) sharedCharset[p1];
      nr  ^= (((nr & 63) + add) * tmp) + (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;
      tmp  = (uint32_t) (uint8_t) sharedCharset[p2];
      nr  ^= (((nr & 63) + add) * tmp) + (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;

      out1 = nr & (((uint32_t) 1L << 31) -1L);
      out2 = nr2 & (((uint32_t) 1L << 31) -1L);

      /*printf("Pass: %c%c%c  Hash: %08x%08x\n",
        sharedCharset[p0], sharedCharset[p1], sharedCharset[p2],
        out1, out2);*/

      out1 = (out1 & 0xff) << 24 | ((out1 >> 8) & 0xff) << 16 | ((out1 >> 16) & 0xff) << 8 | ((out1 >> 24) & 0xff);
      out2 = (out2 & 0xff) << 24 | ((out2 >> 8) & 0xff) << 16 | ((out2 >> 16) & 0xff) << 8 | ((out2 >> 24) & 0xff);

checkHashMySQL323(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords,
		DEVICE_Hashes_32, success, OutputPassword,
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
		out1, out2);
  password_count++; 
  incrementCounters3Multi();
  } 
}


__global__ void CUDA_MYSQL323_Search_8 (unsigned char *OutputPassword, unsigned char *success,
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions,
				unsigned int count, unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) {
  const int pass_length = 8;
  uint32_t add, nr, nr2, tmp, out1, out2;

  uint32_t thread_index = blockIdx.x*blockDim.x + threadIdx.x;
  uint32_t *DEVICE_Hashes_32 = (uint32_t *)DEVICE_Hashes;
  unsigned char p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15;
  UINT4 password_count = 0;
  __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN];
  __shared__ __align__(16) unsigned char sharedBitmap[8192];
  __shared__ __align__(8) unsigned char sharedLengths[16];
  copyCharsetAndBitmap(sharedCharset, sharedBitmap, sharedLengths, charsetLen, pass_length);
  loadStartPositions(pass_length, thread_index, DEVICE_Start_Positions,
		   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15);
  while (password_count < count) {
      // Init constants
      nr = (uint32_t)1345345333;
      add = 7;
      nr2 = 0x12345671;

      tmp  = (uint32_t) (uint8_t) sharedCharset[p0];
      nr  ^= (((nr & 63) + add) * tmp) + (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;
      tmp  = (uint32_t) (uint8_t) sharedCharset[p1];
      nr  ^= (((nr & 63) + add) * tmp) + (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;
      tmp  = (uint32_t) (uint8_t) sharedCharset[p2];
      nr  ^= (((nr & 63) + add) * tmp) + (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;
      tmp  = (uint32_t) (uint8_t) sharedCharset[p4];
      nr  ^= (((nr & 63) + add) * tmp) + (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;
      tmp  = (uint32_t) (uint8_t) sharedCharset[p5];
      nr  ^= (((nr & 63) + add) * tmp) + (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;
      tmp  = (uint32_t) (uint8_t) sharedCharset[p6];
      nr  ^= (((nr & 63) + add) * tmp) + (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;
      tmp  = (uint32_t) (uint8_t) sharedCharset[p7];
      nr  ^= (((nr & 63) + add) * tmp) + (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;
      tmp  = (uint32_t) (uint8_t) sharedCharset[p8];
      nr  ^= (((nr & 63) + add) * tmp) + (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;

      out1 = nr & (((uint32_t) 1L << 31) -1L);
      out2 = nr2 & (((uint32_t) 1L << 31) -1L);

      /*printf("Pass: %c%c%c  Hash: %08x%08x\n",
        sharedCharset[p0], sharedCharset[p1], sharedCharset[p2],
        out1, out2);*/

      out1 = (out1 & 0xff) << 24 | ((out1 >> 8) & 0xff) << 16 | ((out1 >> 16) & 0xff) << 8 | ((out1 >> 24) & 0xff);
      out2 = (out2 & 0xff) << 24 | ((out2 >> 8) & 0xff) << 16 | ((out2 >> 16) & 0xff) << 8 | ((out2 >> 24) & 0xff);

checkHashMySQL323(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords,
		DEVICE_Hashes_32, success, OutputPassword,
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
		out1, out2);
  password_count++;
  incrementCounters8Multi();
  }
}

// Copy the shared variables to the host
extern "C" void copyMySQL323DataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId) {
    //printf("Thread %d in CHHashTypeNTLM.cu, copyNTLMDataToCharset()\n", threadId);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceCharset, hostCharset, (MAX_CHARSET_LENGTH * charsetLength)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(constantBitmap, hostSharedBitmap, 8192));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(charsetLengths, hostCharsetLengths, 16));
}

extern "C" void Launch_CUDA_MySQL323_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
						unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions, uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes, unsigned char *DEVICE_Bitmap) {

    /*if (passlength == 1) {
	  CUDA_NTLM_Search_1 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 2) {
	  CUDA_NTLM_Search_2 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else*/ if (passlength == 3) {
	  CUDA_MYSQL323_Search_3 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	}/* else if (passlength == 4) {
	  CUDA_NTLM_Search_4 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 5) {
	  CUDA_NTLM_Search_5 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 6) {
	  CUDA_NTLM_Search_6 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 7) {
	  CUDA_NTLM_Search_7 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	}*/ else if (passlength == 8) {
	  CUDA_MYSQL323_Search_8 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	}/* else if (passlength == 9) {
	  CUDA_NTLM_Search_9 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 10) {
	  CUDA_NTLM_Search_10 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 11) {
	  CUDA_NTLM_Search_11 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 12) {
	  CUDA_NTLM_Search_12 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 13) {
	  CUDA_NTLM_Search_13 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 14) {
	  CUDA_NTLM_Search_14 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 15) {
	  CUDA_NTLM_Search_15 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 16) {
	  CUDA_NTLM_Search_16 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	}*/

	cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
      {
        sprintf(global_interface.exit_message, "Cuda error: %s.\n", cudaGetErrorString( err) );
        global_interface.exit = 1;
        return;
      }
}