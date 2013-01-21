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

#include "CUDA_Common/CUDASHA256.h"


__device__ inline void checkHashMultiSHA256Long(int pass_length, unsigned char *sharedBitmap, unsigned char *DEVICE_HashTable, uint32_t numberOfPasswords,
		uint32_t *DEVICE_Hashes_32, unsigned char *success, unsigned char *OutputPassword,
		unsigned char p0, unsigned char p1, unsigned char p2, unsigned char p3,
		unsigned char p4, unsigned char p5, unsigned char p6, unsigned char p7,
		unsigned char p8, unsigned char p9, unsigned char p10, unsigned char p11,
		unsigned char p12, unsigned char p13, unsigned char p14, unsigned char p15,
                unsigned char p16, unsigned char p17, unsigned char p18, unsigned char p19,
                unsigned char p20, unsigned char p21, unsigned char p22, unsigned char p23,
                unsigned char p24, unsigned char p25, unsigned char p26, unsigned char p27,
                unsigned char p28, unsigned char p29, unsigned char p30, unsigned char p31,
                unsigned char p32, unsigned char p33, unsigned char p34, unsigned char p35,
                unsigned char p36, unsigned char p37, unsigned char p38, unsigned char p39,
                unsigned char p40, unsigned char p41, unsigned char p42, unsigned char p43,
                unsigned char p44, unsigned char p45, unsigned char p46, unsigned char p47,
		UINT4 a, UINT4 b, UINT4 c, UINT4 d, UINT4 e,
		uint32_t &search_index, uint32_t &search_high, uint32_t &search_low, uint32_t &hash_order_a,
		uint32_t &hash_order_mem, uint32_t &temp) {
    //cuPrintf("num hashes: %u\n", numberOfPasswords);
  //uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp;
  if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) {
      //cuPrintf("Bitmap hit a = 0x%08x.\n", a);
  if ((!DEVICE_HashTable) || (DEVICE_HashTable && (DEVICE_HashTable[a >> 3] >> (a & 0x00000007)) & 0x00000001)) {
  // Init binary search through global password space
      
  search_high = numberOfPasswords;
  search_low = 0;
  search_index = 0;
  while (search_low < search_high) {
    // Midpoint between search_high and search_low
    search_index = search_low + (search_high - search_low) / 2;
    // reorder from low endian to big endian for searching, as hashes are sorted by byte.
    temp = DEVICE_Hashes_32[8 * search_index];
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

  while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 8])) {
    search_index--;
  }
  //cuPrintf("Found possible match at %d\n", search_index);

  while ((a == DEVICE_Hashes_32[search_index * 8])) {
  if (a == DEVICE_Hashes_32[search_index * 8]) {
    if (b == DEVICE_Hashes_32[search_index * 8 + 1]) {
      if (c == DEVICE_Hashes_32[search_index * 8 + 2]) {
        if (d == DEVICE_Hashes_32[search_index * 8 + 3]) {
        if (e == DEVICE_Hashes_32[search_index * 8 + 4]) {
            //cuPrintf("HIT!\n");
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
        if (pass_length >= 17) OutputPassword[search_index * MAX_PASSWORD_LEN + 16] = deviceCharset[p16 + (MAX_CHARSET_LENGTH * 16)];
        if (pass_length >= 18) OutputPassword[search_index * MAX_PASSWORD_LEN + 17] = deviceCharset[p17 + (MAX_CHARSET_LENGTH * 17)];
        if (pass_length >= 19) OutputPassword[search_index * MAX_PASSWORD_LEN + 18] = deviceCharset[p18 + (MAX_CHARSET_LENGTH * 18)];
        if (pass_length >= 20) OutputPassword[search_index * MAX_PASSWORD_LEN + 19] = deviceCharset[p19 + (MAX_CHARSET_LENGTH * 19)];
        if (pass_length >= 21) OutputPassword[search_index * MAX_PASSWORD_LEN + 20] = deviceCharset[p20 + (MAX_CHARSET_LENGTH * 20)];
        if (pass_length >= 22) OutputPassword[search_index * MAX_PASSWORD_LEN + 21] = deviceCharset[p21 + (MAX_CHARSET_LENGTH * 21)];
        if (pass_length >= 23) OutputPassword[search_index * MAX_PASSWORD_LEN + 22] = deviceCharset[p22 + (MAX_CHARSET_LENGTH * 22)];
        if (pass_length >= 24) OutputPassword[search_index * MAX_PASSWORD_LEN + 23] = deviceCharset[p23 + (MAX_CHARSET_LENGTH * 23)];
        if (pass_length >= 25) OutputPassword[search_index * MAX_PASSWORD_LEN + 24] = deviceCharset[p24 + (MAX_CHARSET_LENGTH * 24)];
        if (pass_length >= 26) OutputPassword[search_index * MAX_PASSWORD_LEN + 25] = deviceCharset[p25 + (MAX_CHARSET_LENGTH * 25)];
        if (pass_length >= 27) OutputPassword[search_index * MAX_PASSWORD_LEN + 26] = deviceCharset[p26 + (MAX_CHARSET_LENGTH * 26)];
        if (pass_length >= 28) OutputPassword[search_index * MAX_PASSWORD_LEN + 27] = deviceCharset[p27 + (MAX_CHARSET_LENGTH * 27)];
        if (pass_length >= 29) OutputPassword[search_index * MAX_PASSWORD_LEN + 28] = deviceCharset[p28 + (MAX_CHARSET_LENGTH * 28)];
        if (pass_length >= 30) OutputPassword[search_index * MAX_PASSWORD_LEN + 29] = deviceCharset[p29 + (MAX_CHARSET_LENGTH * 29)];
        if (pass_length >= 31) OutputPassword[search_index * MAX_PASSWORD_LEN + 30] = deviceCharset[p30 + (MAX_CHARSET_LENGTH * 30)];
        if (pass_length >= 32) OutputPassword[search_index * MAX_PASSWORD_LEN + 31] = deviceCharset[p31 + (MAX_CHARSET_LENGTH * 31)];
        if (pass_length >= 33) OutputPassword[search_index * MAX_PASSWORD_LEN + 32] = deviceCharset[p32 + (MAX_CHARSET_LENGTH * 32)];
        if (pass_length >= 34) OutputPassword[search_index * MAX_PASSWORD_LEN + 33] = deviceCharset[p33 + (MAX_CHARSET_LENGTH * 33)];
        if (pass_length >= 35) OutputPassword[search_index * MAX_PASSWORD_LEN + 34] = deviceCharset[p34 + (MAX_CHARSET_LENGTH * 34)];
        if (pass_length >= 36) OutputPassword[search_index * MAX_PASSWORD_LEN + 35] = deviceCharset[p35 + (MAX_CHARSET_LENGTH * 35)];
        if (pass_length >= 37) OutputPassword[search_index * MAX_PASSWORD_LEN + 36] = deviceCharset[p36 + (MAX_CHARSET_LENGTH * 36)];
        if (pass_length >= 38) OutputPassword[search_index * MAX_PASSWORD_LEN + 37] = deviceCharset[p37 + (MAX_CHARSET_LENGTH * 37)];
        if (pass_length >= 39) OutputPassword[search_index * MAX_PASSWORD_LEN + 38] = deviceCharset[p38 + (MAX_CHARSET_LENGTH * 38)];
        if (pass_length >= 40) OutputPassword[search_index * MAX_PASSWORD_LEN + 39] = deviceCharset[p39 + (MAX_CHARSET_LENGTH * 39)];
        if (pass_length >= 41) OutputPassword[search_index * MAX_PASSWORD_LEN + 40] = deviceCharset[p40 + (MAX_CHARSET_LENGTH * 40)];
        if (pass_length >= 42) OutputPassword[search_index * MAX_PASSWORD_LEN + 41] = deviceCharset[p41 + (MAX_CHARSET_LENGTH * 41)];
        if (pass_length >= 43) OutputPassword[search_index * MAX_PASSWORD_LEN + 42] = deviceCharset[p42 + (MAX_CHARSET_LENGTH * 42)];
        if (pass_length >= 44) OutputPassword[search_index * MAX_PASSWORD_LEN + 43] = deviceCharset[p43 + (MAX_CHARSET_LENGTH * 43)];
        if (pass_length >= 45) OutputPassword[search_index * MAX_PASSWORD_LEN + 44] = deviceCharset[p44 + (MAX_CHARSET_LENGTH * 44)];
        if (pass_length >= 46) OutputPassword[search_index * MAX_PASSWORD_LEN + 45] = deviceCharset[p45 + (MAX_CHARSET_LENGTH * 45)];
        if (pass_length >= 47) OutputPassword[search_index * MAX_PASSWORD_LEN + 46] = deviceCharset[p46 + (MAX_CHARSET_LENGTH * 46)];
        if (pass_length >= 48) OutputPassword[search_index * MAX_PASSWORD_LEN + 47] = deviceCharset[p47 + (MAX_CHARSET_LENGTH * 47)];
        success[search_index] = (unsigned char) 1;
		}
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
/*
__global__ void CUDA_SHA256_Search_3 (unsigned char *OutputPassword, unsigned char *success, \
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, unsigned int count, unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { \
  const int pass_length = 3; \
  uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
  uint32_t a,b,c,d,e,f,g,h; \
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
  cuPrintf("b0: %08x\n", b0); \
  cuPrintf("pass: %c%c%c\n", sharedCharset[p0], sharedCharset[p1], sharedCharset[p2]); \
  CUDA_SHA256(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,a,b,c,d,e,f,g,h); \
  cuPrintf("a: %08x b: %08x c: %08x ...\n", a, b, c);
  checkHashMultiSHA256Long(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords, \
		DEVICE_Hashes_32, success, OutputPassword, \
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
                p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, \
                p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43, \
                p44, p45, p46, p47, \
		a, b, c, d, e, b0, b1, b2, b3, b4, b5); \
  password_count++; \
  incrementCounters3Multi(); \
  } \
}*/

#define CUDA_SHA256_KERNEL_CREATE(length) \
__global__ void CUDA_SHA256_Search_##length (unsigned char *OutputPassword, unsigned char *success, \
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, unsigned int count, unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { \
  const int pass_length = length; \
  uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
  uint32_t a,b,c,d,e,f,g,h; \
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
  CUDA_SHA256(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,a,b,c,d,e,f,g,h); \
  checkHashMultiSHA256Long(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords, \
		DEVICE_Hashes_32, success, OutputPassword, \
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
                p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, \
                p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43, \
                p44, p45, p46, p47, \
		a, b, c, d, e, b0, b1, b2, b3, b4, b5); \
  password_count++; \
  incrementCounters##length##Multi(); \
  } \
}




CUDA_SHA256_KERNEL_CREATE(1)
CUDA_SHA256_KERNEL_CREATE(2)
CUDA_SHA256_KERNEL_CREATE(3)
CUDA_SHA256_KERNEL_CREATE(4)
CUDA_SHA256_KERNEL_CREATE(5)
CUDA_SHA256_KERNEL_CREATE(6)
CUDA_SHA256_KERNEL_CREATE(7)
CUDA_SHA256_KERNEL_CREATE(8)
CUDA_SHA256_KERNEL_CREATE(9)
CUDA_SHA256_KERNEL_CREATE(10)
/*
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
*/

// Copy the shared variables to the host
extern "C" void copySHA256DataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId) {
    //printf("Thread %d in CHHashTypeNTLM.cu, copyNTLMDataToCharset()\n", threadId);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceCharset, hostCharset, (MAX_CHARSET_LENGTH * charsetLength)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(constantBitmap, hostSharedBitmap, 8192));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(charsetLengths, hostCharsetLengths, 16));
    //cudaPrintfInit();
}



extern "C" void Launch_CUDA_SHA256_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
        unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions, uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes, unsigned char *DEVICE_Bitmap) {
    if (passlength == 1) {
	  CUDA_SHA256_Search_1 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 2) {
	  CUDA_SHA256_Search_2 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 3) {
	  CUDA_SHA256_Search_3 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 4) {
	  CUDA_SHA256_Search_4 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 5) {
	  CUDA_SHA256_Search_5 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 6) {
	  CUDA_SHA256_Search_6 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 7) {
	  CUDA_SHA256_Search_7 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 8) {
	  CUDA_SHA256_Search_8 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 9) {
	  CUDA_SHA256_Search_9 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 10) {
	  CUDA_SHA256_Search_10 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} /*else if (passlength == 11) {
	  CUDA_SHA256_Search_11 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 12) {
	  CUDA_SHA256_Search_12 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 13) {
	  CUDA_SHA256_Search_13 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 14) {
	  CUDA_SHA256_Search_14 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 15) {
	  CUDA_SHA256_Search_15 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 16) {
	  CUDA_SHA256_Search_16 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 17) {
          CUDA_SHA256_Search_17 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 18) {
          CUDA_SHA256_Search_18 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 19) {
          CUDA_SHA256_Search_19 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 20) {
          CUDA_SHA256_Search_20 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 21) {
          CUDA_SHA256_Search_21 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 22) {
          CUDA_SHA256_Search_22 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 23) {
          CUDA_SHA256_Search_23 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 24) {
          CUDA_SHA256_Search_24 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 25) {
          CUDA_SHA256_Search_25 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 26) {
          CUDA_SHA256_Search_26 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 27) {
          CUDA_SHA256_Search_27 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 28) {
          CUDA_SHA256_Search_28 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 29) {
          CUDA_SHA256_Search_29 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 30) {
          CUDA_SHA256_Search_30 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 31) {
          CUDA_SHA256_Search_31 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 32) {
          CUDA_SHA256_Search_32 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 33) {
          CUDA_SHA256_Search_33 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 34) {
          CUDA_SHA256_Search_34 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 35) {
          CUDA_SHA256_Search_35 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 36) {
          CUDA_SHA256_Search_36 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 37) {
          CUDA_SHA256_Search_37 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 38) {
          CUDA_SHA256_Search_38 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 39) {
          CUDA_SHA256_Search_39 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 40) {
          CUDA_SHA256_Search_40 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 41) {
          CUDA_SHA256_Search_41 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 42) {
          CUDA_SHA256_Search_42 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 43) {
          CUDA_SHA256_Search_43 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 44) {
          CUDA_SHA256_Search_44 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 45) {
          CUDA_SHA256_Search_45 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 46) {
          CUDA_SHA256_Search_46 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 47) {
          CUDA_SHA256_Search_47 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        } else if (passlength == 48) {
          CUDA_SHA256_Search_48 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
        }*/ else {
            sprintf(global_interface.exit_message, "SHA256 length >48 not currently supported!\n");
            global_interface.exit = 1;
            return;
        }
        //cudaPrintfDisplay(stdout, true);
	cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
      {
        sprintf(global_interface.exit_message, "Cuda error: %s.\n", cudaGetErrorString( err) );
        global_interface.exit = 1;
        return;
      }
}
