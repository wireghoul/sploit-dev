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

// Common CUDA defines/etc for the various hash files.

#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include "Multiforcer_CUDA_device/CUDAIncrementers.h"

// If the UseB0-B13 stuff is not defined, use all of them.
#ifndef UseB0
    #define UseB0 1
    #define UseB1 1
    #define UseB2 1
    #define UseB3 1
    #define UseB4 1
    #define UseB5 1
    #define UseB6 1
    #define UseB7 1
    #define UseB8 1
    #define UseB9 1
    #define UseB10 1
    #define UseB11 1
    #define UseB12 1
    #define UseB13 1
#endif

// For "hash to ascii" conversion in algorithms

__constant__ char values[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};


__device__ inline void clearB0toB15(UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
               UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {
  b0 = 0x00000000;
  b1 = 0x00000000;
  b2 = 0x00000000;
  b3 = 0x00000000;
  b4 = 0x00000000;
  b5 = 0x00000000;
  b6 = 0x00000000;
  b7 = 0x00000000;
  b8 = 0x00000000;
  b9 = 0x00000000;
  b10 = 0x00000000;
  b11 = 0x00000000;
  b12 = 0x00000000;
  b13 = 0x00000000;
  b14 = 0x00000000;
  b15 = 0x00000000;
}



__device__ inline void copyCharsetAndBitmap(unsigned char *sharedCharset, unsigned char *sharedBitmap, unsigned char *sharedLengths, int charsetLen, int pass_length) {
  uint64_t *sharedCharsetCoalesce = (uint64_t *)sharedCharset;
  uint64_t *deviceCharsetCoalesce = (uint64_t *)deviceCharset;

  uint64_t *sharedBitmapCoalesce = (uint64_t *)sharedBitmap;
  uint64_t *deviceBitmapCoalesce = (uint64_t *)constantBitmap;


  int a, b;

  
  // This has to be alignable somehow...
  if (threadIdx.x < MAX_PASSWORD_LEN) {
      //printf("Shared length %d: %d\n", threadIdx.x, charsetLengths[threadIdx.x]);
      sharedLengths[threadIdx.x] = charsetLengths[threadIdx.x];
  }

  for (b = 0; b < pass_length; b++) {
    if (threadIdx.x <= (MAX_CHARSET_LENGTH /*charsetLengths[b]*/ / 8)) {
        sharedCharsetCoalesce[threadIdx.x + (b * (MAX_CHARSET_LENGTH / sizeof(uint64_t)))]
                = deviceCharsetCoalesce[threadIdx.x + (b * (MAX_CHARSET_LENGTH / sizeof(uint64_t)))];
    }
  }

  for (a = 0; a <= ((8192 / 8) / blockDim.x); a++) {
    if ((a * blockDim.x + threadIdx.x) < (8192 / 8)) {
        sharedBitmapCoalesce[a * blockDim.x + threadIdx.x] = deviceBitmapCoalesce[a * blockDim.x + threadIdx.x];
    }
  }
//  for (a = 0; a < (8192 / 8); a++) {
//      sharedBitmapCoalesce[a] = deviceBitmapCoalesce[a];
//  }

  // Make sure everyone is here and done before we return.
  syncthreads();
}

__device__ inline void copyCharset(unsigned char *sharedCharset, unsigned char *sharedLengths, int charsetLen, int pass_length) {
  uint64_t *sharedCharsetCoalesce = (uint64_t *)sharedCharset;
  uint64_t *deviceCharsetCoalesce = (uint64_t *)deviceCharset;

  int b;
  // This has to be alignable somehow...
  if (threadIdx.x < MAX_PASSWORD_LEN) {
      //printf("Shared length %d: %d\n", threadIdx.x, charsetLengths[threadIdx.x]);
      sharedLengths[threadIdx.x] = charsetLengths[threadIdx.x];
  }

  for (b = 0; b < pass_length; b++) {
    if (threadIdx.x <= (MAX_CHARSET_LENGTH /*charsetLengths[b]*/ / 8)) {
        sharedCharsetCoalesce[threadIdx.x + (b * (MAX_CHARSET_LENGTH / sizeof(uint64_t)))]
                = deviceCharsetCoalesce[threadIdx.x + (b * (MAX_CHARSET_LENGTH / sizeof(uint64_t)))];
    }
  }
  // Make sure everyone is here and done before we return.
  syncthreads();
}


// Loads start positions.  Should work on all hashes.
__device__ inline void loadStartPositions(int count, uint32_t &thread_index, struct start_positions *DEVICE_Start_Positions,
        unsigned char &p0, unsigned char &p1, unsigned char &p2, unsigned char &p3,
        unsigned char &p4, unsigned char &p5, unsigned char &p6, unsigned char &p7,
        unsigned char &p8, unsigned char &p9, unsigned char &p10, unsigned char &p11,
        unsigned char &p12, unsigned char &p13, unsigned char &p14, unsigned char &p15) {
  p0 = 0;
  p1 = 0;
  p2 = 0;
  p3 = 0;
  p4 = 0;
  p5 = 0;
  p6 = 0;
  p7 = 0;
  p8 = 0;
  p9 = 0;
  p10 = 0;
  p11 = 0;
  p12 = 0;
  p13 = 0;
  p14 = 0;
  p15 = 0;

  p0 = DEVICE_Start_Positions[thread_index].p0;
  if (count <= 1) return;
  p1 = DEVICE_Start_Positions[thread_index].p1;
  if (count <= 2) return;
  p2 = DEVICE_Start_Positions[thread_index].p2;
  if (count <= 3) return;
  p3 = DEVICE_Start_Positions[thread_index].p3;
  if (count <= 4) return;
  p4 = DEVICE_Start_Positions[thread_index].p4;
  if (count <= 5) return;
  p5 = DEVICE_Start_Positions[thread_index].p5;
  if (count <= 6) return;
  p6 = DEVICE_Start_Positions[thread_index].p6;
  if (count <= 7) return;
  p7 = DEVICE_Start_Positions[thread_index].p7;
  if (count <= 8) return;
  p8 = DEVICE_Start_Positions[thread_index].p8;
  if (count <= 9) return;
  p9 = DEVICE_Start_Positions[thread_index].p9;
  if (count <= 10) return;
  p10 = DEVICE_Start_Positions[thread_index].p10;
  if (count <= 11) return;
  p11 = DEVICE_Start_Positions[thread_index].p11;
  if (count <= 12) return;
  p12 = DEVICE_Start_Positions[thread_index].p12;
  if (count <= 13) return;
  p13 = DEVICE_Start_Positions[thread_index].p13;
  if (count <= 14) return;
  p14 = DEVICE_Start_Positions[thread_index].p14;
  if (count <= 15) return;
  p15 = DEVICE_Start_Positions[thread_index].p15;
  if (count <= 16) return;
}

__device__ inline void loadStartPositionsLong(int count, uint32_t &thread_index, struct start_positions *DEVICE_Start_Positions,
        unsigned char &p0, unsigned char &p1, unsigned char &p2, unsigned char &p3,
        unsigned char &p4, unsigned char &p5, unsigned char &p6, unsigned char &p7,
        unsigned char &p8, unsigned char &p9, unsigned char &p10, unsigned char &p11,
        unsigned char &p12, unsigned char &p13, unsigned char &p14, unsigned char &p15,
        unsigned char &p16, unsigned char &p17, unsigned char &p18, unsigned char &p19,
        unsigned char &p20, unsigned char &p21, unsigned char &p22, unsigned char &p23,
        unsigned char &p24, unsigned char &p25, unsigned char &p26, unsigned char &p27,
        unsigned char &p28, unsigned char &p29, unsigned char &p30, unsigned char &p31,
        unsigned char &p32, unsigned char &p33, unsigned char &p34, unsigned char &p35,
        unsigned char &p36, unsigned char &p37, unsigned char &p38, unsigned char &p39,
        unsigned char &p40, unsigned char &p41, unsigned char &p42, unsigned char &p43,
        unsigned char &p44, unsigned char &p45, unsigned char &p46, unsigned char &p47
        ) {
  p0 = 0;
  p1 = 0;
  p2 = 0;
  p3 = 0;
  p4 = 0;
  p5 = 0;
  p6 = 0;
  p7 = 0;
  p8 = 0;
  p9 = 0;
  p10 = 0;
  p11 = 0;
  p12 = 0;
  p13 = 0;
  p14 = 0;
  p15 = 0;
  p16 = 0;
  p17 = 0;
  p18 = 0;
  p19 = 0;
  p20 = 0;
  p21 = 0;
  p22 = 0;
  p23 = 0;
  p24 = 0;
  p25 = 0;
  p26 = 0;
  p27 = 0;
  p28 = 0;
  p29 = 0;
  p30 = 0;
  p31 = 0;
  p32 = 0;
  p33 = 0;
  p34 = 0;
  p35 = 0;
  p36 = 0;
  p37 = 0;
  p38 = 0;
  p39 = 0;
  p40 = 0;
  p41 = 0;
  p42 = 0;
  p43 = 0;
  p44 = 0;
  p45 = 0;
  p46 = 0;
  p47 = 0;

  p0 = DEVICE_Start_Positions[thread_index].p0;
  if (count <= 1) return;
  p1 = DEVICE_Start_Positions[thread_index].p1;
  if (count <= 2) return;
  p2 = DEVICE_Start_Positions[thread_index].p2;
  if (count <= 3) return;
  p3 = DEVICE_Start_Positions[thread_index].p3;
  if (count <= 4) return;
  p4 = DEVICE_Start_Positions[thread_index].p4;
  if (count <= 5) return;
  p5 = DEVICE_Start_Positions[thread_index].p5;
  if (count <= 6) return;
  p6 = DEVICE_Start_Positions[thread_index].p6;
  if (count <= 7) return;
  p7 = DEVICE_Start_Positions[thread_index].p7;
  if (count <= 8) return;
  p8 = DEVICE_Start_Positions[thread_index].p8;
  if (count <= 9) return;
  p9 = DEVICE_Start_Positions[thread_index].p9;
  if (count <= 10) return;
  p10 = DEVICE_Start_Positions[thread_index].p10;
  if (count <= 11) return;
  p11 = DEVICE_Start_Positions[thread_index].p11;
  if (count <= 12) return;
  p12 = DEVICE_Start_Positions[thread_index].p12;
  if (count <= 13) return;
  p13 = DEVICE_Start_Positions[thread_index].p13;
  if (count <= 14) return;
  p14 = DEVICE_Start_Positions[thread_index].p14;
  if (count <= 15) return;
  p15 = DEVICE_Start_Positions[thread_index].p15;
  if (count <= 16) return;
  p16 = DEVICE_Start_Positions[thread_index].p16;
  if (count <= 17) return;
  p17 = DEVICE_Start_Positions[thread_index].p17;
  if (count <= 18) return;
  p18 = DEVICE_Start_Positions[thread_index].p18;
  if (count <= 19) return;
  p19 = DEVICE_Start_Positions[thread_index].p19;
  if (count <= 20) return;
  p20 = DEVICE_Start_Positions[thread_index].p20;
  if (count <= 21) return;
  p21 = DEVICE_Start_Positions[thread_index].p21;
  if (count <= 22) return;
  p22 = DEVICE_Start_Positions[thread_index].p22;
  if (count <= 23) return;
  p23 = DEVICE_Start_Positions[thread_index].p23;
  if (count <= 24) return;
  p24 = DEVICE_Start_Positions[thread_index].p24;
  if (count <= 25) return;
  p25 = DEVICE_Start_Positions[thread_index].p25;
  if (count <= 26) return;
  p26 = DEVICE_Start_Positions[thread_index].p26;
  if (count <= 27) return;
  p27 = DEVICE_Start_Positions[thread_index].p27;
  if (count <= 28) return;
  p28 = DEVICE_Start_Positions[thread_index].p28;
  if (count <= 29) return;
  p29 = DEVICE_Start_Positions[thread_index].p29;
  if (count <= 30) return;
  p30 = DEVICE_Start_Positions[thread_index].p30;
  if (count <= 31) return;
  p31 = DEVICE_Start_Positions[thread_index].p31;
  if (count <= 32) return;
  p32 = DEVICE_Start_Positions[thread_index].p32;
  if (count <= 33) return;
  p33 = DEVICE_Start_Positions[thread_index].p33;
  if (count <= 34) return;
  p34 = DEVICE_Start_Positions[thread_index].p34;
  if (count <= 35) return;
  p35 = DEVICE_Start_Positions[thread_index].p35;
  if (count <= 36) return;
  p36 = DEVICE_Start_Positions[thread_index].p36;
  if (count <= 37) return;
  p37 = DEVICE_Start_Positions[thread_index].p37;
  if (count <= 38) return;
  p38 = DEVICE_Start_Positions[thread_index].p38;
  if (count <= 39) return;
  p39 = DEVICE_Start_Positions[thread_index].p39;
  if (count <= 40) return;
  p40 = DEVICE_Start_Positions[thread_index].p40;
  if (count <= 41) return;
  p41 = DEVICE_Start_Positions[thread_index].p41;
  if (count <= 42) return;
  p42 = DEVICE_Start_Positions[thread_index].p42;
  if (count <= 43) return;
  p43 = DEVICE_Start_Positions[thread_index].p43;
  if (count <= 44) return;
  p44 = DEVICE_Start_Positions[thread_index].p44;
  if (count <= 45) return;
  p45 = DEVICE_Start_Positions[thread_index].p45;
  if (count <= 46) return;
  p46 = DEVICE_Start_Positions[thread_index].p46;
  if (count <= 47) return;
  p47 = DEVICE_Start_Positions[thread_index].p47;
  if (count <= 48) return;
}


// initMD loads the charset values into the working blocks for MD4/MD5
// length : length in bytes to load
__device__ inline void initMDSingleCharset(int length, unsigned char *sharedCharset,
        unsigned char &p0, unsigned char &p1, unsigned char &p2, unsigned char &p3,
        unsigned char &p4, unsigned char &p5, unsigned char &p6, unsigned char &p7,
        unsigned char &p8, unsigned char &p9, unsigned char &p10, unsigned char &p11,
        unsigned char &p12, unsigned char &p13, unsigned char &p14, unsigned char &p15,
        UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
        UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

  clearB0toB15(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);

  // Set length properly (length in bits)
  b14 = length * 8;

  if (length == 0) {
    b0 |= 0x00000080;
    return;
  }
  b0 |= sharedCharset[p0];
  if (length == 1) {
    b0 |= 0x00008000;
    return;
  }
  b0 |= sharedCharset[p1] << 8;
  if (length == 2) {
    b0 |= 0x00800000;
    return;
  }
  b0 |= sharedCharset[p2] << 16;
  if (length == 3) {
    b0 |= 0x80000000;
    return;
  }
  b0 |= sharedCharset[p3] << 24;
  if (length == 4) {
    b1 |= 0x00000080;
    return;
  }
  b1 |= sharedCharset[p4];
  if (length == 5) {
    b1 |= 0x00008000;
    return;
  }
  b1 |= sharedCharset[p5] << 8;
  if (length == 6) {
    b1 |= 0x00800000;
    return;
  }
  b1 |= sharedCharset[p6] << 16;
  if (length == 7) {
    b1 |= 0x80000000;
    return;
  }
  b1 |= sharedCharset[p7] << 24;
  if (length == 8) {
    b2 |= 0x00000080;
    return;
  }
  b2 |= sharedCharset[p8];
  if (length == 9) {
    b2 |= 0x00008000;
    return;
  }
  b2 |= sharedCharset[p9] << 8;
  if (length == 10) {
    b2 |= 0x00800000;
    return;
  }
  b2 |= sharedCharset[p10] << 16;
  if (length == 11) {
   b2 |= 0x80000000;
    return;
  }
  b2 |= sharedCharset[p11] << 24;
  if (length == 12) {
    b3 |= 0x00000080;
    return;
  }
  b3 |= sharedCharset[p12];
  if (length == 13) {
    b3 |= 0x00008000;
    return;
  }
  b3 |= sharedCharset[p13] << 8;
  if (length == 14) {
    b3 |= 0x00800000;
    return;
  }
  b3 |= sharedCharset[p14] << 16;
  if (length == 15) {
    b3 |= 0x80000000;
    return;
  }
  b3 |= sharedCharset[p15] << 24;
  if (length == 16) {
    b4 |= 0x00000080;
    return;
  }
}

// initMD loads the charset values into the working blocks for MD4/MD5
// length : length in bytes to load
__device__ inline void initMD(int length, unsigned char *sharedCharset,
		unsigned char &p0, unsigned char &p1, unsigned char &p2, unsigned char &p3,
		unsigned char &p4, unsigned char &p5, unsigned char &p6, unsigned char &p7,
		unsigned char &p8, unsigned char &p9, unsigned char &p10, unsigned char &p11,
		unsigned char &p12, unsigned char &p13, unsigned char &p14, unsigned char &p15,
		UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
		UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

  clearB0toB15(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);

  // Set length properly (length in bits)
  b14 = length * 8;

  if (length == 0) {
    b0 |= 0x00000080;
    return;
  }
  b0 |= sharedCharset[p0];
  if (length == 1) {
    b0 |= 0x00008000;
    return;
  }
  b0 |= sharedCharset[p1 + MAX_CHARSET_LENGTH * 1] << 8;
  if (length == 2) {
    b0 |= 0x00800000;
    return;
  }
  b0 |= sharedCharset[p2 + MAX_CHARSET_LENGTH * 2] << 16;
  if (length == 3) {
    b0 |= 0x80000000;
    return;
  }
  b0 |= sharedCharset[p3 + MAX_CHARSET_LENGTH * 3] << 24;
  if (length == 4) {
    b1 |= 0x00000080;
    return;
  }
  b1 |= sharedCharset[p4 + MAX_CHARSET_LENGTH * 4];
  if (length == 5) {
    b1 |= 0x00008000;
    return;
  }
  b1 |= sharedCharset[p5 + (MAX_CHARSET_LENGTH * 5)] << 8;
  if (length == 6) {
    b1 |= 0x00800000;
    return;
  }
  b1 |= sharedCharset[p6 + (MAX_CHARSET_LENGTH * 6)] << 16;
  if (length == 7) {
    b1 |= 0x80000000;
    return;
  }
  b1 |= sharedCharset[p7 + (MAX_CHARSET_LENGTH * 7)] << 24;
  if (length == 8) {
    b2 |= 0x00000080;
    return;
  }
  b2 |= sharedCharset[p8 + (MAX_CHARSET_LENGTH * 8)];
  if (length == 9) {
    b2 |= 0x00008000;
    return;
  }
  b2 |= sharedCharset[p9 + (MAX_CHARSET_LENGTH * 9)] << 8;
  if (length == 10) {
    b2 |= 0x00800000;
    return;
  }
  b2 |= sharedCharset[p10 + (MAX_CHARSET_LENGTH * 10)] << 16;
  if (length == 11) {
   b2 |= 0x80000000;
    return;
  }
  b2 |= sharedCharset[p11 + (MAX_CHARSET_LENGTH * 11)] << 24;
  if (length == 12) {
    b3 |= 0x00000080;
    return;
  }
  b3 |= sharedCharset[p12 + (MAX_CHARSET_LENGTH * 12)];
  if (length == 13) {
    b3 |= 0x00008000;
    return;
  }
  b3 |= sharedCharset[p13 + (MAX_CHARSET_LENGTH * 13)] << 8;
  if (length == 14) {
    b3 |= 0x00800000;
    return;
  }
  b3 |= sharedCharset[p14 + (MAX_CHARSET_LENGTH * 14)] << 16;
  if (length == 15) {
    b3 |= 0x80000000;
    return;
  }
  b3 |= sharedCharset[p15 + (MAX_CHARSET_LENGTH * 15)] << 24;
  if (length == 16) {
    b4 |= 0x00000080;
    return;
  }
}


__device__ inline void initSHA1(int length, unsigned char *sharedCharset,
		unsigned char &p0, unsigned char &p1, unsigned char &p2, unsigned char &p3,
		unsigned char &p4, unsigned char &p5, unsigned char &p6, unsigned char &p7,
		unsigned char &p8, unsigned char &p9, unsigned char &p10, unsigned char &p11,
		unsigned char &p12, unsigned char &p13, unsigned char &p14, unsigned char &p15,
		UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
		UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

  clearB0toB15(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);

  // Set length properly (length in bits)
  // Note that this is /different/ from MD!
  b15 = length * 8 << 24;

  if (length == 0) {
    b0 |= 0x00000080;
    return;
  }
  b0 |= sharedCharset[p0];
  if (length == 1) {
    b0 |= 0x00008000;
    return;
  }
  b0 |= sharedCharset[p1 + MAX_CHARSET_LENGTH] << 8;
  if (length == 2) {
    b0 |= 0x00800000;
    return;
  }
  b0 |= sharedCharset[p2 + (MAX_CHARSET_LENGTH * 2)] << 16;
  if (length == 3) {
    b0 |= 0x80000000;
    return;
  }
  b0 |= sharedCharset[p3 + (MAX_CHARSET_LENGTH * 3)] << 24;
  if (length == 4) {
    b1 |= 0x00000080;
    return;
  }
  b1 |= sharedCharset[p4 + (MAX_CHARSET_LENGTH * 4)];
  if (length == 5) {
    b1 |= 0x00008000;
    return;
  }
  b1 |= sharedCharset[p5 + (MAX_CHARSET_LENGTH * 5)] << 8;
  if (length == 6) {
    b1 |= 0x00800000;
    return;
  }
  b1 |= sharedCharset[p6 + (MAX_CHARSET_LENGTH * 6)] << 16;
  if (length == 7) {
    b1 |= 0x80000000;
    return;
  }
  b1 |= sharedCharset[p7 + (MAX_CHARSET_LENGTH * 7)] << 24;
  if (length == 8) {
    b2 |= 0x00000080;
    return;
  }
  b2 |= sharedCharset[p8 + (MAX_CHARSET_LENGTH * 8)];
  if (length == 9) {
    b2 |= 0x00008000;
    return;
  }
  b2 |= sharedCharset[p9 + (MAX_CHARSET_LENGTH * 9)] << 8;
  if (length == 10) {
    b2 |= 0x00800000;
    return;
  }
  b2 |= sharedCharset[p10 + (MAX_CHARSET_LENGTH * 10)] << 16;
  if (length == 11) {
   b2 |= 0x80000000;
    return;
  }
  b2 |= sharedCharset[p11 + (MAX_CHARSET_LENGTH * 11)] << 24;
  if (length == 12) {
    b3 |= 0x00000080;
    return;
  }
  b3 |= sharedCharset[p12 + (MAX_CHARSET_LENGTH * 12)];
  if (length == 13) {
    b3 |= 0x00008000;
    return;
  }
  b3 |= sharedCharset[p13 + (MAX_CHARSET_LENGTH * 13)] << 8;
  if (length == 14) {
    b3 |= 0x00800000;
    return;
  }
  b3 |= sharedCharset[p14 + (MAX_CHARSET_LENGTH * 14)] << 16;
  if (length == 15) {
    b3 |= 0x80000000;
    return;
  }
  b3 |= sharedCharset[p15 + (MAX_CHARSET_LENGTH * 15)] << 24;
  if (length == 16) {
    b4 |= 0x00000080;
    return;
  }
}




// length : length in bytes to load
__device__ inline void initNTLM(int length, unsigned char *sharedCharset,
		unsigned char &p0, unsigned char &p1, unsigned char &p2, unsigned char &p3,
		unsigned char &p4, unsigned char &p5, unsigned char &p6, unsigned char &p7,
		unsigned char &p8, unsigned char &p9, unsigned char &p10, unsigned char &p11,
		unsigned char &p12, unsigned char &p13, unsigned char &p14, unsigned char &p15,
		UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
		UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

  clearB0toB15(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);

  // Set length properly (length in bits - zero padded unicode, so twice the "password" length)
  b14 = length * 16;

  if (length == 0) {
    b0 |= 0x00000080;
    return;
  }
  b0 |= sharedCharset[p0];
  if (length == 1) {
    b0 |= 0x00800000;
    return;
  }
  b0 |= sharedCharset[p1 + MAX_CHARSET_LENGTH] << 16;
  if (length == 2) {
    b1 |= 0x00000080;
	return;
  }
  b1 |= sharedCharset[p2 + MAX_CHARSET_LENGTH * 2];
  if (length == 3) {
    b1 |= 0x00800000;
    return;
  }
  b1 |= sharedCharset[p3 + MAX_CHARSET_LENGTH * 3] << 16;
  if (length == 4) {
    b2 |= 0x00000080;
    return;
  }
  b2 |= sharedCharset[p4 + MAX_CHARSET_LENGTH * 4];
  if (length == 5) {
    b2 |= 0x00800000;
    return;
  }
  b2 |= sharedCharset[p5 + (MAX_CHARSET_LENGTH * 5)] << 16;
  if (length == 6) {
    b3 |= 0x00000080;
    return;
  }
  b3 |= sharedCharset[p6 + (MAX_CHARSET_LENGTH * 6)];
  if (length == 7) {
    b3 |= 0x00800000;
    return;
  }
  b3 |= sharedCharset[p7 + (MAX_CHARSET_LENGTH * 7)] << 16;
  if (length == 8) {
    b4 |= 0x00000080;
    return;
  }
  b4 |= sharedCharset[p8 + (MAX_CHARSET_LENGTH * 8)];
  if (length == 9) {
    b4 |= 0x00800000;
    return;
  }
  b4 |= sharedCharset[p9 + (MAX_CHARSET_LENGTH * 9)] << 16;
  if (length == 10) {
    b5 |= 0x00000080;
    return;
  }
  b5 |= sharedCharset[p10 + (MAX_CHARSET_LENGTH * 10)];
  if (length == 11) {
   b5 |= 0x00800000;
    return;
  }
  b5 |= sharedCharset[p11 + (MAX_CHARSET_LENGTH * 11)] << 16;
  if (length == 12) {
    b6 |= 0x00000080;
    return;
  }
  b6 |= sharedCharset[p12 + (MAX_CHARSET_LENGTH * 12)];
  if (length == 13) {
    b6 |= 0x00800000;
    return;
  }
  b6 |= sharedCharset[p13 + (MAX_CHARSET_LENGTH * 13)] << 16;
  if (length == 14) {
    b7 |= 0x00000080;
    return;
  }
  b7 |= sharedCharset[p14 + (MAX_CHARSET_LENGTH * 14)];
  if (length == 15) {
    b7 |= 0x00800000;
    return;
  }
  b7 |= sharedCharset[p15 + (MAX_CHARSET_LENGTH * 15)] << 16;
  if (length == 16) {
    b8 |= 0x00000080;
    return;
  }
}

__device__ inline void checkHashMulti(int pass_length, unsigned char *sharedBitmap, unsigned char *DEVICE_HashTable, uint32_t numberOfPasswords,
		uint32_t *DEVICE_Hashes_32, unsigned char *success, unsigned char *OutputPassword,
		unsigned char p0, unsigned char p1, unsigned char p2, unsigned char p3,
		unsigned char p4, unsigned char p5, unsigned char p6, unsigned char p7,
		unsigned char p8, unsigned char p9, unsigned char p10, unsigned char p11,
		unsigned char p12, unsigned char p13, unsigned char p14, unsigned char p15,
		UINT4 a, UINT4 b, UINT4 c, UINT4 d,
		uint32_t &search_index, uint32_t &search_high, uint32_t &search_low, uint32_t &hash_order_a,
		uint32_t &hash_order_mem, uint32_t &temp) {
  //uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp;
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
    temp = DEVICE_Hashes_32[4 * search_index];
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

  while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 4])) {
    search_index--;
  }


  while ((a == DEVICE_Hashes_32[search_index * 4])) {
  /*if (a == DEVICE_Hashes_32[search_index * 4])*/ {
    if (b == DEVICE_Hashes_32[search_index * 4 + 1]) {
      if (c == DEVICE_Hashes_32[search_index * 4 + 2]) {
        if (d == DEVICE_Hashes_32[search_index * 4 + 3]) {
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


__device__ inline void checkHashMultiLong(int pass_length, unsigned char *sharedBitmap, unsigned char *DEVICE_HashTable, uint32_t numberOfPasswords,
		uint32_t *DEVICE_Hashes_32, unsigned char *success, unsigned char *OutputPassword,
		unsigned char p0, unsigned char p1, unsigned char p2, unsigned char p3,
		unsigned char p4, unsigned char p5, unsigned char p6, unsigned char p7,
		unsigned char p8, unsigned char p9, unsigned char p10, unsigned char p11,
		unsigned char p12, unsigned char p13, unsigned char p14, unsigned char p15,
                unsigned char &p16, unsigned char &p17, unsigned char &p18, unsigned char &p19,
                unsigned char &p20, unsigned char &p21, unsigned char &p22, unsigned char &p23,
                unsigned char &p24, unsigned char &p25, unsigned char &p26, unsigned char &p27,
                unsigned char &p28, unsigned char &p29, unsigned char &p30, unsigned char &p31,
                unsigned char &p32, unsigned char &p33, unsigned char &p34, unsigned char &p35,
                unsigned char &p36, unsigned char &p37, unsigned char &p38, unsigned char &p39,
                unsigned char &p40, unsigned char &p41, unsigned char &p42, unsigned char &p43,
                unsigned char &p44, unsigned char &p45, unsigned char &p46, unsigned char &p47,
		UINT4 a, UINT4 b, UINT4 c, UINT4 d,
		uint32_t &search_index, uint32_t &search_high, uint32_t &search_low, uint32_t &hash_order_a,
		uint32_t &hash_order_mem, uint32_t &temp) {
  //uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp;
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
    temp = DEVICE_Hashes_32[4 * search_index];
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

  while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 4])) {
    search_index--;
  }


  while ((a == DEVICE_Hashes_32[search_index * 4])) {
  /*if (a == DEVICE_Hashes_32[search_index * 4])*/ {
    if (b == DEVICE_Hashes_32[search_index * 4 + 1]) {
      if (c == DEVICE_Hashes_32[search_index * 4 + 2]) {
        if (d == DEVICE_Hashes_32[search_index * 4 + 3]) {
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


__device__ inline void checkDuplicatedHashMultiLong(int pass_length, unsigned char *sharedBitmap, unsigned char *DEVICE_HashTable, uint32_t numberOfPasswords,
		uint32_t *DEVICE_Hashes_32, unsigned char *success, unsigned char *OutputPassword,
		unsigned char p0, unsigned char p1, unsigned char p2, unsigned char p3,
		unsigned char p4, unsigned char p5, unsigned char p6, unsigned char p7,
		unsigned char p8, unsigned char p9, unsigned char p10, unsigned char p11,
		unsigned char p12, unsigned char p13, unsigned char p14, unsigned char p15,
                unsigned char &p16, unsigned char &p17, unsigned char &p18, unsigned char &p19,
                unsigned char &p20, unsigned char &p21, unsigned char &p22, unsigned char &p23,
                unsigned char &p24, unsigned char &p25, unsigned char &p26, unsigned char &p27,
                unsigned char &p28, unsigned char &p29, unsigned char &p30, unsigned char &p31,
                unsigned char &p32, unsigned char &p33, unsigned char &p34, unsigned char &p35,
                unsigned char &p36, unsigned char &p37, unsigned char &p38, unsigned char &p39,
                unsigned char &p40, unsigned char &p41, unsigned char &p42, unsigned char &p43,
                unsigned char &p44, unsigned char &p45, unsigned char &p46, unsigned char &p47,
		UINT4 a, UINT4 b, UINT4 c, UINT4 d,
		uint32_t &search_index, uint32_t &search_high, uint32_t &search_low, uint32_t &hash_order_a,
		uint32_t &hash_order_mem, uint32_t &temp) {
  //uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp;
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
    temp = DEVICE_Hashes_32[4 * search_index];
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

  while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 4])) {
    search_index--;
  }


  while ((a == DEVICE_Hashes_32[search_index * 4])) {
  /*if (a == DEVICE_Hashes_32[search_index * 4])*/ {
    if (b == DEVICE_Hashes_32[search_index * 4 + 1]) {
      if (c == DEVICE_Hashes_32[search_index * 4 + 2]) {
        if (d == DEVICE_Hashes_32[search_index * 4 + 3]) {
        if (pass_length >= 1) OutputPassword[search_index * MAX_PASSWORD_LEN + 0] = deviceCharset[p0];
        if (pass_length >= 1) OutputPassword[search_index * MAX_PASSWORD_LEN + 0 + pass_length] = deviceCharset[p0];
        if (pass_length >= 2) OutputPassword[search_index * MAX_PASSWORD_LEN + 1] = deviceCharset[p1 + (MAX_CHARSET_LENGTH * 1)];
        if (pass_length >= 2) OutputPassword[search_index * MAX_PASSWORD_LEN + 1 + pass_length] = deviceCharset[p1 + (MAX_CHARSET_LENGTH * 1)];
        if (pass_length >= 3) OutputPassword[search_index * MAX_PASSWORD_LEN + 2] = deviceCharset[p2 + (MAX_CHARSET_LENGTH * 2)];
        if (pass_length >= 3) OutputPassword[search_index * MAX_PASSWORD_LEN + 2 + pass_length] = deviceCharset[p2 + (MAX_CHARSET_LENGTH * 2)];
        if (pass_length >= 4) OutputPassword[search_index * MAX_PASSWORD_LEN + 3] = deviceCharset[p3 + (MAX_CHARSET_LENGTH * 3)];
        if (pass_length >= 4) OutputPassword[search_index * MAX_PASSWORD_LEN + 3 + pass_length] = deviceCharset[p3 + (MAX_CHARSET_LENGTH * 3)];
        if (pass_length >= 5) OutputPassword[search_index * MAX_PASSWORD_LEN + 4] = deviceCharset[p4 + (MAX_CHARSET_LENGTH * 4)];
        if (pass_length >= 5) OutputPassword[search_index * MAX_PASSWORD_LEN + 4 + pass_length] = deviceCharset[p4 + (MAX_CHARSET_LENGTH * 4)];
        if (pass_length >= 6) OutputPassword[search_index * MAX_PASSWORD_LEN + 5] = deviceCharset[p5 + (MAX_CHARSET_LENGTH * 5)];
        if (pass_length >= 6) OutputPassword[search_index * MAX_PASSWORD_LEN + 5 + pass_length] = deviceCharset[p5 + (MAX_CHARSET_LENGTH * 5)];
        if (pass_length >= 7) OutputPassword[search_index * MAX_PASSWORD_LEN + 6] = deviceCharset[p6 + (MAX_CHARSET_LENGTH * 6)];
        if (pass_length >= 7) OutputPassword[search_index * MAX_PASSWORD_LEN + 6 + pass_length] = deviceCharset[p6 + (MAX_CHARSET_LENGTH * 6)];
        if (pass_length >= 8) OutputPassword[search_index * MAX_PASSWORD_LEN + 7] = deviceCharset[p7 + (MAX_CHARSET_LENGTH * 7)];
        if (pass_length >= 8) OutputPassword[search_index * MAX_PASSWORD_LEN + 7 + pass_length] = deviceCharset[p7 + (MAX_CHARSET_LENGTH * 7)];
        if (pass_length >= 9) OutputPassword[search_index * MAX_PASSWORD_LEN + 8] = deviceCharset[p8 + (MAX_CHARSET_LENGTH * 8)];
        if (pass_length >= 9) OutputPassword[search_index * MAX_PASSWORD_LEN + 8 + pass_length] = deviceCharset[p8 + (MAX_CHARSET_LENGTH * 8)];
        if (pass_length >= 10) OutputPassword[search_index * MAX_PASSWORD_LEN + 9] = deviceCharset[p9 + (MAX_CHARSET_LENGTH * 9)];
        if (pass_length >= 10) OutputPassword[search_index * MAX_PASSWORD_LEN + 9 + pass_length] = deviceCharset[p9 + (MAX_CHARSET_LENGTH * 9)];
        if (pass_length >= 11) OutputPassword[search_index * MAX_PASSWORD_LEN + 10] = deviceCharset[p10 + (MAX_CHARSET_LENGTH * 10)];
        if (pass_length >= 11) OutputPassword[search_index * MAX_PASSWORD_LEN + 10 + pass_length] = deviceCharset[p10 + (MAX_CHARSET_LENGTH * 10)];
        if (pass_length >= 12) OutputPassword[search_index * MAX_PASSWORD_LEN + 11] = deviceCharset[p11 + (MAX_CHARSET_LENGTH * 11)];
        if (pass_length >= 12) OutputPassword[search_index * MAX_PASSWORD_LEN + 11 + pass_length] = deviceCharset[p11 + (MAX_CHARSET_LENGTH * 11)];
        if (pass_length >= 13) OutputPassword[search_index * MAX_PASSWORD_LEN + 12] = deviceCharset[p12 + (MAX_CHARSET_LENGTH * 12)];
        if (pass_length >= 13) OutputPassword[search_index * MAX_PASSWORD_LEN + 12 + pass_length] = deviceCharset[p12 + (MAX_CHARSET_LENGTH * 12)];
        if (pass_length >= 14) OutputPassword[search_index * MAX_PASSWORD_LEN + 13] = deviceCharset[p13 + (MAX_CHARSET_LENGTH * 13)];
        if (pass_length >= 14) OutputPassword[search_index * MAX_PASSWORD_LEN + 13 + pass_length] = deviceCharset[p13 + (MAX_CHARSET_LENGTH * 13)];
        if (pass_length >= 15) OutputPassword[search_index * MAX_PASSWORD_LEN + 14] = deviceCharset[p14 + (MAX_CHARSET_LENGTH * 14)];
        if (pass_length >= 15) OutputPassword[search_index * MAX_PASSWORD_LEN + 14 + pass_length] = deviceCharset[p14 + (MAX_CHARSET_LENGTH * 14)];
        if (pass_length >= 16) OutputPassword[search_index * MAX_PASSWORD_LEN + 15] = deviceCharset[p15 + (MAX_CHARSET_LENGTH * 15)];
        if (pass_length >= 16) OutputPassword[search_index * MAX_PASSWORD_LEN + 15 + pass_length] = deviceCharset[p15 + (MAX_CHARSET_LENGTH * 15)];
        if (pass_length >= 17) OutputPassword[search_index * MAX_PASSWORD_LEN + 16] = deviceCharset[p16 + (MAX_CHARSET_LENGTH * 16)];
        if (pass_length >= 17) OutputPassword[search_index * MAX_PASSWORD_LEN + 16 + pass_length] = deviceCharset[p16 + (MAX_CHARSET_LENGTH * 16)];
        if (pass_length >= 18) OutputPassword[search_index * MAX_PASSWORD_LEN + 17] = deviceCharset[p17 + (MAX_CHARSET_LENGTH * 17)];
        if (pass_length >= 18) OutputPassword[search_index * MAX_PASSWORD_LEN + 17 + pass_length] = deviceCharset[p17 + (MAX_CHARSET_LENGTH * 17)];
        if (pass_length >= 19) OutputPassword[search_index * MAX_PASSWORD_LEN + 18] = deviceCharset[p18 + (MAX_CHARSET_LENGTH * 18)];
        if (pass_length >= 19) OutputPassword[search_index * MAX_PASSWORD_LEN + 18 + pass_length] = deviceCharset[p18 + (MAX_CHARSET_LENGTH * 18)];
        if (pass_length >= 20) OutputPassword[search_index * MAX_PASSWORD_LEN + 19] = deviceCharset[p19 + (MAX_CHARSET_LENGTH * 19)];
        if (pass_length >= 20) OutputPassword[search_index * MAX_PASSWORD_LEN + 19 + pass_length] = deviceCharset[p19 + (MAX_CHARSET_LENGTH * 19)];
        if (pass_length >= 21) OutputPassword[search_index * MAX_PASSWORD_LEN + 20] = deviceCharset[p20 + (MAX_CHARSET_LENGTH * 20)];
        if (pass_length >= 21) OutputPassword[search_index * MAX_PASSWORD_LEN + 20 + pass_length] = deviceCharset[p20 + (MAX_CHARSET_LENGTH * 20)];
        if (pass_length >= 22) OutputPassword[search_index * MAX_PASSWORD_LEN + 21] = deviceCharset[p21 + (MAX_CHARSET_LENGTH * 21)];
        if (pass_length >= 22) OutputPassword[search_index * MAX_PASSWORD_LEN + 21 + pass_length] = deviceCharset[p21 + (MAX_CHARSET_LENGTH * 21)];
        if (pass_length >= 23) OutputPassword[search_index * MAX_PASSWORD_LEN + 22] = deviceCharset[p22 + (MAX_CHARSET_LENGTH * 22)];
        if (pass_length >= 23) OutputPassword[search_index * MAX_PASSWORD_LEN + 22 + pass_length] = deviceCharset[p22 + (MAX_CHARSET_LENGTH * 22)];
        if (pass_length >= 24) OutputPassword[search_index * MAX_PASSWORD_LEN + 23] = deviceCharset[p23 + (MAX_CHARSET_LENGTH * 23)];
        if (pass_length >= 24) OutputPassword[search_index * MAX_PASSWORD_LEN + 23 + pass_length] = deviceCharset[p23 + (MAX_CHARSET_LENGTH * 23)];
        if (pass_length >= 25) OutputPassword[search_index * MAX_PASSWORD_LEN + 24] = deviceCharset[p24 + (MAX_CHARSET_LENGTH * 24)];
        if (pass_length >= 25) OutputPassword[search_index * MAX_PASSWORD_LEN + 24 + pass_length] = deviceCharset[p24 + (MAX_CHARSET_LENGTH * 24)];
        if (pass_length >= 26) OutputPassword[search_index * MAX_PASSWORD_LEN + 25] = deviceCharset[p25 + (MAX_CHARSET_LENGTH * 25)];
        if (pass_length >= 26) OutputPassword[search_index * MAX_PASSWORD_LEN + 25 + pass_length] = deviceCharset[p25 + (MAX_CHARSET_LENGTH * 25)];
        if (pass_length >= 27) OutputPassword[search_index * MAX_PASSWORD_LEN + 26] = deviceCharset[p26 + (MAX_CHARSET_LENGTH * 26)];
        if (pass_length >= 27) OutputPassword[search_index * MAX_PASSWORD_LEN + 26 + pass_length] = deviceCharset[p26 + (MAX_CHARSET_LENGTH * 26)];
        if (pass_length >= 28) OutputPassword[search_index * MAX_PASSWORD_LEN + 27] = deviceCharset[p27 + (MAX_CHARSET_LENGTH * 27)];
        if (pass_length >= 28) OutputPassword[search_index * MAX_PASSWORD_LEN + 27 + pass_length] = deviceCharset[p27 + (MAX_CHARSET_LENGTH * 27)];
        if (pass_length >= 29) OutputPassword[search_index * MAX_PASSWORD_LEN + 28] = deviceCharset[p28 + (MAX_CHARSET_LENGTH * 28)];
        if (pass_length >= 29) OutputPassword[search_index * MAX_PASSWORD_LEN + 28 + pass_length] = deviceCharset[p28 + (MAX_CHARSET_LENGTH * 28)];
        if (pass_length >= 30) OutputPassword[search_index * MAX_PASSWORD_LEN + 29] = deviceCharset[p29 + (MAX_CHARSET_LENGTH * 29)];
        if (pass_length >= 30) OutputPassword[search_index * MAX_PASSWORD_LEN + 29 + pass_length] = deviceCharset[p29 + (MAX_CHARSET_LENGTH * 29)];
        if (pass_length >= 31) OutputPassword[search_index * MAX_PASSWORD_LEN + 30] = deviceCharset[p30 + (MAX_CHARSET_LENGTH * 30)];
        if (pass_length >= 31) OutputPassword[search_index * MAX_PASSWORD_LEN + 30 + pass_length] = deviceCharset[p30 + (MAX_CHARSET_LENGTH * 30)];
        if (pass_length >= 32) OutputPassword[search_index * MAX_PASSWORD_LEN + 31] = deviceCharset[p31 + (MAX_CHARSET_LENGTH * 31)];
        if (pass_length >= 32) OutputPassword[search_index * MAX_PASSWORD_LEN + 31 + pass_length] = deviceCharset[p31 + (MAX_CHARSET_LENGTH * 31)];
        if (pass_length >= 33) OutputPassword[search_index * MAX_PASSWORD_LEN + 32] = deviceCharset[p32 + (MAX_CHARSET_LENGTH * 32)];
        if (pass_length >= 33) OutputPassword[search_index * MAX_PASSWORD_LEN + 32 + pass_length] = deviceCharset[p32 + (MAX_CHARSET_LENGTH * 32)];
        if (pass_length >= 34) OutputPassword[search_index * MAX_PASSWORD_LEN + 33] = deviceCharset[p33 + (MAX_CHARSET_LENGTH * 33)];
        if (pass_length >= 34) OutputPassword[search_index * MAX_PASSWORD_LEN + 33 + pass_length] = deviceCharset[p33 + (MAX_CHARSET_LENGTH * 33)];
        if (pass_length >= 35) OutputPassword[search_index * MAX_PASSWORD_LEN + 34] = deviceCharset[p34 + (MAX_CHARSET_LENGTH * 34)];
        if (pass_length >= 35) OutputPassword[search_index * MAX_PASSWORD_LEN + 34 + pass_length] = deviceCharset[p34 + (MAX_CHARSET_LENGTH * 34)];
        if (pass_length >= 36) OutputPassword[search_index * MAX_PASSWORD_LEN + 35] = deviceCharset[p35 + (MAX_CHARSET_LENGTH * 35)];
        if (pass_length >= 36) OutputPassword[search_index * MAX_PASSWORD_LEN + 35 + pass_length] = deviceCharset[p35 + (MAX_CHARSET_LENGTH * 35)];
        if (pass_length >= 37) OutputPassword[search_index * MAX_PASSWORD_LEN + 36] = deviceCharset[p36 + (MAX_CHARSET_LENGTH * 36)];
        if (pass_length >= 37) OutputPassword[search_index * MAX_PASSWORD_LEN + 36 + pass_length] = deviceCharset[p36 + (MAX_CHARSET_LENGTH * 36)];
        if (pass_length >= 38) OutputPassword[search_index * MAX_PASSWORD_LEN + 37] = deviceCharset[p37 + (MAX_CHARSET_LENGTH * 37)];
        if (pass_length >= 38) OutputPassword[search_index * MAX_PASSWORD_LEN + 37 + pass_length] = deviceCharset[p37 + (MAX_CHARSET_LENGTH * 37)];
        if (pass_length >= 39) OutputPassword[search_index * MAX_PASSWORD_LEN + 38] = deviceCharset[p38 + (MAX_CHARSET_LENGTH * 38)];
        if (pass_length >= 39) OutputPassword[search_index * MAX_PASSWORD_LEN + 38 + pass_length] = deviceCharset[p38 + (MAX_CHARSET_LENGTH * 38)];
        if (pass_length >= 40) OutputPassword[search_index * MAX_PASSWORD_LEN + 39] = deviceCharset[p39 + (MAX_CHARSET_LENGTH * 39)];
        if (pass_length >= 40) OutputPassword[search_index * MAX_PASSWORD_LEN + 39 + pass_length] = deviceCharset[p39 + (MAX_CHARSET_LENGTH * 39)];
        if (pass_length >= 41) OutputPassword[search_index * MAX_PASSWORD_LEN + 40] = deviceCharset[p40 + (MAX_CHARSET_LENGTH * 40)];
        if (pass_length >= 41) OutputPassword[search_index * MAX_PASSWORD_LEN + 40 + pass_length] = deviceCharset[p40 + (MAX_CHARSET_LENGTH * 40)];
        if (pass_length >= 42) OutputPassword[search_index * MAX_PASSWORD_LEN + 41] = deviceCharset[p41 + (MAX_CHARSET_LENGTH * 41)];
        if (pass_length >= 42) OutputPassword[search_index * MAX_PASSWORD_LEN + 41 + pass_length] = deviceCharset[p41 + (MAX_CHARSET_LENGTH * 41)];
        if (pass_length >= 43) OutputPassword[search_index * MAX_PASSWORD_LEN + 42] = deviceCharset[p42 + (MAX_CHARSET_LENGTH * 42)];
        if (pass_length >= 43) OutputPassword[search_index * MAX_PASSWORD_LEN + 42 + pass_length] = deviceCharset[p42 + (MAX_CHARSET_LENGTH * 42)];
        if (pass_length >= 44) OutputPassword[search_index * MAX_PASSWORD_LEN + 43] = deviceCharset[p43 + (MAX_CHARSET_LENGTH * 43)];
        if (pass_length >= 44) OutputPassword[search_index * MAX_PASSWORD_LEN + 43 + pass_length] = deviceCharset[p43 + (MAX_CHARSET_LENGTH * 43)];
        if (pass_length >= 45) OutputPassword[search_index * MAX_PASSWORD_LEN + 44] = deviceCharset[p44 + (MAX_CHARSET_LENGTH * 44)];
        if (pass_length >= 45) OutputPassword[search_index * MAX_PASSWORD_LEN + 44 + pass_length] = deviceCharset[p44 + (MAX_CHARSET_LENGTH * 44)];
        if (pass_length >= 46) OutputPassword[search_index * MAX_PASSWORD_LEN + 45] = deviceCharset[p45 + (MAX_CHARSET_LENGTH * 45)];
        if (pass_length >= 46) OutputPassword[search_index * MAX_PASSWORD_LEN + 45 + pass_length] = deviceCharset[p45 + (MAX_CHARSET_LENGTH * 45)];
        if (pass_length >= 47) OutputPassword[search_index * MAX_PASSWORD_LEN + 46] = deviceCharset[p46 + (MAX_CHARSET_LENGTH * 46)];
        if (pass_length >= 47) OutputPassword[search_index * MAX_PASSWORD_LEN + 46 + pass_length] = deviceCharset[p46 + (MAX_CHARSET_LENGTH * 46)];
        if (pass_length >= 48) OutputPassword[search_index * MAX_PASSWORD_LEN + 47] = deviceCharset[p47 + (MAX_CHARSET_LENGTH * 47)];
        if (pass_length >= 48) OutputPassword[search_index * MAX_PASSWORD_LEN + 47 + pass_length] = deviceCharset[p47 + (MAX_CHARSET_LENGTH * 47)];

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

__device__ inline void checkHashMultiSHA1(int pass_length, unsigned char *sharedBitmap, unsigned char *DEVICE_HashTable, uint32_t numberOfPasswords,
		uint32_t *DEVICE_Hashes_32, unsigned char *success, unsigned char *OutputPassword,
		unsigned char p0, unsigned char p1, unsigned char p2, unsigned char p3,
		unsigned char p4, unsigned char p5, unsigned char p6, unsigned char p7,
		unsigned char p8, unsigned char p9, unsigned char p10, unsigned char p11,
		unsigned char p12, unsigned char p13, unsigned char p14, unsigned char p15,
		UINT4 a, UINT4 b, UINT4 c, UINT4 d, UINT4 e,
		uint32_t &search_index, uint32_t &search_high, uint32_t &search_low, uint32_t &hash_order_a,
		uint32_t &hash_order_mem, uint32_t &temp) {
  //uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp;
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
    temp = DEVICE_Hashes_32[5 * search_index];
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

  while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 5])) {
    search_index--;
  }


  while ((a == DEVICE_Hashes_32[search_index * 5])) {
  if (a == DEVICE_Hashes_32[search_index * 5]) {
    if (b == DEVICE_Hashes_32[search_index * 5 + 1]) {
      if (c == DEVICE_Hashes_32[search_index * 5 + 2]) {
        if (d == DEVICE_Hashes_32[search_index * 5 + 3]) {
        if (e == DEVICE_Hashes_32[search_index * 5 + 4]) {
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


__device__ inline void checkHashMultiSHA1Long(int pass_length, unsigned char *sharedBitmap, unsigned char *DEVICE_HashTable, uint32_t numberOfPasswords,
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
  //uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp;
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
    temp = DEVICE_Hashes_32[5 * search_index];
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

  while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 5])) {
    search_index--;
  }


  while ((a == DEVICE_Hashes_32[search_index * 5])) {
  if (a == DEVICE_Hashes_32[search_index * 5]) {
    if (b == DEVICE_Hashes_32[search_index * 5 + 1]) {
      if (c == DEVICE_Hashes_32[search_index * 5 + 2]) {
        if (d == DEVICE_Hashes_32[search_index * 5 + 3]) {
        if (e == DEVICE_Hashes_32[search_index * 5 + 4]) {
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

// Resets the character at the given position.  Works with data there.
__device__ inline void ResetCharacterAtPosition(unsigned char character, unsigned char position,
        UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
	UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

    int offset = position / 4;

    if (UseB0 & offset == 0) {
        b0 &= ~(0x000000ff << (8 * (position % 4)));
        b0 |= character << (8 * (position % 4));
    } else if (UseB1 & offset == 1) {
        b1 &= ~(0x000000ff << (8 * (position % 4)));
        b1 |= character << (8 * (position % 4));
    } else if (UseB2 & offset == 2) {
        b2 &= ~(0x000000ff << (8 * (position % 4)));
        b2 |= character << (8 * (position % 4));
    } else if (UseB3 & offset == 3) {
        b3 &= ~(0x000000ff << (8 * (position % 4)));
        b3 |= character << (8 * (position % 4));
    } else if (UseB4 & offset == 4) {
        b4 &= ~(0x000000ff << (8 * (position % 4)));
        b4 |= character << (8 * (position % 4));
    } else if (UseB5 & offset == 5) {
        b5 &= ~(0x000000ff << (8 * (position % 4)));
        b5 |= character << (8 * (position % 4));
    } else if (UseB6 & offset == 6) {
        b6 &= ~(0x000000ff << (8 * (position % 4)));
        b6 |= character << (8 * (position % 4));
    } else if (UseB7 & offset == 7) {
        b7 &= ~(0x000000ff << (8 * (position % 4)));
        b7 |= character << (8 * (position % 4));
    } else if (UseB8 & offset == 8) {
        b8 &= ~(0x000000ff << (8 * (position % 4)));
        b8 |= character << (8 * (position % 4));
    } else if (UseB9 & offset == 9) {
        b9 &= ~(0x000000ff << (8 * (position % 4)));
        b9 |= character << (8 * (position % 4));
    } else if (UseB10 & offset == 10) {
        b10 &= ~(0x000000ff << (8 * (position % 4)));
        b10 |= character << (8 * (position % 4));
    } else if (UseB11 & offset == 11) {
        b11 &= ~(0x000000ff << (8 * (position % 4)));
        b11 |= character << (8 * (position % 4));
    } else if (UseB12 & offset == 12) {
        b12 &= ~(0x000000ff << (8 * (position % 4)));
        b12 |= character << (8 * (position % 4));
    } else if (UseB13 & offset == 13) {
        b13 &= ~(0x000000ff << (8 * (position % 4)));
        b13 |= character << (8 * (position % 4));
    }
}


// Sets the character at the given position.  Requires things to be zeroed first!
__device__ inline void SetCharacterAtPosition(unsigned char character, unsigned char position,
        UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
	UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

    int offset = position / 4;

    if (UseB0 & offset == 0) {
        b0 |= character << (8 * (position % 4));
    } else if (UseB1 & offset == 1) {
        b1 |= character << (8 * (position % 4));
    } else if (UseB2 & offset == 2) {
        b2 |= character << (8 * (position % 4));
    } else if (UseB3 & offset == 3) {
        b3 |= character << (8 * (position % 4));
    } else if (UseB4 & offset == 4) {
        b4 |= character << (8 * (position % 4));
    } else if (UseB5 & offset == 5) {
        b5 |= character << (8 * (position % 4));
    } else if (UseB6 & offset == 6) {
        b6 |= character << (8 * (position % 4));
    } else if (UseB7 & offset == 7) {
        b7 |= character << (8 * (position % 4));
    } else if (UseB8 & offset == 8) {
        b8 |= character << (8 * (position % 4));
    } else if (UseB9 & offset == 9) {
        b9 |= character << (8 * (position % 4));
    } else if (UseB10 & offset == 10) {
        b10 |= character << (8 * (position % 4));
    } else if (UseB11 & offset == 11) {
        b11 |= character << (8 * (position % 4));
    } else if (UseB12 & offset == 12) {
        b12 |= character << (8 * (position % 4));
    } else if (UseB13 & offset == 13) {
        b13 |= character << (8 * (position % 4));
    }
}



// Load salts non-destructively.
__device__ inline void LoadSalt(int password_length, uint32_t SaltToLoad, unsigned char *Salts, unsigned char *SaltLengths,
        UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
	UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

    int i;

    for (i = 0; i < SaltLengths[SaltToLoad]; i++) {
        SetCharacterAtPosition(Salts[SaltToLoad * MAX_SALT_LENGTH + i], (i + password_length),
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
    }

}

// initMD loads the charset values into the working blocks for MD4/MD5
// length : length in bytes to load
__device__ inline void LoadPasswordAtPosition(int length, int startPosition, unsigned char *sharedCharset,
		unsigned char &p0, unsigned char &p1, unsigned char &p2, unsigned char &p3,
		unsigned char &p4, unsigned char &p5, unsigned char &p6, unsigned char &p7,
		unsigned char &p8, unsigned char &p9, unsigned char &p10, unsigned char &p11,
		unsigned char &p12, unsigned char &p13, unsigned char &p14, unsigned char &p15,
		UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
		UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

    // Trying this out...
    switch(length) {
        case 16:
            SetCharacterAtPosition(sharedCharset[p15 + (MAX_CHARSET_LENGTH * 15)],
                    startPosition + 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 15:
            SetCharacterAtPosition(sharedCharset[p14 + (MAX_CHARSET_LENGTH * 14)],
                    startPosition + 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 14:
            SetCharacterAtPosition(sharedCharset[p13 + (MAX_CHARSET_LENGTH * 13)],
                    startPosition + 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 13:
            SetCharacterAtPosition(sharedCharset[p12 + (MAX_CHARSET_LENGTH * 12)],
                    startPosition + 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 12:
            SetCharacterAtPosition(sharedCharset[p11 + (MAX_CHARSET_LENGTH * 11)],
                    startPosition + 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 11:
            SetCharacterAtPosition(sharedCharset[p10 + (MAX_CHARSET_LENGTH * 10)],
                    startPosition + 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 10:
            SetCharacterAtPosition(sharedCharset[p9 + (MAX_CHARSET_LENGTH * 9)],
                    startPosition + 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 9:
            SetCharacterAtPosition(sharedCharset[p8 + (MAX_CHARSET_LENGTH * 8)],
                    startPosition + 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 8:
            SetCharacterAtPosition(sharedCharset[p7 + (MAX_CHARSET_LENGTH * 7)],
                    startPosition + 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 7:
            SetCharacterAtPosition(sharedCharset[p6 + (MAX_CHARSET_LENGTH * 6)],
                    startPosition + 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 6:
            SetCharacterAtPosition(sharedCharset[p5 + (MAX_CHARSET_LENGTH * 5)],
                    startPosition + 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 5:
            SetCharacterAtPosition(sharedCharset[p4 + (MAX_CHARSET_LENGTH * 4)],
                    startPosition + 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 4:
            SetCharacterAtPosition(sharedCharset[p3 + (MAX_CHARSET_LENGTH * 3)],
                    startPosition + 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 3:
            SetCharacterAtPosition(sharedCharset[p2 + (MAX_CHARSET_LENGTH * 2)],
                    startPosition + 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 2:
            SetCharacterAtPosition(sharedCharset[p1 + (MAX_CHARSET_LENGTH * 1)],
                    startPosition + 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 1:
            SetCharacterAtPosition(sharedCharset[p0 + (MAX_CHARSET_LENGTH * 0)],
                    startPosition + 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
    }
}




__device__ inline void LoadPasswordAtPositionLong(int length, int startPosition, unsigned char *sharedCharset,
		unsigned char &p0, unsigned char &p1, unsigned char &p2, unsigned char &p3,
		unsigned char &p4, unsigned char &p5, unsigned char &p6, unsigned char &p7,
		unsigned char &p8, unsigned char &p9, unsigned char &p10, unsigned char &p11,
		unsigned char &p12, unsigned char &p13, unsigned char &p14, unsigned char &p15,
                unsigned char &p16, unsigned char &p17, unsigned char &p18, unsigned char &p19,
                unsigned char &p20, unsigned char &p21, unsigned char &p22, unsigned char &p23,
                unsigned char &p24, unsigned char &p25, unsigned char &p26, unsigned char &p27,
                unsigned char &p28, unsigned char &p29, unsigned char &p30, unsigned char &p31,
                unsigned char &p32, unsigned char &p33, unsigned char &p34, unsigned char &p35,
                unsigned char &p36, unsigned char &p37, unsigned char &p38, unsigned char &p39,
                unsigned char &p40, unsigned char &p41, unsigned char &p42, unsigned char &p43,
                unsigned char &p44, unsigned char &p45, unsigned char &p46, unsigned char &p47,
		UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
		UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

    // Trying this out...
    switch(length) {
        case 48:
            SetCharacterAtPosition(sharedCharset[p47 + (MAX_CHARSET_LENGTH * 47)],
                    startPosition + 47, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 47:
            SetCharacterAtPosition(sharedCharset[p46 + (MAX_CHARSET_LENGTH * 46)],
                    startPosition + 46, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 46:
            SetCharacterAtPosition(sharedCharset[p45 + (MAX_CHARSET_LENGTH * 45)],
                    startPosition + 45, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 45:
            SetCharacterAtPosition(sharedCharset[p44 + (MAX_CHARSET_LENGTH * 44)],
                    startPosition + 44, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 44:
            SetCharacterAtPosition(sharedCharset[p43 + (MAX_CHARSET_LENGTH * 43)],
                    startPosition + 43, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 43:
            SetCharacterAtPosition(sharedCharset[p42 + (MAX_CHARSET_LENGTH * 42)],
                    startPosition + 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 42:
            SetCharacterAtPosition(sharedCharset[p41 + (MAX_CHARSET_LENGTH * 41)],
                    startPosition + 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 41:
            SetCharacterAtPosition(sharedCharset[p40 + (MAX_CHARSET_LENGTH * 40)],
                    startPosition + 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 40:
            SetCharacterAtPosition(sharedCharset[p39 + (MAX_CHARSET_LENGTH * 39)],
                    startPosition + 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 39:
            SetCharacterAtPosition(sharedCharset[p38 + (MAX_CHARSET_LENGTH * 38)],
                    startPosition + 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 38:
            SetCharacterAtPosition(sharedCharset[p37 + (MAX_CHARSET_LENGTH * 37)],
                    startPosition + 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 37:
            SetCharacterAtPosition(sharedCharset[p36 + (MAX_CHARSET_LENGTH * 36)],
                    startPosition + 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 36:
            SetCharacterAtPosition(sharedCharset[p35 + (MAX_CHARSET_LENGTH * 35)],
                    startPosition + 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 35:
            SetCharacterAtPosition(sharedCharset[p34 + (MAX_CHARSET_LENGTH * 34)],
                    startPosition + 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 34:
            SetCharacterAtPosition(sharedCharset[p33 + (MAX_CHARSET_LENGTH * 33)],
                    startPosition + 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 33:
            SetCharacterAtPosition(sharedCharset[p32 + (MAX_CHARSET_LENGTH * 32)],
                    startPosition + 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 32:
            SetCharacterAtPosition(sharedCharset[p31 + (MAX_CHARSET_LENGTH * 31)],
                    startPosition + 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 31:
            SetCharacterAtPosition(sharedCharset[p30 + (MAX_CHARSET_LENGTH * 30)],
                    startPosition + 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 30:
            SetCharacterAtPosition(sharedCharset[p29 + (MAX_CHARSET_LENGTH * 29)],
                    startPosition + 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 29:
            SetCharacterAtPosition(sharedCharset[p28 + (MAX_CHARSET_LENGTH * 28)],
                    startPosition + 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 28:
            SetCharacterAtPosition(sharedCharset[p27 + (MAX_CHARSET_LENGTH * 27)],
                    startPosition + 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 27:
            SetCharacterAtPosition(sharedCharset[p26 + (MAX_CHARSET_LENGTH * 26)],
                    startPosition + 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 26:
            SetCharacterAtPosition(sharedCharset[p25 + (MAX_CHARSET_LENGTH * 25)],
                    startPosition + 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 25:
            SetCharacterAtPosition(sharedCharset[p24 + (MAX_CHARSET_LENGTH * 24)],
                    startPosition + 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 24:
            SetCharacterAtPosition(sharedCharset[p23 + (MAX_CHARSET_LENGTH * 23)],
                    startPosition + 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 23:
            SetCharacterAtPosition(sharedCharset[p22 + (MAX_CHARSET_LENGTH * 22)],
                    startPosition + 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 22:
            SetCharacterAtPosition(sharedCharset[p21 + (MAX_CHARSET_LENGTH * 21)],
                    startPosition + 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 21:
            SetCharacterAtPosition(sharedCharset[p20 + (MAX_CHARSET_LENGTH * 20)],
                    startPosition + 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 20:
            SetCharacterAtPosition(sharedCharset[p19 + (MAX_CHARSET_LENGTH * 19)],
                    startPosition + 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 19:
            SetCharacterAtPosition(sharedCharset[p18 + (MAX_CHARSET_LENGTH * 18)],
                    startPosition + 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 18:
            SetCharacterAtPosition(sharedCharset[p17 + (MAX_CHARSET_LENGTH * 17)],
                    startPosition + 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 17:
            SetCharacterAtPosition(sharedCharset[p16 + (MAX_CHARSET_LENGTH * 16)],
                    startPosition + 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 16:
            SetCharacterAtPosition(sharedCharset[p15 + (MAX_CHARSET_LENGTH * 15)],
                    startPosition + 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 15:
            SetCharacterAtPosition(sharedCharset[p14 + (MAX_CHARSET_LENGTH * 14)],
                    startPosition + 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 14:
            SetCharacterAtPosition(sharedCharset[p13 + (MAX_CHARSET_LENGTH * 13)],
                    startPosition + 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 13:
            SetCharacterAtPosition(sharedCharset[p12 + (MAX_CHARSET_LENGTH * 12)],
                    startPosition + 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 12:
            SetCharacterAtPosition(sharedCharset[p11 + (MAX_CHARSET_LENGTH * 11)],
                    startPosition + 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 11:
            SetCharacterAtPosition(sharedCharset[p10 + (MAX_CHARSET_LENGTH * 10)],
                    startPosition + 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 10:
            SetCharacterAtPosition(sharedCharset[p9 + (MAX_CHARSET_LENGTH * 9)],
                    startPosition + 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 9:
            SetCharacterAtPosition(sharedCharset[p8 + (MAX_CHARSET_LENGTH * 8)],
                    startPosition + 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 8:
            SetCharacterAtPosition(sharedCharset[p7 + (MAX_CHARSET_LENGTH * 7)],
                    startPosition + 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 7:
            SetCharacterAtPosition(sharedCharset[p6 + (MAX_CHARSET_LENGTH * 6)],
                    startPosition + 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 6:
            SetCharacterAtPosition(sharedCharset[p5 + (MAX_CHARSET_LENGTH * 5)],
                    startPosition + 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 5:
            SetCharacterAtPosition(sharedCharset[p4 + (MAX_CHARSET_LENGTH * 4)],
                    startPosition + 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 4:
            SetCharacterAtPosition(sharedCharset[p3 + (MAX_CHARSET_LENGTH * 3)],
                    startPosition + 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 3:
            SetCharacterAtPosition(sharedCharset[p2 + (MAX_CHARSET_LENGTH * 2)],
                    startPosition + 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 2:
            SetCharacterAtPosition(sharedCharset[p1 + (MAX_CHARSET_LENGTH * 1)],
                    startPosition + 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 1:
            SetCharacterAtPosition(sharedCharset[p0 + (MAX_CHARSET_LENGTH * 0)],
                    startPosition + 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
    }
}


__device__ inline void LoadNTLMPasswordAtPositionLong(int length, int startPosition, unsigned char *sharedCharset,
		unsigned char &p0, unsigned char &p1, unsigned char &p2, unsigned char &p3,
		unsigned char &p4, unsigned char &p5, unsigned char &p6, unsigned char &p7,
		unsigned char &p8, unsigned char &p9, unsigned char &p10, unsigned char &p11,
		unsigned char &p12, unsigned char &p13, unsigned char &p14, unsigned char &p15,
                unsigned char &p16, unsigned char &p17, unsigned char &p18, unsigned char &p19,
                unsigned char &p20, unsigned char &p21, unsigned char &p22, unsigned char &p23,
                unsigned char &p24, unsigned char &p25, unsigned char &p26, unsigned char &p27,
                unsigned char &p28, unsigned char &p29, unsigned char &p30, unsigned char &p31,
                unsigned char &p32, unsigned char &p33, unsigned char &p34, unsigned char &p35,
                unsigned char &p36, unsigned char &p37, unsigned char &p38, unsigned char &p39,
                unsigned char &p40, unsigned char &p41, unsigned char &p42, unsigned char &p43,
                unsigned char &p44, unsigned char &p45, unsigned char &p46, unsigned char &p47,
		UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
		UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

    // Trying this out...
    switch(length) {
        case 28:
            SetCharacterAtPosition(sharedCharset[p27 + (MAX_CHARSET_LENGTH * 27)],
                    startPosition + 54, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 27:
            SetCharacterAtPosition(sharedCharset[p26 + (MAX_CHARSET_LENGTH * 26)],
                    startPosition + 52, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 26:
            SetCharacterAtPosition(sharedCharset[p25 + (MAX_CHARSET_LENGTH * 25)],
                    startPosition + 50, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 25:
            SetCharacterAtPosition(sharedCharset[p24 + (MAX_CHARSET_LENGTH * 24)],
                    startPosition + 48, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 24:
            SetCharacterAtPosition(sharedCharset[p23 + (MAX_CHARSET_LENGTH * 23)],
                    startPosition + 46, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 23:
            SetCharacterAtPosition(sharedCharset[p22 + (MAX_CHARSET_LENGTH * 22)],
                    startPosition + 44, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 22:
            SetCharacterAtPosition(sharedCharset[p21 + (MAX_CHARSET_LENGTH * 21)],
                    startPosition + 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 21:
            SetCharacterAtPosition(sharedCharset[p20 + (MAX_CHARSET_LENGTH * 20)],
                    startPosition + 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 20:
            SetCharacterAtPosition(sharedCharset[p19 + (MAX_CHARSET_LENGTH * 19)],
                    startPosition + 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 19:
            SetCharacterAtPosition(sharedCharset[p18 + (MAX_CHARSET_LENGTH * 18)],
                    startPosition + 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 18:
            SetCharacterAtPosition(sharedCharset[p17 + (MAX_CHARSET_LENGTH * 17)],
                    startPosition + 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 17:
            SetCharacterAtPosition(sharedCharset[p16 + (MAX_CHARSET_LENGTH * 16)],
                    startPosition + 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 16:
            SetCharacterAtPosition(sharedCharset[p15 + (MAX_CHARSET_LENGTH * 15)],
                    startPosition + 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 15:
            SetCharacterAtPosition(sharedCharset[p14 + (MAX_CHARSET_LENGTH * 14)],
                    startPosition + 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 14:
            SetCharacterAtPosition(sharedCharset[p13 + (MAX_CHARSET_LENGTH * 13)],
                    startPosition + 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 13:
            SetCharacterAtPosition(sharedCharset[p12 + (MAX_CHARSET_LENGTH * 12)],
                    startPosition + 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 12:
            SetCharacterAtPosition(sharedCharset[p11 + (MAX_CHARSET_LENGTH * 11)],
                    startPosition + 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 11:
            SetCharacterAtPosition(sharedCharset[p10 + (MAX_CHARSET_LENGTH * 10)],
                    startPosition + 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 10:
            SetCharacterAtPosition(sharedCharset[p9 + (MAX_CHARSET_LENGTH * 9)],
                    startPosition + 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 9:
            SetCharacterAtPosition(sharedCharset[p8 + (MAX_CHARSET_LENGTH * 8)],
                    startPosition + 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 8:
            SetCharacterAtPosition(sharedCharset[p7 + (MAX_CHARSET_LENGTH * 7)],
                    startPosition + 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 7:
            SetCharacterAtPosition(sharedCharset[p6 + (MAX_CHARSET_LENGTH * 6)],
                    startPosition + 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 6:
            SetCharacterAtPosition(sharedCharset[p5 + (MAX_CHARSET_LENGTH * 5)],
                    startPosition + 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 5:
            SetCharacterAtPosition(sharedCharset[p4 + (MAX_CHARSET_LENGTH * 4)],
                    startPosition + 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 4:
            SetCharacterAtPosition(sharedCharset[p3 + (MAX_CHARSET_LENGTH * 3)],
                    startPosition + 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 3:
            SetCharacterAtPosition(sharedCharset[p2 + (MAX_CHARSET_LENGTH * 2)],
                    startPosition + 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 2:
            SetCharacterAtPosition(sharedCharset[p1 + (MAX_CHARSET_LENGTH * 1)],
                    startPosition + 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        case 1:
            SetCharacterAtPosition(sharedCharset[p0 + (MAX_CHARSET_LENGTH * 0)],
                    startPosition + 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
    }
}

// Loads a hash into the given registers as a string.
__device__ inline void LoadHashAsString(char hashLookup[256][2],
        uint32_t a, uint32_t b, uint32_t c, uint32_t d,
        uint32_t &b0, uint32_t &b1, uint32_t &b2, uint32_t &b3,
        uint32_t &b4, uint32_t &b5, uint32_t &b6, uint32_t &b7) {
    // We have a fixed length here.

    // a = 0x00112233
    // b0: 0x31313030
    // a = 0x12345678
    // b0: 0x34333231

    b1 = (uint32_t)hashLookup[(a >> 16) & 0xff][0] | (uint32_t)hashLookup[(a >> 16) & 0xff][1] << 8 |
            (uint32_t)hashLookup[(a >> 24) & 0xff][0] << 16 | (uint32_t)hashLookup[(a >> 24) & 0xff][1] << 24;
    b0 = (uint32_t)hashLookup[(a >> 0) & 0xff][0] | (uint32_t)hashLookup[(a >> 0) & 0xff][1] << 8 |
            (uint32_t)hashLookup[(a >> 8) & 0xff][0] << 16 | (uint32_t)hashLookup[(a >> 8) & 0xff][1] << 24;

    b3 = (uint32_t)hashLookup[(b >> 16) & 0xff][0] | (uint32_t)hashLookup[(b >> 16) & 0xff][1] << 8 |
            (uint32_t)hashLookup[(b >> 24) & 0xff][0] << 16 | (uint32_t)hashLookup[(b >> 24) & 0xff][1] << 24;
    b2 = (uint32_t)hashLookup[(b >> 0) & 0xff][0] | (uint32_t)hashLookup[(b >> 0) & 0xff][1] << 8 |
            (uint32_t)hashLookup[(b >> 8) & 0xff][0] << 16 | (uint32_t)hashLookup[(b >> 8) & 0xff][1] << 24;

    b5 = (uint32_t)hashLookup[(c >> 16) & 0xff][0] | (uint32_t)hashLookup[(c >> 16) & 0xff][1] << 8 |
            (uint32_t)hashLookup[(c >> 24) & 0xff][0] << 16 | (uint32_t)hashLookup[(c >> 24) & 0xff][1] << 24;
    b4 = (uint32_t)hashLookup[(c >> 0) & 0xff][0] | (uint32_t)hashLookup[(c >> 0) & 0xff][1] << 8 |
            (uint32_t)hashLookup[(c >> 8) & 0xff][0] << 16 | (uint32_t)hashLookup[(c >> 8) & 0xff][1] << 24;

    b7 = (uint32_t)hashLookup[(d >> 16) & 0xff][0] | (uint32_t)hashLookup[(d >> 16) & 0xff][1] << 8 |
            (uint32_t)hashLookup[(d >> 24) & 0xff][0] << 16 | (uint32_t)hashLookup[(d >> 24) & 0xff][1] << 24;
    b6 = (uint32_t)hashLookup[(d >> 0) & 0xff][0] | (uint32_t)hashLookup[(d >> 0) & 0xff][1] << 8 |
            (uint32_t)hashLookup[(d >> 8) & 0xff][0] << 16 | (uint32_t)hashLookup[(d >> 8) & 0xff][1] << 24;

}

// Optimize the setup here...
__device__ inline void loadHashLookup(char hashLookup[256][2], char *values) {
    int i;
    if (blockDim.x >= 256) {
        if (threadIdx.x < 256) {
            hashLookup[threadIdx.x][0] = values[threadIdx.x / 16];
            hashLookup[threadIdx.x][1] = values[threadIdx.x % 16];
        }
        syncthreads();
        return;
    } else if (blockDim.x >= 128) {
        if (threadIdx.x < 128) {
            hashLookup[(2 * threadIdx.x)][0] = values[(2 * threadIdx.x) / 16];
            hashLookup[(2 * threadIdx.x)][1] = values[(2 * threadIdx.x) % 16];
            hashLookup[(2 * threadIdx.x) + 1][0] = values[((2 * threadIdx.x) + 1) / 16];
            hashLookup[(2 * threadIdx.x) + 1][1] = values[((2 * threadIdx.x) + 1) % 16];
        }
        syncthreads();
        return;
    }
    if (threadIdx.x == 0) {
        for (i = 0; i < 256; i++) {
            hashLookup[i][0] = values[i / 16];
            hashLookup[i][1] = values[i % 16];
        }
    }
    syncthreads();
}

#endif
