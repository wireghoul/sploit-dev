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

#include "Multiforcer_CUDA_device/CUDAReverseMD5Incrementers.h"

extern struct global_commands global_interface;


typedef uint32_t UINT4;
__device__ __constant__ char deviceCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN];
__device__ __constant__ unsigned char charsetLengths[MAX_PASSWORD_LEN];
__device__ __constant__ unsigned char constantBitmap[1]; // for lookups


#include "Multiforcer_CUDA_device/CUDAcommon.h"
#include "CUDA_Common/CUDAMD5.h"

#define MD5ROTATE_RIGHT(x, n) (((x) >> (n)) | ((x) << (32-(n))))

#define REV_II(a,b,c,d,data,shift,constant) \
    a = MD5ROTATE_RIGHT((a - b), shift) - data - constant - (c ^ (b | (~d)));


// This is a MD5 function that sets it's own bits/etc.
// Call it with all non-used b[0-15] bits zeroed, it will take care of things.
__device__ inline void CUDA_MD5_Reverse(UINT4 b0, UINT4 b1, UINT4 b2, UINT4 b3, UINT4 b4, UINT4 b5, UINT4 b6, UINT4 b7,
               UINT4 b8, UINT4 b9, UINT4 b10, UINT4 b11, UINT4 b12, UINT4 b13, UINT4 b14, UINT4 b15,
               UINT4 &a, UINT4 &b, UINT4 &c, UINT4 &d,
               int data_length_bytes) {

  // Data length and padding are already set.

  a = 0x67452301;
  b = 0xefcdab89;
  c = 0x98badcfe;
  d = 0x10325476;

  MD5FF (a, b, c, d, b0, MD5S11, 0xd76aa478); /* 1 */
  MD5FF (d, a, b, c, b1, MD5S12, 0xe8c7b756); /* 2 */
  MD5FF (c, d, a, b, b2, MD5S13, 0x242070db); /* 3 */
  MD5FF (b, c, d, a, b3, MD5S14, 0xc1bdceee); /* 4 */
  MD5FF (a, b, c, d, b4, MD5S11, 0xf57c0faf); /* 5 */
  MD5FF (d, a, b, c, b5, MD5S12, 0x4787c62a); /* 6 */
  MD5FF (c, d, a, b, b6, MD5S13, 0xa8304613); /* 7 */
  MD5FF (b, c, d, a, b7, MD5S14, 0xfd469501); /* 8 */
  MD5FF (a, b, c, d, b8, MD5S11, 0x698098d8); /* 9 */
  MD5FF (d, a, b, c, b9, MD5S12, 0x8b44f7af); /* 10 */
  MD5FF (c, d, a, b, b10, MD5S13, 0xffff5bb1); /* 11 */
  MD5FF (b, c, d, a, b11, MD5S14, 0x895cd7be); /* 12 */
  MD5FF (a, b, c, d, b12, MD5S11, 0x6b901122); /* 13 */
  MD5FF (d, a, b, c, b13, MD5S12, 0xfd987193); /* 14 */
  MD5FF (c, d, a, b, b14, MD5S13, 0xa679438e); /* 15 */
  MD5FF (b, c, d, a, b15, MD5S14, 0x49b40821); /* 16 */

 /* Round 2 */
  MD5GG (a, b, c, d, b1, MD5S21, 0xf61e2562); /* 17 */
  MD5GG (d, a, b, c, b6, MD5S22, 0xc040b340); /* 18 */
  MD5GG (c, d, a, b, b11, MD5S23, 0x265e5a51); /* 19 */
  MD5GG (b, c, d, a, b0, MD5S24, 0xe9b6c7aa); /* 20 */
  MD5GG (a, b, c, d, b5, MD5S21, 0xd62f105d); /* 21 */
  MD5GG (d, a, b, c, b10, MD5S22,  0x2441453); /* 22 */
  MD5GG (c, d, a, b, b15, MD5S23, 0xd8a1e681); /* 23 */
  MD5GG (b, c, d, a, b4, MD5S24, 0xe7d3fbc8); /* 24 */
  MD5GG (a, b, c, d, b9, MD5S21, 0x21e1cde6); /* 25 */
  MD5GG (d, a, b, c, b14, MD5S22, 0xc33707d6); /* 26 */
  MD5GG (c, d, a, b, b3, MD5S23, 0xf4d50d87); /* 27 */
  MD5GG (b, c, d, a, b8, MD5S24, 0x455a14ed); /* 28 */
  MD5GG (a, b, c, d, b13, MD5S21, 0xa9e3e905); /* 29 */
  MD5GG (d, a, b, c, b2, MD5S22, 0xfcefa3f8); /* 30 */
  MD5GG (c, d, a, b, b7, MD5S23, 0x676f02d9); /* 31 */
  MD5GG (b, c, d, a, b12, MD5S24, 0x8d2a4c8a); /* 32 */

  /* Round 3 */
  MD5HH (a, b, c, d, b5, MD5S31, 0xfffa3942); /* 33 */
  MD5HH (d, a, b, c, b8, MD5S32, 0x8771f681); /* 34 */
  MD5HH (c, d, a, b, b11, MD5S33, 0x6d9d6122); /* 35 */
  MD5HH (b, c, d, a, b14, MD5S34, 0xfde5380c); /* 36 */
  MD5HH (a, b, c, d, b1, MD5S31, 0xa4beea44); /* 37 */
  MD5HH (d, a, b, c, b4, MD5S32, 0x4bdecfa9); /* 38 */
  MD5HH (c, d, a, b, b7, MD5S33, 0xf6bb4b60); /* 39 */
  MD5HH (b, c, d, a, b10, MD5S34, 0xbebfbc70); /* 40 */
  MD5HH (a, b, c, d, b13, MD5S31, 0x289b7ec6); /* 41 */
  MD5HH (d, a, b, c, b0, MD5S32, 0xeaa127fa); /* 42 */
  MD5HH (c, d, a, b, b3, MD5S33, 0xd4ef3085); /* 43 */
  MD5HH (b, c, d, a, b6, MD5S34,  0x4881d05); /* 44 */
  MD5HH (a, b, c, d, b9, MD5S31, 0xd9d4d039); /* 45 */

  // These are done after the initial "a" check.
  // MD5HH (d, a, b, c, b12, MD5S32, 0xe6db99e5); /* 46 */
  // MD5HH (c, d, a, b, b15, MD5S33, 0x1fa27cf8); /* 47 */
  // MD5HH (b, c, d, a, b2, MD5S34, 0xc4ac5665); /* 48 */

  // Round 4 and the final constants go byebye!
}



// Reverses the MD5
__device__ void MD5_Reverse(UINT4 b0, UINT4 b1, UINT4 b2, UINT4 b3, UINT4 b4, UINT4 b5, UINT4 b6, UINT4 b7,
               UINT4 b8, UINT4 b9, UINT4 b10, UINT4 b11, UINT4 b12, UINT4 b13, UINT4 b14, UINT4 b15,
               UINT4 &a, UINT4 &b, UINT4 &c, UINT4 &d,
               int data_length_bytes, uint32_t *DEVICE_Hashes_32) {


    a = DEVICE_Hashes_32[0];
    b = DEVICE_Hashes_32[1];
    c = DEVICE_Hashes_32[2];
    d = DEVICE_Hashes_32[3];


    a -= 0x67452301;
    b -= 0xefcdab89;
    c -= 0x98badcfe;
    d -= 0x10325476;

    // Reverse the last few steps
    REV_II (b, c, d, a, b9, MD5S44, 0xeb86d391); //64
    REV_II (c, d, a, b, b2, MD5S43, 0x2ad7d2bb); //63
    REV_II (d, a, b, c, b11, MD5S42, 0xbd3af235); //62
    REV_II (a, b, c, d, b4, MD5S41, 0xf7537e82); //61
    REV_II (b, c, d, a, b13, MD5S44, 0x4e0811a1); //60
    REV_II (c, d, a, b, b6, MD5S43, 0xa3014314); //59
    REV_II (d, a, b, c, b15, MD5S42, 0xfe2ce6e0); //58
    REV_II (a, b, c, d, b8, MD5S41, 0x6fa87e4f); //57
    REV_II (b, c, d, a, b1, MD5S44, 0x85845dd1); //56
    REV_II (c, d, a, b, b10, MD5S43, 0xffeff47d); //55
    REV_II (d, a, b, c, b3, MD5S42, 0x8f0ccc92); //54
    REV_II (a, b, c, d, b12, MD5S41, 0x655b59c3); //53
    REV_II (b, c, d, a, b5, MD5S44, 0xfc93a039); //52
    REV_II (c, d, a, b, b14, MD5S43, 0xab9423a7); //51
    REV_II (d, a, b, c, b7, MD5S42, 0x432aff97); //50

    // Special case, reversing b0 here.
    REV_II (a, b, c, d, 0, MD5S41, 0xf4292244); /* 49 */
}


#define MD5SINGLE_CUDA_KERNEL_CREATE_LONG(length) \
__global__ void CUDA_MD5Single_Search_##length (unsigned char *OutputPassword, unsigned char *success, \
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, unsigned int count,  \
				unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { \
    const int pass_length = length; \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
    uint32_t a, b, c, d; \
    uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x; \
    uint32_t target_a, target_b, target_c, target_d; \
    uint32_t *DEVICE_Hashes_32 = (uint32_t *) DEVICE_Hashes; \
    unsigned char p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15; \
    UINT4 password_count = 0; \
    __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN]; \
    __shared__ unsigned char sharedLengths[MAX_PASSWORD_LEN]; \
    copyCharset(sharedCharset, sharedLengths, charsetLen, pass_length); \
    loadStartPositions(pass_length, thread_index, DEVICE_Start_Positions, \
            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15); \
    clearB0toB15(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    LoadPasswordAtPosition(pass_length, 0, sharedCharset, \
            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    b14 = pass_length * 8; \
    SetCharacterAtPosition(0x80, pass_length, \
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, \
            target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32); \
    while (password_count < count) { \
        CUDA_MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, pass_length); \
        if ((a + b0) == target_a) { \
            MD5HH(d, a, b, c, b12, MD5S32, 0xe6db99e5); \
            if (d == target_d) { \
                MD5HH(c, d, a, b, b15, MD5S33, 0x1fa27cf8); \
                if (c == target_c) { \
                    MD5HH(b, c, d, a, b2, MD5S34, 0xc4ac5665); \
                    if (b == target_b) { \
                        if (pass_length >= 1) OutputPassword[0] = deviceCharset[p0 + (MAX_CHARSET_LENGTH * 0)];\
                        if (pass_length >= 2) OutputPassword[1] = deviceCharset[p1 + (MAX_CHARSET_LENGTH * 1)];\
                        if (pass_length >= 3) OutputPassword[2] = deviceCharset[p2 + (MAX_CHARSET_LENGTH * 2)];\
                        if (pass_length >= 4) OutputPassword[3] = deviceCharset[p3 + (MAX_CHARSET_LENGTH * 3)];\
                        if (pass_length >= 5) OutputPassword[4] = deviceCharset[p4 + (MAX_CHARSET_LENGTH * 4)];\
                        if (pass_length >= 6) OutputPassword[5] = deviceCharset[p5 + (MAX_CHARSET_LENGTH * 5)];\
                        if (pass_length >= 7) OutputPassword[6] = deviceCharset[p6 + (MAX_CHARSET_LENGTH * 6)];\
                        if (pass_length >= 8) OutputPassword[7] = deviceCharset[p7 + (MAX_CHARSET_LENGTH * 7)];\
                        if (pass_length >= 9) OutputPassword[8] = deviceCharset[p8 + (MAX_CHARSET_LENGTH * 8)];\
                        if (pass_length >= 10) OutputPassword[9] = deviceCharset[p9 + (MAX_CHARSET_LENGTH * 9)];\
                        if (pass_length >= 11) OutputPassword[10] = deviceCharset[p10 + (MAX_CHARSET_LENGTH * 10)];\
                        if (pass_length >= 12) OutputPassword[11] = deviceCharset[p11 + (MAX_CHARSET_LENGTH * 11)];\
                        if (pass_length >= 13) OutputPassword[12] = deviceCharset[p12 + (MAX_CHARSET_LENGTH * 12)];\
                        if (pass_length >= 14) OutputPassword[13] = deviceCharset[p13 + (MAX_CHARSET_LENGTH * 13)];\
                        if (pass_length >= 15) OutputPassword[14] = deviceCharset[p14 + (MAX_CHARSET_LENGTH * 14)];\
                        if (pass_length >= 16) OutputPassword[15] = deviceCharset[p15 + (MAX_CHARSET_LENGTH * 15)];\
/*                        if (pass_length >= 17) OutputPassword[16] = deviceCharset[p16 + (MAX_CHARSET_LENGTH * 16)];\
                        if (pass_length >= 18) OutputPassword[17] = deviceCharset[p17 + (MAX_CHARSET_LENGTH * 17)];\
                        if (pass_length >= 19) OutputPassword[18] = deviceCharset[p18 + (MAX_CHARSET_LENGTH * 18)];\
                        if (pass_length >= 20) OutputPassword[19] = deviceCharset[p19 + (MAX_CHARSET_LENGTH * 19)];\
                        if (pass_length >= 21) OutputPassword[20] = deviceCharset[p20 + (MAX_CHARSET_LENGTH * 20)];\
                        if (pass_length >= 22) OutputPassword[21] = deviceCharset[p21 + (MAX_CHARSET_LENGTH * 21)];\
                        if (pass_length >= 23) OutputPassword[22] = deviceCharset[p22 + (MAX_CHARSET_LENGTH * 22)];\
                        if (pass_length >= 24) OutputPassword[23] = deviceCharset[p23 + (MAX_CHARSET_LENGTH * 23)];\
                        if (pass_length >= 25) OutputPassword[24] = deviceCharset[p24 + (MAX_CHARSET_LENGTH * 24)];\
                        if (pass_length >= 26) OutputPassword[25] = deviceCharset[p25 + (MAX_CHARSET_LENGTH * 25)];\
                        if (pass_length >= 27) OutputPassword[26] = deviceCharset[p26 + (MAX_CHARSET_LENGTH * 26)];\
                        if (pass_length >= 28) OutputPassword[27] = deviceCharset[p27 + (MAX_CHARSET_LENGTH * 27)];\
                        if (pass_length >= 29) OutputPassword[28] = deviceCharset[p28 + (MAX_CHARSET_LENGTH * 28)];\
                        if (pass_length >= 30) OutputPassword[29] = deviceCharset[p29 + (MAX_CHARSET_LENGTH * 29)];\
                        if (pass_length >= 31) OutputPassword[30] = deviceCharset[p30 + (MAX_CHARSET_LENGTH * 30)];\
                        if (pass_length >= 32) OutputPassword[31] = deviceCharset[p31 + (MAX_CHARSET_LENGTH * 31)];\
                        if (pass_length >= 33) OutputPassword[32] = deviceCharset[p32 + (MAX_CHARSET_LENGTH * 32)];\
                        if (pass_length >= 34) OutputPassword[33] = deviceCharset[p33 + (MAX_CHARSET_LENGTH * 33)];\
                        if (pass_length >= 35) OutputPassword[34] = deviceCharset[p34 + (MAX_CHARSET_LENGTH * 34)];\
                        if (pass_length >= 36) OutputPassword[35] = deviceCharset[p35 + (MAX_CHARSET_LENGTH * 35)];\
                        if (pass_length >= 37) OutputPassword[36] = deviceCharset[p36 + (MAX_CHARSET_LENGTH * 36)];\
                        if (pass_length >= 38) OutputPassword[37] = deviceCharset[p37 + (MAX_CHARSET_LENGTH * 37)];\
                        if (pass_length >= 39) OutputPassword[38] = deviceCharset[p38 + (MAX_CHARSET_LENGTH * 38)];\
                        if (pass_length >= 40) OutputPassword[39] = deviceCharset[p39 + (MAX_CHARSET_LENGTH * 39)];\
                        if (pass_length >= 41) OutputPassword[40] = deviceCharset[p40 + (MAX_CHARSET_LENGTH * 40)];\
                        if (pass_length >= 42) OutputPassword[41] = deviceCharset[p41 + (MAX_CHARSET_LENGTH * 41)];\
                        if (pass_length >= 43) OutputPassword[42] = deviceCharset[p42 + (MAX_CHARSET_LENGTH * 42)];\
                        if (pass_length >= 44) OutputPassword[43] = deviceCharset[p43 + (MAX_CHARSET_LENGTH * 43)];\
                        if (pass_length >= 45) OutputPassword[44] = deviceCharset[p44 + (MAX_CHARSET_LENGTH * 44)];\
                        if (pass_length >= 46) OutputPassword[45] = deviceCharset[p45 + (MAX_CHARSET_LENGTH * 45)];\
                        if (pass_length >= 47) OutputPassword[46] = deviceCharset[p46 + (MAX_CHARSET_LENGTH * 46)];\
                        if (pass_length >= 48) OutputPassword[47] = deviceCharset[p47 + (MAX_CHARSET_LENGTH * 47)];\
*/                        success[0] = 1;\
                    }\
                } \
            }\
        }\
        password_count++; \
        reverseMD5incrementCounters##length (); \
    } \
}

MD5SINGLE_CUDA_KERNEL_CREATE_LONG(5)
MD5SINGLE_CUDA_KERNEL_CREATE_LONG(6)
MD5SINGLE_CUDA_KERNEL_CREATE_LONG(7)
MD5SINGLE_CUDA_KERNEL_CREATE_LONG(8)
MD5SINGLE_CUDA_KERNEL_CREATE_LONG(9)
MD5SINGLE_CUDA_KERNEL_CREATE_LONG(10)
MD5SINGLE_CUDA_KERNEL_CREATE_LONG(11)
MD5SINGLE_CUDA_KERNEL_CREATE_LONG(12)
MD5SINGLE_CUDA_KERNEL_CREATE_LONG(13)
MD5SINGLE_CUDA_KERNEL_CREATE_LONG(14)
MD5SINGLE_CUDA_KERNEL_CREATE_LONG(15)
MD5SINGLE_CUDA_KERNEL_CREATE_LONG(16)

  


// Copy the shared variables to the host
extern "C" void copyMD5SingleDataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId) {
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceCharset, hostCharset, (MAX_CHARSET_LENGTH * charsetLength)));
    //CUDA_SAFE_CALL(cudaMemcpyToSymbol(constantBitmap, hostSharedBitmap, 8192));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(charsetLengths, hostCharsetLengths, MAX_PASSWORD_LEN));
}

extern "C" void Launch_CUDA_MD5Single_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
						unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions, uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes, unsigned char *DEVICE_Bitmap) {
    /*if (passlength == 1) {
	  CUDA_MD5_Search_1 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 2) {
	  CUDA_MD5_Search_2 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 3) {
	  CUDA_MD5_Search_3 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 4) {
	  CUDA_MD5_Search_4 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else*/if (passlength == 5) {
	  CUDA_MD5Single_Search_5 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 6) {
	  CUDA_MD5Single_Search_6 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 7) {
	  CUDA_MD5Single_Search_7 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 8) {
	  CUDA_MD5Single_Search_8 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 9) {
	  CUDA_MD5Single_Search_9 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 10) {
	  CUDA_MD5Single_Search_10 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 11) {
	  CUDA_MD5Single_Search_11 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 12) {
	  CUDA_MD5Single_Search_12 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 13) {
	  CUDA_MD5Single_Search_13 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 14) {
	  CUDA_MD5Single_Search_14 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 15) {
	  CUDA_MD5Single_Search_15 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 16) {
	  CUDA_MD5Single_Search_16 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	}
        else {
            sprintf(global_interface.exit_message, "Error: Length %d not supported!\n", passlength);
            global_interface.exit = 1;
        }

	cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
      {
        sprintf(global_interface.exit_message, "Cuda error: %s.\n", cudaGetErrorString( err) );
        global_interface.exit = 1;
        return;
      }
}