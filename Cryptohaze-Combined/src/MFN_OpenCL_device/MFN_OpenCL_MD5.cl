/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
Copyright (C) 2012  Bitweasil (http://www.cryptohaze.com/)

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

/**
 * This file contains various implementations of MD5 for use in various
 * hash functions.  There are a lot of special purpose implementations here
 * for register use reduction or other optimizations, so ensure you are using
 * the right one for your needs!
 */



// Hash defines

#define MD5S11 7
#define MD5S12 12
#define MD5S13 17
#define MD5S14 22
#define MD5S21 5
#define MD5S22 9
#define MD5S23 14
#define MD5S24 20
#define MD5S31 4
#define MD5S32 11
#define MD5S33 16
#define MD5S34 23
#define MD5S41 6
#define MD5S42 10
#define MD5S43 15
#define MD5S44 21

// Define F and G with bitselect.  This should be optimized into a bfi_int on
// all recent drivers and cards, so BITALIGN is no longer needed.
//#define MD5F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD5F(x, y, z) bitselect((z), (y), (x))
//#define MD5G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define MD5G(x, y, z) bitselect((y), (x), (z))

#define MD5H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5I(x, y, z) ((y) ^ ((x) | (~z)))

// Doing this with OpenCL common types for other GPUs.
#define MD5ROTATE_LEFT(x, y) rotate((vector_type)x, (uint)y)

#define MD5FF(a, b, c, d, x, s, ac) { \
 (a) += MD5F ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5GG(a, b, c, d, x, s, ac) { \
 (a) += MD5G ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }

// We can't easily optimize these with hardware primatives.
#define MD5HH(a, b, c, d, x, s, ac) { \
 (a) += MD5H ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5II(a, b, c, d, x, s, ac) { \
 (a) += MD5I ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }


/**
 * This is a full first block of MD5.  Useful for input lengths of 55 data bytes
 * or less.  This cannot be used for subsequent rounds, as it resets a/b/c/d at
 * the beginning of the function.  Requires the padding bit to be set, b14 to be
 * set, the works.
 */
#define MD5_FULL_HASH() { \
a = (vector_type)0x67452301; \
b = (vector_type)0xefcdab89; \
c = (vector_type)0x98badcfe; \
d = (vector_type)0x10325476; \
MD5FF(a, b, c, d, b0,  MD5S11, 0xd76aa478); \
MD5FF(d, a, b, c, b1,  MD5S12, 0xe8c7b756); \
MD5FF(c, d, a, b, b2,  MD5S13, 0x242070db); \
MD5FF(b, c, d, a, b3,  MD5S14, 0xc1bdceee); \
MD5FF(a, b, c, d, b4,  MD5S11, 0xf57c0faf); \
MD5FF(d, a, b, c, b5,  MD5S12, 0x4787c62a); \
MD5FF(c, d, a, b, b6,  MD5S13, 0xa8304613); \
MD5FF(b, c, d, a, b7,  MD5S14, 0xfd469501); \
MD5FF(a, b, c, d, b8,  MD5S11, 0x698098d8); \
MD5FF(d, a, b, c, b9,  MD5S12, 0x8b44f7af); \
MD5FF(c, d, a, b, b10, MD5S13, 0xffff5bb1); \
MD5FF(b, c, d, a, b11, MD5S14, 0x895cd7be); \
MD5FF(a, b, c, d, b12, MD5S11, 0x6b901122); \
MD5FF(d, a, b, c, b13, MD5S12, 0xfd987193); \
MD5FF(c, d, a, b, b14, MD5S13, 0xa679438e); \
MD5FF(b, c, d, a, b15, MD5S14, 0x49b40821); \
MD5GG(a, b, c, d, b1,  MD5S21, 0xf61e2562); \
MD5GG(d, a, b, c, b6,  MD5S22, 0xc040b340); \
MD5GG(c, d, a, b, b11, MD5S23, 0x265e5a51); \
MD5GG(b, c, d, a, b0,  MD5S24, 0xe9b6c7aa); \
MD5GG(a, b, c, d, b5,  MD5S21, 0xd62f105d); \
MD5GG(d, a, b, c, b10, MD5S22, 0x02441453); \
MD5GG(c, d, a, b, b15, MD5S23, 0xd8a1e681); \
MD5GG(b, c, d, a, b4,  MD5S24, 0xe7d3fbc8); \
MD5GG(a, b, c, d, b9,  MD5S21, 0x21e1cde6); \
MD5GG(d, a, b, c, b14, MD5S22, 0xc33707d6); \
MD5GG(c, d, a, b, b3,  MD5S23, 0xf4d50d87); \
MD5GG(b, c, d, a, b8,  MD5S24, 0x455a14ed); \
MD5GG(a, b, c, d, b13, MD5S21, 0xa9e3e905); \
MD5GG(d, a, b, c, b2,  MD5S22, 0xfcefa3f8); \
MD5GG(c, d, a, b, b7,  MD5S23, 0x676f02d9); \
MD5GG(b, c, d, a, b12, MD5S24, 0x8d2a4c8a); \
MD5HH(a, b, c, d, b5,  MD5S31, 0xfffa3942); \
MD5HH(d, a, b, c, b8,  MD5S32, 0x8771f681); \
MD5HH(c, d, a, b, b11, MD5S33, 0x6d9d6122); \
MD5HH(b, c, d, a, b14, MD5S34, 0xfde5380c); \
MD5HH(a, b, c, d, b1,  MD5S31, 0xa4beea44); \
MD5HH(d, a, b, c, b4,  MD5S32, 0x4bdecfa9); \
MD5HH(c, d, a, b, b7,  MD5S33, 0xf6bb4b60); \
MD5HH(b, c, d, a, b10, MD5S34, 0xbebfbc70); \
MD5HH(a, b, c, d, b13, MD5S31, 0x289b7ec6); \
MD5HH(d, a, b, c, b0,  MD5S32, 0xeaa127fa); \
MD5HH(c, d, a, b, b3,  MD5S33, 0xd4ef3085); \
MD5HH(b, c, d, a, b6,  MD5S34, 0x04881d05); \
MD5HH(a, b, c, d, b9,  MD5S31, 0xd9d4d039); \
MD5HH(d, a, b, c, b12, MD5S32, 0xe6db99e5); \
MD5HH(c, d, a, b, b15, MD5S33, 0x1fa27cf8); \
MD5HH(b, c, d, a, b2,  MD5S34, 0xc4ac5665); \
MD5II(a, b, c, d, b0,  MD5S41, 0xf4292244); \
MD5II(d, a, b, c, b7,  MD5S42, 0x432aff97); \
MD5II(c, d, a, b, b14, MD5S43, 0xab9423a7); \
MD5II(b, c, d, a, b5,  MD5S44, 0xfc93a039); \
MD5II(a, b, c, d, b12, MD5S41, 0x655b59c3); \
MD5II(d, a, b, c, b3,  MD5S42, 0x8f0ccc92); \
MD5II(c, d, a, b, b10, MD5S43, 0xffeff47d); \
MD5II(b, c, d, a, b1,  MD5S44, 0x85845dd1); \
MD5II(a, b, c, d, b8,  MD5S41, 0x6fa87e4f); \
MD5II(d, a, b, c, b15, MD5S42, 0xfe2ce6e0); \
MD5II(c, d, a, b, b6,  MD5S43, 0xa3014314); \
MD5II(b, c, d, a, b13, MD5S44, 0x4e0811a1); \
MD5II(a, b, c, d, b4,  MD5S41, 0xf7537e82); \
MD5II(d, a, b, c, b11, MD5S42, 0xbd3af235); \
MD5II(c, d, a, b, b2,  MD5S43, 0x2ad7d2bb); \
MD5II(b, c, d, a, b9,  MD5S44, 0xeb86d391); \
a += (vector_type)0x67452301; \
b += (vector_type)0xefcdab89; \
c += (vector_type)0x98badcfe; \
d += (vector_type)0x10325476; \
}


/**
 * Perform the first 3 rounds of MD5 on the initial block.  This is used in the
 * unsalted plain MD5 to allow for reversing and early-out checking.
 */
#define MD5_FIRST_3_ROUNDS() { \
a = (vector_type)0x67452301; \
b = (vector_type)0xefcdab89; \
c = (vector_type)0x98badcfe; \
d = (vector_type)0x10325476; \
MD5FF(a, b, c, d, b0,  MD5S11, 0xd76aa478); \
MD5FF(d, a, b, c, b1,  MD5S12, 0xe8c7b756); \
MD5FF(c, d, a, b, b2,  MD5S13, 0x242070db); \
MD5FF(b, c, d, a, b3,  MD5S14, 0xc1bdceee); \
MD5FF(a, b, c, d, b4,  MD5S11, 0xf57c0faf); \
MD5FF(d, a, b, c, b5,  MD5S12, 0x4787c62a); \
MD5FF(c, d, a, b, b6,  MD5S13, 0xa8304613); \
MD5FF(b, c, d, a, b7,  MD5S14, 0xfd469501); \
MD5FF(a, b, c, d, b8,  MD5S11, 0x698098d8); \
MD5FF(d, a, b, c, b9,  MD5S12, 0x8b44f7af); \
MD5FF(c, d, a, b, b10, MD5S13, 0xffff5bb1); \
MD5FF(b, c, d, a, b11, MD5S14, 0x895cd7be); \
MD5FF(a, b, c, d, b12, MD5S11, 0x6b901122); \
MD5FF(d, a, b, c, b13, MD5S12, 0xfd987193); \
MD5FF(c, d, a, b, b14, MD5S13, 0xa679438e); \
MD5FF(b, c, d, a, b15, MD5S14, 0x49b40821); \
MD5GG(a, b, c, d, b1,  MD5S21, 0xf61e2562); \
MD5GG(d, a, b, c, b6,  MD5S22, 0xc040b340); \
MD5GG(c, d, a, b, b11, MD5S23, 0x265e5a51); \
MD5GG(b, c, d, a, b0,  MD5S24, 0xe9b6c7aa); \
MD5GG(a, b, c, d, b5,  MD5S21, 0xd62f105d); \
MD5GG(d, a, b, c, b10, MD5S22, 0x02441453); \
MD5GG(c, d, a, b, b15, MD5S23, 0xd8a1e681); \
MD5GG(b, c, d, a, b4,  MD5S24, 0xe7d3fbc8); \
MD5GG(a, b, c, d, b9,  MD5S21, 0x21e1cde6); \
MD5GG(d, a, b, c, b14, MD5S22, 0xc33707d6); \
MD5GG(c, d, a, b, b3,  MD5S23, 0xf4d50d87); \
MD5GG(b, c, d, a, b8,  MD5S24, 0x455a14ed); \
MD5GG(a, b, c, d, b13, MD5S21, 0xa9e3e905); \
MD5GG(d, a, b, c, b2,  MD5S22, 0xfcefa3f8); \
MD5GG(c, d, a, b, b7,  MD5S23, 0x676f02d9); \
MD5GG(b, c, d, a, b12, MD5S24, 0x8d2a4c8a); \
MD5HH(a, b, c, d, b5,  MD5S31, 0xfffa3942); \
MD5HH(d, a, b, c, b8,  MD5S32, 0x8771f681); \
MD5HH(c, d, a, b, b11, MD5S33, 0x6d9d6122); \
MD5HH(b, c, d, a, b14, MD5S34, 0xfde5380c); \
MD5HH(a, b, c, d, b1,  MD5S31, 0xa4beea44); \
MD5HH(d, a, b, c, b4,  MD5S32, 0x4bdecfa9); \
MD5HH(c, d, a, b, b7,  MD5S33, 0xf6bb4b60); \
MD5HH(b, c, d, a, b10, MD5S34, 0xbebfbc70); \
MD5HH(a, b, c, d, b13, MD5S31, 0x289b7ec6); \
MD5HH(d, a, b, c, b0,  MD5S32, 0xeaa127fa); \
MD5HH(c, d, a, b, b3,  MD5S33, 0xd4ef3085); \
MD5HH(b, c, d, a, b6,  MD5S34, 0x04881d05); \
MD5HH(a, b, c, d, b9,  MD5S31, 0xd9d4d039); \
MD5HH(d, a, b, c, b12, MD5S32, 0xe6db99e5); \
MD5HH(c, d, a, b, b15, MD5S33, 0x1fa27cf8); \
MD5HH(b, c, d, a, b2,  MD5S34, 0xc4ac5665); \
}


/**
 * Run an MD5 round with zero data except for the length field.  This is used
 * for block length data of 56-64 - everything is zeroed except for b14 which
 * contains the length.
 */
#define MD5_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d) { \
MD5FF(a, b, c, d, 0,   MD5S11, 0xd76aa478); \
MD5FF(d, a, b, c, 0,   MD5S12, 0xe8c7b756); \
MD5FF(c, d, a, b, 0,   MD5S13, 0x242070db); \
MD5FF(b, c, d, a, 0,   MD5S14, 0xc1bdceee); \
MD5FF(a, b, c, d, 0,   MD5S11, 0xf57c0faf); \
MD5FF(d, a, b, c, 0,   MD5S12, 0x4787c62a); \
MD5FF(c, d, a, b, 0,   MD5S13, 0xa8304613); \
MD5FF(b, c, d, a, 0,   MD5S14, 0xfd469501); \
MD5FF(a, b, c, d, 0,   MD5S11, 0x698098d8); \
MD5FF(d, a, b, c, 0,   MD5S12, 0x8b44f7af); \
MD5FF(c, d, a, b, 0,   MD5S13, 0xffff5bb1); \
MD5FF(b, c, d, a, 0,   MD5S14, 0x895cd7be); \
MD5FF(a, b, c, d, 0,   MD5S11, 0x6b901122); \
MD5FF(d, a, b, c, 0,   MD5S12, 0xfd987193); \
MD5FF(c, d, a, b, b14, MD5S13, 0xa679438e); \
MD5FF(b, c, d, a, 0,   MD5S14, 0x49b40821); \
MD5GG(a, b, c, d, 0,   MD5S21, 0xf61e2562); \
MD5GG(d, a, b, c, 0,   MD5S22, 0xc040b340); \
MD5GG(c, d, a, b, 0,   MD5S23, 0x265e5a51); \
MD5GG(b, c, d, a, 0,   MD5S24, 0xe9b6c7aa); \
MD5GG(a, b, c, d, 0,   MD5S21, 0xd62f105d); \
MD5GG(d, a, b, c, 0,   MD5S22, 0x02441453); \
MD5GG(c, d, a, b, 0,   MD5S23, 0xd8a1e681); \
MD5GG(b, c, d, a, 0,   MD5S24, 0xe7d3fbc8); \
MD5GG(a, b, c, d, 0,   MD5S21, 0x21e1cde6); \
MD5GG(d, a, b, c, b14, MD5S22, 0xc33707d6); \
MD5GG(c, d, a, b, 0,   MD5S23, 0xf4d50d87); \
MD5GG(b, c, d, a, 0,   MD5S24, 0x455a14ed); \
MD5GG(a, b, c, d, 0,   MD5S21, 0xa9e3e905); \
MD5GG(d, a, b, c, 0,   MD5S22, 0xfcefa3f8); \
MD5GG(c, d, a, b, 0,   MD5S23, 0x676f02d9); \
MD5GG(b, c, d, a, 0,   MD5S24, 0x8d2a4c8a); \
MD5HH(a, b, c, d, 0,   MD5S31, 0xfffa3942); \
MD5HH(d, a, b, c, 0,   MD5S32, 0x8771f681); \
MD5HH(c, d, a, b, 0,   MD5S33, 0x6d9d6122); \
MD5HH(b, c, d, a, b14, MD5S34, 0xfde5380c); \
MD5HH(a, b, c, d, 0,   MD5S31, 0xa4beea44); \
MD5HH(d, a, b, c, 0,   MD5S32, 0x4bdecfa9); \
MD5HH(c, d, a, b, 0,   MD5S33, 0xf6bb4b60); \
MD5HH(b, c, d, a, 0,   MD5S34, 0xbebfbc70); \
MD5HH(a, b, c, d, 0,   MD5S31, 0x289b7ec6); \
MD5HH(d, a, b, c, 0,   MD5S32, 0xeaa127fa); \
MD5HH(c, d, a, b, 0,   MD5S33, 0xd4ef3085); \
MD5HH(b, c, d, a, 0,   MD5S34, 0x04881d05); \
MD5HH(a, b, c, d, 0,   MD5S31, 0xd9d4d039); \
MD5HH(d, a, b, c, 0,   MD5S32, 0xe6db99e5); \
MD5HH(c, d, a, b, 0,   MD5S33, 0x1fa27cf8); \
MD5HH(b, c, d, a, 0,   MD5S34, 0xc4ac5665); \
MD5II(a, b, c, d, 0,   MD5S41, 0xf4292244); \
MD5II(d, a, b, c, 0,   MD5S42, 0x432aff97); \
MD5II(c, d, a, b, b14, MD5S43, 0xab9423a7); \
MD5II(b, c, d, a, 0,   MD5S44, 0xfc93a039); \
MD5II(a, b, c, d, 0,   MD5S41, 0x655b59c3); \
MD5II(d, a, b, c, 0,   MD5S42, 0x8f0ccc92); \
MD5II(c, d, a, b, 0,   MD5S43, 0xffeff47d); \
MD5II(b, c, d, a, 0,   MD5S44, 0x85845dd1); \
MD5II(a, b, c, d, 0,   MD5S41, 0x6fa87e4f); \
MD5II(d, a, b, c, 0,   MD5S42, 0xfe2ce6e0); \
MD5II(c, d, a, b, 0,   MD5S43, 0xa3014314); \
MD5II(b, c, d, a, 0,   MD5S44, 0x4e0811a1); \
MD5II(a, b, c, d, 0,   MD5S41, 0xf7537e82); \
MD5II(d, a, b, c, 0,   MD5S42, 0xbd3af235); \
MD5II(c, d, a, b, 0,   MD5S43, 0x2ad7d2bb); \
MD5II(b, c, d, a, 0,   MD5S44, 0xeb86d391); \
a += (vector_type)prev_a; \
b += (vector_type)prev_b; \
c += (vector_type)prev_c; \
d += (vector_type)prev_d; \
}

/**
 * Run a second (or later) full round of MD5.  This is used for longer data
 * length processing.
 */
#define MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d) { \
MD5FF(a, b, c, d, b0,  MD5S11, 0xd76aa478); \
MD5FF(d, a, b, c, b1,  MD5S12, 0xe8c7b756); \
MD5FF(c, d, a, b, b2,  MD5S13, 0x242070db); \
MD5FF(b, c, d, a, b3,  MD5S14, 0xc1bdceee); \
MD5FF(a, b, c, d, b4,  MD5S11, 0xf57c0faf); \
MD5FF(d, a, b, c, b5,  MD5S12, 0x4787c62a); \
MD5FF(c, d, a, b, b6,  MD5S13, 0xa8304613); \
MD5FF(b, c, d, a, b7,  MD5S14, 0xfd469501); \
MD5FF(a, b, c, d, b8,  MD5S11, 0x698098d8); \
MD5FF(d, a, b, c, b9,  MD5S12, 0x8b44f7af); \
MD5FF(c, d, a, b, b10, MD5S13, 0xffff5bb1); \
MD5FF(b, c, d, a, b11, MD5S14, 0x895cd7be); \
MD5FF(a, b, c, d, b12, MD5S11, 0x6b901122); \
MD5FF(d, a, b, c, b13, MD5S12, 0xfd987193); \
MD5FF(c, d, a, b, b14, MD5S13, 0xa679438e); \
MD5FF(b, c, d, a, b15, MD5S14, 0x49b40821); \
MD5GG(a, b, c, d, b1,  MD5S21, 0xf61e2562); \
MD5GG(d, a, b, c, b6,  MD5S22, 0xc040b340); \
MD5GG(c, d, a, b, b11, MD5S23, 0x265e5a51); \
MD5GG(b, c, d, a, b0,  MD5S24, 0xe9b6c7aa); \
MD5GG(a, b, c, d, b5,  MD5S21, 0xd62f105d); \
MD5GG(d, a, b, c, b10, MD5S22, 0x02441453); \
MD5GG(c, d, a, b, b15, MD5S23, 0xd8a1e681); \
MD5GG(b, c, d, a, b4,  MD5S24, 0xe7d3fbc8); \
MD5GG(a, b, c, d, b9,  MD5S21, 0x21e1cde6); \
MD5GG(d, a, b, c, b14, MD5S22, 0xc33707d6); \
MD5GG(c, d, a, b, b3,  MD5S23, 0xf4d50d87); \
MD5GG(b, c, d, a, b8,  MD5S24, 0x455a14ed); \
MD5GG(a, b, c, d, b13, MD5S21, 0xa9e3e905); \
MD5GG(d, a, b, c, b2,  MD5S22, 0xfcefa3f8); \
MD5GG(c, d, a, b, b7,  MD5S23, 0x676f02d9); \
MD5GG(b, c, d, a, b12, MD5S24, 0x8d2a4c8a); \
MD5HH(a, b, c, d, b5,  MD5S31, 0xfffa3942); \
MD5HH(d, a, b, c, b8,  MD5S32, 0x8771f681); \
MD5HH(c, d, a, b, b11, MD5S33, 0x6d9d6122); \
MD5HH(b, c, d, a, b14, MD5S34, 0xfde5380c); \
MD5HH(a, b, c, d, b1,  MD5S31, 0xa4beea44); \
MD5HH(d, a, b, c, b4,  MD5S32, 0x4bdecfa9); \
MD5HH(c, d, a, b, b7,  MD5S33, 0xf6bb4b60); \
MD5HH(b, c, d, a, b10, MD5S34, 0xbebfbc70); \
MD5HH(a, b, c, d, b13, MD5S31, 0x289b7ec6); \
MD5HH(d, a, b, c, b0,  MD5S32, 0xeaa127fa); \
MD5HH(c, d, a, b, b3,  MD5S33, 0xd4ef3085); \
MD5HH(b, c, d, a, b6,  MD5S34, 0x04881d05); \
MD5HH(a, b, c, d, b9,  MD5S31, 0xd9d4d039); \
MD5HH(d, a, b, c, b12, MD5S32, 0xe6db99e5); \
MD5HH(c, d, a, b, b15, MD5S33, 0x1fa27cf8); \
MD5HH(b, c, d, a, b2,  MD5S34, 0xc4ac5665); \
MD5II(a, b, c, d, b0,  MD5S41, 0xf4292244); \
MD5II(d, a, b, c, b7,  MD5S42, 0x432aff97); \
MD5II(c, d, a, b, b14, MD5S43, 0xab9423a7); \
MD5II(b, c, d, a, b5,  MD5S44, 0xfc93a039); \
MD5II(a, b, c, d, b12, MD5S41, 0x655b59c3); \
MD5II(d, a, b, c, b3,  MD5S42, 0x8f0ccc92); \
MD5II(c, d, a, b, b10, MD5S43, 0xffeff47d); \
MD5II(b, c, d, a, b1,  MD5S44, 0x85845dd1); \
MD5II(a, b, c, d, b8,  MD5S41, 0x6fa87e4f); \
MD5II(d, a, b, c, b15, MD5S42, 0xfe2ce6e0); \
MD5II(c, d, a, b, b6,  MD5S43, 0xa3014314); \
MD5II(b, c, d, a, b13, MD5S44, 0x4e0811a1); \
MD5II(a, b, c, d, b4,  MD5S41, 0xf7537e82); \
MD5II(d, a, b, c, b11, MD5S42, 0xbd3af235); \
MD5II(c, d, a, b, b2,  MD5S43, 0x2ad7d2bb); \
MD5II(b, c, d, a, b9,  MD5S44, 0xeb86d391); \
a += (vector_type)prev_a; \
b += (vector_type)prev_b; \
c += (vector_type)prev_c; \
d += (vector_type)prev_d; \
}


/**
 * Run a hash for padded data of length 5.  This is used in the IPB kernel for
 * the salts, and is only usable for this.  The length is hard coded for better
 * performance and register utilization.
 */
#define MD5_FULL_HASH_LEN5() { \
a = (vector_type)0x67452301; \
b = (vector_type)0xefcdab89; \
c = (vector_type)0x98badcfe; \
d = (vector_type)0x10325476; \
MD5FF(a, b, c, d, b0, MD5S11, 0xd76aa478); \
MD5FF(d, a, b, c, b1, MD5S12, 0xe8c7b756); \
MD5FF(c, d, a, b, 0,  MD5S13, 0x242070db); \
MD5FF(b, c, d, a, 0,  MD5S14, 0xc1bdceee); \
MD5FF(a, b, c, d, 0,  MD5S11, 0xf57c0faf); \
MD5FF(d, a, b, c, 0,  MD5S12, 0x4787c62a); \
MD5FF(c, d, a, b, 0,  MD5S13, 0xa8304613); \
MD5FF(b, c, d, a, 0,  MD5S14, 0xfd469501); \
MD5FF(a, b, c, d, 0,  MD5S11, 0x698098d8); \
MD5FF(d, a, b, c, 0,  MD5S12, 0x8b44f7af); \
MD5FF(c, d, a, b, 0,  MD5S13, 0xffff5bb1); \
MD5FF(b, c, d, a, 0,  MD5S14, 0x895cd7be); \
MD5FF(a, b, c, d, 0,  MD5S11, 0x6b901122); \
MD5FF(d, a, b, c, 0,  MD5S12, 0xfd987193); \
MD5FF(c, d, a, b, 40, MD5S13, 0xa679438e); \
MD5FF(b, c, d, a, 0,  MD5S14, 0x49b40821); \
MD5GG(a, b, c, d, b1, MD5S21, 0xf61e2562); \
MD5GG(d, a, b, c, 0,  MD5S22, 0xc040b340); \
MD5GG(c, d, a, b, 0,  MD5S23, 0x265e5a51); \
MD5GG(b, c, d, a, b0, MD5S24, 0xe9b6c7aa); \
MD5GG(a, b, c, d, 0,  MD5S21, 0xd62f105d); \
MD5GG(d, a, b, c, 0,  MD5S22, 0x2441453); \
MD5GG(c, d, a, b, 0,  MD5S23, 0xd8a1e681); \
MD5GG(b, c, d, a, 0,  MD5S24, 0xe7d3fbc8); \
MD5GG(a, b, c, d, 0,  MD5S21, 0x21e1cde6); \
MD5GG(d, a, b, c, 40, MD5S22, 0xc33707d6); \
MD5GG(c, d, a, b, 0,  MD5S23, 0xf4d50d87); \
MD5GG(b, c, d, a, 0,  MD5S24, 0x455a14ed); \
MD5GG(a, b, c, d, 0,  MD5S21, 0xa9e3e905); \
MD5GG(d, a, b, c, 0,  MD5S22, 0xfcefa3f8); \
MD5GG(c, d, a, b, 0,  MD5S23, 0x676f02d9); \
MD5GG(b, c, d, a, 0,  MD5S24, 0x8d2a4c8a); \
MD5HH(a, b, c, d, 0,  MD5S31, 0xfffa3942); \
MD5HH(d, a, b, c, 0,  MD5S32, 0x8771f681); \
MD5HH(c, d, a, b, 0,  MD5S33, 0x6d9d6122); \
MD5HH(b, c, d, a, 40, MD5S34, 0xfde5380c); \
MD5HH(a, b, c, d, b1, MD5S31, 0xa4beea44); \
MD5HH(d, a, b, c, 0,  MD5S32, 0x4bdecfa9); \
MD5HH(c, d, a, b, 0,  MD5S33, 0xf6bb4b60); \
MD5HH(b, c, d, a, 0,  MD5S34, 0xbebfbc70); \
MD5HH(a, b, c, d, 0,  MD5S31, 0x289b7ec6); \
MD5HH(d, a, b, c, b0, MD5S32, 0xeaa127fa); \
MD5HH(c, d, a, b, 0,  MD5S33, 0xd4ef3085); \
MD5HH(b, c, d, a, 0,  MD5S34, 0x4881d05); \
MD5HH(a, b, c, d, 0,  MD5S31, 0xd9d4d039); \
MD5HH(d, a, b, c, 0,  MD5S32, 0xe6db99e5); \
MD5HH(c, d, a, b, 0,  MD5S33, 0x1fa27cf8); \
MD5HH(b, c, d, a, 0,  MD5S34, 0xc4ac5665); \
MD5II(a, b, c, d, b0, MD5S41, 0xf4292244); \
MD5II(d, a, b, c, 0,  MD5S42, 0x432aff97); \
MD5II(c, d, a, b, 40, MD5S43, 0xab9423a7); \
MD5II(b, c, d, a, 0,  MD5S44, 0xfc93a039); \
MD5II(a, b, c, d, 0,  MD5S41, 0x655b59c3); \
MD5II(d, a, b, c, 0,  MD5S42, 0x8f0ccc92); \
MD5II(c, d, a, b, 0,  MD5S43, 0xffeff47d); \
MD5II(b, c, d, a, b1, MD5S44, 0x85845dd1); \
MD5II(a, b, c, d, 0,  MD5S41, 0x6fa87e4f); \
MD5II(d, a, b, c, 0,  MD5S42, 0xfe2ce6e0); \
MD5II(c, d, a, b, 0,  MD5S43, 0xa3014314); \
MD5II(b, c, d, a, 0,  MD5S44, 0x4e0811a1); \
MD5II(a, b, c, d, 0,  MD5S41, 0xf7537e82); \
MD5II(d, a, b, c, 0,  MD5S42, 0xbd3af235); \
MD5II(c, d, a, b, 0,  MD5S43, 0x2ad7d2bb); \
MD5II(b, c, d, a, 0,  MD5S44, 0xeb86d391); \
a += (vector_type)0x67452301; \
b += (vector_type)0xefcdab89; \
c += (vector_type)0x98badcfe; \
d += (vector_type)0x10325476; \
}



// Hash for 32 ascii characters
#define MD5_FULL_HASH_32_ASCII() { \
a = (vector_type)0x67452301; \
b = (vector_type)0xefcdab89; \
c = (vector_type)0x98badcfe; \
d = (vector_type)0x10325476; \
MD5FF(a, b, c, d, b0, MD5S11, 0xd76aa478); \
MD5FF(d, a, b, c, b1, MD5S12, 0xe8c7b756); \
MD5FF(c, d, a, b, b2, MD5S13, 0x242070db); \
MD5FF(b, c, d, a, b3, MD5S14, 0xc1bdceee); \
MD5FF(a, b, c, d, b4, MD5S11, 0xf57c0faf); \
MD5FF(d, a, b, c, b5, MD5S12, 0x4787c62a); \
MD5FF(c, d, a, b, b6, MD5S13, 0xa8304613); \
MD5FF(b, c, d, a, b7, MD5S14, 0xfd469501); \
MD5FF(a, b, c, d, 0x00000080, MD5S11, 0x698098d8); \
MD5FF(d, a, b, c, 0, MD5S12, 0x8b44f7af); \
MD5FF(c, d, a, b, 0, MD5S13, 0xffff5bb1); \
MD5FF(b, c, d, a, 0, MD5S14, 0x895cd7be); \
MD5FF(a, b, c, d, 0, MD5S11, 0x6b901122); \
MD5FF(d, a, b, c, 0, MD5S12, 0xfd987193); \
MD5FF(c, d, a, b, (32 * 8), MD5S13, 0xa679438e); \
MD5FF(b, c, d, a, 0, MD5S14, 0x49b40821); \
MD5GG(a, b, c, d, b1, MD5S21, 0xf61e2562); \
MD5GG(d, a, b, c, b6, MD5S22, 0xc040b340); \
MD5GG(c, d, a, b, 0, MD5S23, 0x265e5a51); \
MD5GG(b, c, d, a, b0, MD5S24, 0xe9b6c7aa); \
MD5GG(a, b, c, d, b5, MD5S21, 0xd62f105d); \
MD5GG(d, a, b, c, 0, MD5S22, 0x2441453); \
MD5GG(c, d, a, b, 0, MD5S23, 0xd8a1e681); \
MD5GG(b, c, d, a, b4, MD5S24, 0xe7d3fbc8); \
MD5GG(a, b, c, d, 0, MD5S21, 0x21e1cde6); \
MD5GG(d, a, b, c, (32 * 8), MD5S22, 0xc33707d6); \
MD5GG(c, d, a, b, b3, MD5S23, 0xf4d50d87); \
MD5GG(b, c, d, a, 0x00000080, MD5S24, 0x455a14ed); \
MD5GG(a, b, c, d, 0, MD5S21, 0xa9e3e905); \
MD5GG(d, a, b, c, b2, MD5S22, 0xfcefa3f8); \
MD5GG(c, d, a, b, b7, MD5S23, 0x676f02d9); \
MD5GG(b, c, d, a, 0, MD5S24, 0x8d2a4c8a); \
MD5HH(a, b, c, d, b5, MD5S31, 0xfffa3942); \
MD5HH(d, a, b, c, 0x00000080, MD5S32, 0x8771f681); \
MD5HH(c, d, a, b, 0, MD5S33, 0x6d9d6122); \
MD5HH(b, c, d, a, (32 * 8), MD5S34, 0xfde5380c); \
MD5HH(a, b, c, d, b1, MD5S31, 0xa4beea44); \
MD5HH(d, a, b, c, b4, MD5S32, 0x4bdecfa9); \
MD5HH(c, d, a, b, b7, MD5S33, 0xf6bb4b60); \
MD5HH(b, c, d, a, 0, MD5S34, 0xbebfbc70); \
MD5HH(a, b, c, d, 0, MD5S31, 0x289b7ec6); \
MD5HH(d, a, b, c, b0, MD5S32, 0xeaa127fa); \
MD5HH(c, d, a, b, b3, MD5S33, 0xd4ef3085); \
MD5HH(b, c, d, a, b6, MD5S34, 0x4881d05); \
MD5HH(a, b, c, d, 0, MD5S31, 0xd9d4d039); \
MD5HH(d, a, b, c, 0, MD5S32, 0xe6db99e5); \
MD5HH(c, d, a, b, 0, MD5S33, 0x1fa27cf8); \
MD5HH(b, c, d, a, b2, MD5S34, 0xc4ac5665); \
MD5II(a, b, c, d, b0, MD5S41, 0xf4292244); \
MD5II(d, a, b, c, b7, MD5S42, 0x432aff97); \
MD5II(c, d, a, b, (32 * 8), MD5S43, 0xab9423a7); \
MD5II(b, c, d, a, b5, MD5S44, 0xfc93a039); \
MD5II(a, b, c, d, 0, MD5S41, 0x655b59c3); \
MD5II(d, a, b, c, b3, MD5S42, 0x8f0ccc92); \
MD5II(c, d, a, b, 0, MD5S43, 0xffeff47d); \
MD5II(b, c, d, a, b1, MD5S44, 0x85845dd1); \
MD5II(a, b, c, d, 0x00000080, MD5S41, 0x6fa87e4f); \
MD5II(d, a, b, c, 0, MD5S42, 0xfe2ce6e0); \
MD5II(c, d, a, b, b6, MD5S43, 0xa3014314); \
MD5II(b, c, d, a, 0, MD5S44, 0x4e0811a1); \
MD5II(a, b, c, d, b4, MD5S41, 0xf7537e82); \
MD5II(d, a, b, c, 0, MD5S42, 0xbd3af235); \
MD5II(c, d, a, b, b2, MD5S43, 0x2ad7d2bb); \
MD5II(b, c, d, a, 0, MD5S44, 0xeb86d391); \
a += (vector_type)0x67452301; \
b += (vector_type)0xefcdab89; \
c += (vector_type)0x98badcfe; \
d += (vector_type)0x10325476; \
}


// Second round for length 64 data.
#define MD5_SECOND_ROUND_LEN_64(prev_a, prev_b, prev_c, prev_d) { \
MD5FF(a, b, c, d, 0x80,  MD5S11, 0xd76aa478); \
MD5FF(d, a, b, c, 0,   MD5S12, 0xe8c7b756); \
MD5FF(c, d, a, b, 0,   MD5S13, 0x242070db); \
MD5FF(b, c, d, a, 0,   MD5S14, 0xc1bdceee); \
MD5FF(a, b, c, d, 0,   MD5S11, 0xf57c0faf); \
MD5FF(d, a, b, c, 0,   MD5S12, 0x4787c62a); \
MD5FF(c, d, a, b, 0,   MD5S13, 0xa8304613); \
MD5FF(b, c, d, a, 0,   MD5S14, 0xfd469501); \
MD5FF(a, b, c, d, 0,   MD5S11, 0x698098d8); \
MD5FF(d, a, b, c, 0,   MD5S12, 0x8b44f7af); \
MD5FF(c, d, a, b, 0, MD5S13, 0xffff5bb1); \
MD5FF(b, c, d, a, 0, MD5S14, 0x895cd7be); \
MD5FF(a, b, c, d, 0, MD5S11, 0x6b901122); \
MD5FF(d, a, b, c, 0, MD5S12, 0xfd987193); \
MD5FF(c, d, a, b, (64*8), MD5S13, 0xa679438e); \
MD5FF(b, c, d, a, 0, MD5S14, 0x49b40821); \
MD5GG(a, b, c, d, 0,   MD5S21, 0xf61e2562); \
MD5GG(d, a, b, c, 0,   MD5S22, 0xc040b340); \
MD5GG(c, d, a, b, 0, MD5S23, 0x265e5a51); \
MD5GG(b, c, d, a, 0x80,  MD5S24, 0xe9b6c7aa); \
MD5GG(a, b, c, d, 0,   MD5S21, 0xd62f105d); \
MD5GG(d, a, b, c, 0, MD5S22, 0x02441453); \
MD5GG(c, d, a, b, 0, MD5S23, 0xd8a1e681); \
MD5GG(b, c, d, a, 0,   MD5S24, 0xe7d3fbc8); \
MD5GG(a, b, c, d, 0,   MD5S21, 0x21e1cde6); \
MD5GG(d, a, b, c, (64*8), MD5S22, 0xc33707d6); \
MD5GG(c, d, a, b, 0,   MD5S23, 0xf4d50d87); \
MD5GG(b, c, d, a, 0,   MD5S24, 0x455a14ed); \
MD5GG(a, b, c, d, 0, MD5S21, 0xa9e3e905); \
MD5GG(d, a, b, c, 0,   MD5S22, 0xfcefa3f8); \
MD5GG(c, d, a, b, 0,   MD5S23, 0x676f02d9); \
MD5GG(b, c, d, a, 0, MD5S24, 0x8d2a4c8a); \
MD5HH(a, b, c, d, 0,   MD5S31, 0xfffa3942); \
MD5HH(d, a, b, c, 0,   MD5S32, 0x8771f681); \
MD5HH(c, d, a, b, 0, MD5S33, 0x6d9d6122); \
MD5HH(b, c, d, a, (64*8), MD5S34, 0xfde5380c); \
MD5HH(a, b, c, d, 0,   MD5S31, 0xa4beea44); \
MD5HH(d, a, b, c, 0,   MD5S32, 0x4bdecfa9); \
MD5HH(c, d, a, b, 0,   MD5S33, 0xf6bb4b60); \
MD5HH(b, c, d, a, 0, MD5S34, 0xbebfbc70); \
MD5HH(a, b, c, d, 0, MD5S31, 0x289b7ec6); \
MD5HH(d, a, b, c, 0x80,  MD5S32, 0xeaa127fa); \
MD5HH(c, d, a, b, 0,   MD5S33, 0xd4ef3085); \
MD5HH(b, c, d, a, 0,   MD5S34, 0x04881d05); \
MD5HH(a, b, c, d, 0,   MD5S31, 0xd9d4d039); \
MD5HH(d, a, b, c, 0, MD5S32, 0xe6db99e5); \
MD5HH(c, d, a, b, 0, MD5S33, 0x1fa27cf8); \
MD5HH(b, c, d, a, 0,   MD5S34, 0xc4ac5665); \
MD5II(a, b, c, d, 0x80,  MD5S41, 0xf4292244); \
MD5II(d, a, b, c, 0,   MD5S42, 0x432aff97); \
MD5II(c, d, a, b, (64*8), MD5S43, 0xab9423a7); \
MD5II(b, c, d, a, 0,   MD5S44, 0xfc93a039); \
MD5II(a, b, c, d, 0, MD5S41, 0x655b59c3); \
MD5II(d, a, b, c, 0,   MD5S42, 0x8f0ccc92); \
MD5II(c, d, a, b, 0, MD5S43, 0xffeff47d); \
MD5II(b, c, d, a, 0,   MD5S44, 0x85845dd1); \
MD5II(a, b, c, d, 0,   MD5S41, 0x6fa87e4f); \
MD5II(d, a, b, c, 0, MD5S42, 0xfe2ce6e0); \
MD5II(c, d, a, b, 0,   MD5S43, 0xa3014314); \
MD5II(b, c, d, a, 0, MD5S44, 0x4e0811a1); \
MD5II(a, b, c, d, 0,   MD5S41, 0xf7537e82); \
MD5II(d, a, b, c, 0, MD5S42, 0xbd3af235); \
MD5II(c, d, a, b, 0,   MD5S43, 0x2ad7d2bb); \
MD5II(b, c, d, a, 0,   MD5S44, 0xeb86d391); \
a += (vector_type)prev_a; \
b += (vector_type)prev_b; \
c += (vector_type)prev_c; \
d += (vector_type)prev_d; \
}
