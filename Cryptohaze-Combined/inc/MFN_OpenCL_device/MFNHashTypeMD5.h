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

/**
 * This is an implementation of MD5 for the OpenCL platform.  It's reasonably
 * optimized for AMD, but should be fine with out BITALIGN on any platform.
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

/* F, G, H and I are basic MD5 functions.
 */

// Define F and G with bitselect.  If bfi_int is not being used, this works
// properly, and I understand the 7970 natively understands this function.
#define MD5F(x, y, z) bitselect((z), (y), (x))
#define MD5G(x, y, z) bitselect((y), (x), (z))

#define MD5H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5I(x, y, z) ((y) ^ ((x) | (~z)))



#ifdef BITALIGN
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
// Use bitalign if we are making use of ATI GPUs.
#define MD5ROTATE_LEFT(x, y) rotate((vector_type)x, (uint)y)
#define MD5FF(a, b, c, d, x, s, ac) { \
 (a) += amd_bytealign((b),(c),(d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
// HERP DERP LOOK I CAN ACCELERATE GG TOO!
#define MD5GG(a, b, c, d, x, s, ac) { \
 (a) += amd_bytealign((d),(b),(c)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#else
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
#endif


// We can't easily optimize these.
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



#define MD5_FIRST_3_ROUNDS() { \
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
MD5FF(a, b, c, d, b8, MD5S11, 0x698098d8); \
MD5FF(d, a, b, c, b9, MD5S12, 0x8b44f7af); \
MD5FF(c, d, a, b, b10, MD5S13, 0xffff5bb1); \
MD5FF(b, c, d, a, b11, MD5S14, 0x895cd7be); \
MD5FF(a, b, c, d, b12, MD5S11, 0x6b901122); \
MD5FF(d, a, b, c, b13, MD5S12, 0xfd987193); \
MD5FF(c, d, a, b, b14, MD5S13, 0xa679438e); \
MD5FF(b, c, d, a, b15, MD5S14, 0x49b40821); \
MD5GG(a, b, c, d, b1, MD5S21, 0xf61e2562); \
MD5GG(d, a, b, c, b6, MD5S22, 0xc040b340); \
MD5GG(c, d, a, b, b11, MD5S23, 0x265e5a51); \
MD5GG(b, c, d, a, b0, MD5S24, 0xe9b6c7aa); \
MD5GG(a, b, c, d, b5, MD5S21, 0xd62f105d); \
MD5GG(d, a, b, c, b10, MD5S22, 0x2441453); \
MD5GG(c, d, a, b, b15, MD5S23, 0xd8a1e681); \
MD5GG(b, c, d, a, b4, MD5S24, 0xe7d3fbc8); \
MD5GG(a, b, c, d, b9, MD5S21, 0x21e1cde6); \
MD5GG(d, a, b, c, b14, MD5S22, 0xc33707d6); \
MD5GG(c, d, a, b, b3, MD5S23, 0xf4d50d87); \
MD5GG(b, c, d, a, b8, MD5S24, 0x455a14ed); \
MD5GG(a, b, c, d, b13, MD5S21, 0xa9e3e905); \
MD5GG(d, a, b, c, b2, MD5S22, 0xfcefa3f8); \
MD5GG(c, d, a, b, b7, MD5S23, 0x676f02d9); \
MD5GG(b, c, d, a, b12, MD5S24, 0x8d2a4c8a); \
MD5HH(a, b, c, d, b5, MD5S31, 0xfffa3942); \
MD5HH(d, a, b, c, b8, MD5S32, 0x8771f681); \
MD5HH(c, d, a, b, b11, MD5S33, 0x6d9d6122); \
MD5HH(b, c, d, a, b14, MD5S34, 0xfde5380c); \
MD5HH(a, b, c, d, b1, MD5S31, 0xa4beea44); \
MD5HH(d, a, b, c, b4, MD5S32, 0x4bdecfa9); \
MD5HH(c, d, a, b, b7, MD5S33, 0xf6bb4b60); \
MD5HH(b, c, d, a, b10, MD5S34, 0xbebfbc70); \
MD5HH(a, b, c, d, b13, MD5S31, 0x289b7ec6); \
MD5HH(d, a, b, c, b0, MD5S32, 0xeaa127fa); \
MD5HH(c, d, a, b, b3, MD5S33, 0xd4ef3085); \
MD5HH(b, c, d, a, b6, MD5S34, 0x4881d05); \
MD5HH(a, b, c, d, b9, MD5S31, 0xd9d4d039); \
MD5HH(d, a, b, c, b12, MD5S32, 0xe6db99e5); \
MD5HH(c, d, a, b, b15, MD5S33, 0x1fa27cf8); \
MD5HH(b, c, d, a, b2, MD5S34, 0xc4ac5665); \
}



#define MD5_FULL_HASH() { \
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
MD5FF(a, b, c, d, b8, MD5S11, 0x698098d8); \
MD5FF(d, a, b, c, b9, MD5S12, 0x8b44f7af); \
MD5FF(c, d, a, b, b10, MD5S13, 0xffff5bb1); \
MD5FF(b, c, d, a, b11, MD5S14, 0x895cd7be); \
MD5FF(a, b, c, d, b12, MD5S11, 0x6b901122); \
MD5FF(d, a, b, c, b13, MD5S12, 0xfd987193); \
MD5FF(c, d, a, b, b14, MD5S13, 0xa679438e); \
MD5FF(b, c, d, a, b15, MD5S14, 0x49b40821); \
MD5GG(a, b, c, d, b1, MD5S21, 0xf61e2562); \
MD5GG(d, a, b, c, b6, MD5S22, 0xc040b340); \
MD5GG(c, d, a, b, b11, MD5S23, 0x265e5a51); \
MD5GG(b, c, d, a, b0, MD5S24, 0xe9b6c7aa); \
MD5GG(a, b, c, d, b5, MD5S21, 0xd62f105d); \
MD5GG(d, a, b, c, b10, MD5S22, 0x2441453); \
MD5GG(c, d, a, b, b15, MD5S23, 0xd8a1e681); \
MD5GG(b, c, d, a, b4, MD5S24, 0xe7d3fbc8); \
MD5GG(a, b, c, d, b9, MD5S21, 0x21e1cde6); \
MD5GG(d, a, b, c, b14, MD5S22, 0xc33707d6); \
MD5GG(c, d, a, b, b3, MD5S23, 0xf4d50d87); \
MD5GG(b, c, d, a, b8, MD5S24, 0x455a14ed); \
MD5GG(a, b, c, d, b13, MD5S21, 0xa9e3e905); \
MD5GG(d, a, b, c, b2, MD5S22, 0xfcefa3f8); \
MD5GG(c, d, a, b, b7, MD5S23, 0x676f02d9); \
MD5GG(b, c, d, a, b12, MD5S24, 0x8d2a4c8a); \
MD5HH(a, b, c, d, b5, MD5S31, 0xfffa3942); \
MD5HH(d, a, b, c, b8, MD5S32, 0x8771f681); \
MD5HH(c, d, a, b, b11, MD5S33, 0x6d9d6122); \
MD5HH(b, c, d, a, b14, MD5S34, 0xfde5380c); \
MD5HH(a, b, c, d, b1, MD5S31, 0xa4beea44); \
MD5HH(d, a, b, c, b4, MD5S32, 0x4bdecfa9); \
MD5HH(c, d, a, b, b7, MD5S33, 0xf6bb4b60); \
MD5HH(b, c, d, a, b10, MD5S34, 0xbebfbc70); \
MD5HH(a, b, c, d, b13, MD5S31, 0x289b7ec6); \
MD5HH(d, a, b, c, b0, MD5S32, 0xeaa127fa); \
MD5HH(c, d, a, b, b3, MD5S33, 0xd4ef3085); \
MD5HH(b, c, d, a, b6, MD5S34, 0x4881d05); \
MD5HH(a, b, c, d, b9, MD5S31, 0xd9d4d039); \
MD5HH(d, a, b, c, b12, MD5S32, 0xe6db99e5); \
MD5HH(c, d, a, b, b15, MD5S33, 0x1fa27cf8); \
MD5HH(b, c, d, a, b2, MD5S34, 0xc4ac5665); \
MD5II(a, b, c, d, b0, MD5S41, 0xf4292244); \
MD5II(d, a, b, c, b7, MD5S42, 0x432aff97); \
MD5II(c, d, a, b, b14, MD5S43, 0xab9423a7); \
MD5II(b, c, d, a, b5, MD5S44, 0xfc93a039); \
MD5II(a, b, c, d, b12, MD5S41, 0x655b59c3); \
MD5II(d, a, b, c, b3, MD5S42, 0x8f0ccc92); \
MD5II(c, d, a, b, b10, MD5S43, 0xffeff47d); \
MD5II(b, c, d, a, b1, MD5S44, 0x85845dd1); \
MD5II(a, b, c, d, b8, MD5S41, 0x6fa87e4f); \
MD5II(d, a, b, c, b15, MD5S42, 0xfe2ce6e0); \
MD5II(c, d, a, b, b6, MD5S43, 0xa3014314); \
MD5II(b, c, d, a, b13, MD5S44, 0x4e0811a1); \
MD5II(a, b, c, d, b4, MD5S41, 0xf7537e82); \
MD5II(d, a, b, c, b11, MD5S42, 0xbd3af235); \
MD5II(c, d, a, b, b2, MD5S43, 0x2ad7d2bb); \
MD5II(b, c, d, a, b9, MD5S44, 0xeb86d391); \
a += (vector_type)0x67452301; \
b += (vector_type)0xefcdab89; \
c += (vector_type)0x98badcfe; \
d += (vector_type)0x10325476; \
}