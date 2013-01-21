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
 * Some of code is used with permission from Neinbrucke out of his EmDebr tools.
 *
 * Permission to use his source as GPLv2 was obtained from him directly.
 *
 * The interlaced MD5 operations are his, and are integrated into the Cryptohaze
 * toolchain by Bitweasil.
 */

#include <emmintrin.h>

typedef unsigned int UINT4;
#define HASHLEN 16


#ifdef WIN32
//these seem to be defined already in linux
inline __m128i operator +(const __m128i &a, const __m128i &b){
	return _mm_add_epi32(a,b);
}
inline __m128i operator -(const __m128i &a, const __m128i &b){
	return _mm_sub_epi32(a,b);
}
inline __m128i operator ^(const __m128i &a, const __m128i &b){
	return _mm_xor_si128(a,b);
}
inline __m128i operator |(const __m128i &a, const __m128i &b){
	return _mm_or_si128(a,b);
}
inline __m128i operator &(const __m128i &a, const __m128i &b){
	return _mm_and_si128(a,b);
}
#endif

// Define constants
#define AC1				 0xd76aa478
#define AC2				 0xe8c7b756
#define AC3				 0x242070db
#define AC4				 0xc1bdceee
#define AC5				 0xf57c0faf
#define AC6				 0x4787c62a
#define AC7				 0xa8304613
#define AC8				 0xfd469501
#define AC9				 0x698098d8
#define AC10			 0x8b44f7af
#define AC11			 0xffff5bb1
#define AC12			 0x895cd7be
#define AC13			 0x6b901122
#define AC14			 0xfd987193
#define AC15			 0xa679438e
#define AC16			 0x49b40821
#define AC17			 0xf61e2562
#define AC18			 0xc040b340
#define AC19			 0x265e5a51
#define AC20			 0xe9b6c7aa
#define AC21			 0xd62f105d
#define AC22			 0x02441453
#define AC23			 0xd8a1e681
#define AC24			 0xe7d3fbc8
#define AC25			 0x21e1cde6
#define AC26			 0xc33707d6
#define AC27			 0xf4d50d87
#define AC28			 0x455a14ed
#define AC29			 0xa9e3e905
#define AC30			 0xfcefa3f8
#define AC31			 0x676f02d9
#define AC32			 0x8d2a4c8a
#define AC33			 0xfffa3942
#define AC34			 0x8771f681
#define AC35			 0x6d9d6122
#define AC36			 0xfde5380c
#define AC37			 0xa4beea44
#define AC38			 0x4bdecfa9
#define AC39			 0xf6bb4b60
#define AC40			 0xbebfbc70
#define AC41			 0x289b7ec6
#define AC42			 0xeaa127fa
#define AC43			 0xd4ef3085
#define AC44			 0x04881d05
#define AC45			 0xd9d4d039
#define AC46			 0xe6db99e5
#define AC47			 0x1fa27cf8
#define AC48			 0xc4ac5665
#define AC49			 0xf4292244
#define AC50			 0x432aff97
#define AC51			 0xab9423a7
#define AC52			 0xfc93a039
#define AC53			 0x655b59c3
#define AC54			 0x8f0ccc92
#define AC55			 0xffeff47d
#define AC56			 0x85845dd1
#define AC57			 0x6fa87e4f
#define AC58			 0xfe2ce6e0
#define AC59			 0xa3014314
#define AC60			 0x4e0811a1
#define AC61			 0xf7537e82
#define AC62			 0xbd3af235
#define AC63			 0x2ad7d2bb
#define AC64			 0xeb86d391

// Define rotations
#define S11				 7
#define S12				12
#define S13				17
#define S14				22
#define S21				 5
#define S22				 9
#define S23				14
#define S24				20
#define S31				 4
#define S32				11
#define S33				16
#define S34				23
#define S41				 6
#define S42				10
#define S43				15
#define S44				21

// Define initial values
#define Ca 				0x67452301
#define Cb 				0xefcdab89
#define Cc 				0x98badcfe
#define Cd 				0x10325476

// Define functions for use in the 4 rounds of MD5
#define F(x, y, z)			((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z)			((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z)			((x) ^ (y) ^ (z))
#define I(x, y, z)			((y) ^ ((x) | ~(z)))

#define ROTATE_LEFT(a,n) _mm_or_si128(_mm_slli_epi32(a, n), _mm_srli_epi32(a, (32-n)))
#define ROTATE_RIGHT(a,n) _mm_or_si128(_mm_srli_epi32(a, n), _mm_slli_epi32(a, (32-n)))
// Rotate right is only used in reversing code, which isn't SSE2 code
#define ROTATE_RIGHT_SMALL(x, n)	(((x) >> (n)) | ((x) << (32-(n))))

// Define steps per round, 3xSSE2 interlaced, 1 instruction per line
#define MD5STEP_ROUND1(f, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, AC, x, x2, x3, s) { \
	tmp1 = (c) ^ (d);\
	tmp1_2 = (c2) ^ (d2);\
	tmp1_3 = (c3) ^ (d3);\
	tmp1 = tmp1 & (b);\
	tmp1_2 = tmp1_2 & (b2);\
	tmp1_3 = tmp1_3 & (b3);\
	tmp1 = tmp1 ^ (d);\
	tmp1_2 = tmp1_2 ^ (d2);\
	tmp1_3 = tmp1_3 ^ (d3);\
	(a) = _mm_add_epi32(a,tmp1);         \
	(a2) = _mm_add_epi32(a2,tmp1_2);         \
	(a3) = _mm_add_epi32(a3,tmp1_3);         \
	(a) = _mm_add_epi32(a,AC);  \
	(a2) = _mm_add_epi32(a2,AC);  \
	(a3) = _mm_add_epi32(a3,AC);  \
	(a) = _mm_add_epi32(a,x);  \
	(a2) = _mm_add_epi32(a2,x2);  \
	(a3) = _mm_add_epi32(a3,x3);  \
	tmp1 = _mm_slli_epi32((a), (s));\
	tmp1_2 = _mm_slli_epi32((a2), (s));\
	tmp1_3 = _mm_slli_epi32((a3), (s));\
	(a) = _mm_srli_epi32((a), (32-s));\
	(a2) = _mm_srli_epi32((a2), (32-s));\
	(a3) = _mm_srli_epi32((a3), (32-s));\
	(a) = tmp1 | (a);\
	(a2) = tmp1_2 | (a2);\
	(a3) = tmp1_3 | (a3);\
	(a) = _mm_add_epi32(a,b);                       \
	(a2) = _mm_add_epi32(a2,b2);                       \
	(a3) = _mm_add_epi32(a3,b3);                       \
}

#define MD5STEP_ROUND1_NULL(f, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, AC, s) { \
	tmp1 = (c) ^ (d);\
	tmp1_2 = (c2) ^ (d2);\
	tmp1_3 = (c3) ^ (d3);\
	tmp1 = tmp1 & (b);\
	tmp1_2 = tmp1_2 & (b2);\
	tmp1_3 = tmp1_3 & (b3);\
	tmp1 = tmp1 ^ (d);\
	tmp1_2 = tmp1_2 ^ (d2);\
	tmp1_3 = tmp1_3 ^ (d3);\
        (a) = _mm_add_epi32(a,tmp1);         \
	(a2) = _mm_add_epi32(a2,tmp1_2);         \
	(a3) = _mm_add_epi32(a3,tmp1_3);         \
	(a) = _mm_add_epi32(a,AC);  \
	(a2) = _mm_add_epi32(a2,AC);  \
	(a3) = _mm_add_epi32(a3,AC);  \
	tmp1 = _mm_slli_epi32((a), (s));\
	tmp1_2 = _mm_slli_epi32((a2), (s));\
	tmp1_3 = _mm_slli_epi32((a3), (s));\
	(a) = _mm_srli_epi32((a), (32-s));\
	(a2) = _mm_srli_epi32((a2), (32-s));\
	(a3) = _mm_srli_epi32((a3), (32-s));\
	(a) = tmp1 | (a);\
	(a2) = tmp1_2 | (a2);\
	(a3) = tmp1_3 | (a3);\
	(a) = _mm_add_epi32(a,b);                       \
	(a2) = _mm_add_epi32(a2,b2);                       \
	(a3) = _mm_add_epi32(a3,b3);                       \
}

#define MD5STEP_ROUND2(f, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, AC, x, x2, x3, s) { \
	tmp1 = (b) ^ (c);\
	tmp1_2 = (b2) ^ (c2);\
	tmp1_3 = (b3) ^ (c3);\
	tmp1 = tmp1 & (d);\
	tmp1_2 = tmp1_2 & (d2);\
	tmp1_3 = tmp1_3 & (d3);\
	tmp1 = tmp1 ^ (c);\
	tmp1_2 = tmp1_2 ^ (c2);\
	tmp1_3 = tmp1_3 ^ (c3);\
        (a) = _mm_add_epi32(a,tmp1);         \
	(a2) = _mm_add_epi32(a2,tmp1_2);         \
	(a3) = _mm_add_epi32(a3,tmp1_3);         \
	(a) = _mm_add_epi32(a,AC);  \
	(a2) = _mm_add_epi32(a2,AC);  \
	(a3) = _mm_add_epi32(a3,AC);  \
	(a) = _mm_add_epi32(a,x);  \
	(a2) = _mm_add_epi32(a2,x2);  \
	(a3) = _mm_add_epi32(a3,x3);  \
	tmp1 = _mm_slli_epi32((a), (s));\
	tmp1_2 = _mm_slli_epi32((a2), (s));\
	tmp1_3 = _mm_slli_epi32((a3), (s));\
	(a) = _mm_srli_epi32((a), (32-s));\
	(a2) = _mm_srli_epi32((a2), (32-s));\
	(a3) = _mm_srli_epi32((a3), (32-s));\
	(a) = tmp1 | (a);\
	(a2) = tmp1_2 | (a2);\
	(a3) = tmp1_3 | (a3);\
	(a) = _mm_add_epi32(a,b);                       \
	(a2) = _mm_add_epi32(a2,b2);                       \
	(a3) = _mm_add_epi32(a3,b3);                       \
}

#define MD5STEP_ROUND2_NULL(f, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, AC, s) { \
	tmp1 = (b) ^ (c);\
	tmp1_2 = (b2) ^ (c2);\
	tmp1_3 = (b3) ^ (c3);\
	tmp1 = tmp1 & (d);\
	tmp1_2 = tmp1_2 & (d2);\
	tmp1_3 = tmp1_3 & (d3);\
	tmp1 = tmp1 ^ (c);\
	tmp1_2 = tmp1_2 ^ (c2);\
	tmp1_3 = tmp1_3 ^ (c3);\
        (a) = _mm_add_epi32(a,tmp1);         \
	(a2) = _mm_add_epi32(a2,tmp1_2);         \
	(a3) = _mm_add_epi32(a3,tmp1_3);         \
	(a) = _mm_add_epi32(a,AC);  \
	(a2) = _mm_add_epi32(a2,AC);  \
	(a3) = _mm_add_epi32(a3,AC);  \
	tmp1 = _mm_slli_epi32((a), (s));\
	tmp1_2 = _mm_slli_epi32((a2), (s));\
	tmp1_3 = _mm_slli_epi32((a3), (s));\
	(a) = _mm_srli_epi32((a), (32-s));\
	(a2) = _mm_srli_epi32((a2), (32-s));\
	(a3) = _mm_srli_epi32((a3), (32-s));\
	(a) = tmp1 | (a);\
	(a2) = tmp1_2 | (a2);\
	(a3) = tmp1_3 | (a3);\
	(a) = _mm_add_epi32(a,b);                       \
	(a2) = _mm_add_epi32(a2,b2);                       \
	(a3) = _mm_add_epi32(a3,b3);                       \
}

#define MD5STEP_ROUND3(f, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, AC, x, x2, x3, s) { \
	tmp1 = (b) ^ (c);\
	tmp1_2 = (b2) ^ (c2);\
	tmp1_3 = (b3) ^ (c3);\
	tmp1 = tmp1 ^ (d);\
	tmp1_2 = tmp1_2 ^ (d2);\
	tmp1_3 = tmp1_3 ^ (d3);\
	(a) = _mm_add_epi32(a,tmp1);         \
	(a2) = _mm_add_epi32(a2,tmp1_2);         \
	(a3) = _mm_add_epi32(a3,tmp1_3);         \
	(a) = _mm_add_epi32(a,AC);  \
	(a2) = _mm_add_epi32(a2,AC);  \
	(a3) = _mm_add_epi32(a3,AC);  \
	(a) = _mm_add_epi32(a,x);  \
	(a2) = _mm_add_epi32(a2,x2);  \
	(a3) = _mm_add_epi32(a3,x3);  \
	tmp1 = _mm_slli_epi32((a), (s));\
	tmp1_2 = _mm_slli_epi32((a2), (s));\
	tmp1_3 = _mm_slli_epi32((a3), (s));\
	(a) = _mm_srli_epi32((a), (32-s));\
	(a2) = _mm_srli_epi32((a2), (32-s));\
	(a3) = _mm_srli_epi32((a3), (32-s));\
	(a) = tmp1 | (a);\
	(a2) = tmp1_2 | (a2);\
	(a3) = tmp1_3 | (a3);\
	(a) = _mm_add_epi32(a,b);                       \
	(a2) = _mm_add_epi32(a2,b2);                       \
	(a3) = _mm_add_epi32(a3,b3);                       \
}

#define MD5STEP_ROUND3_NULL(f, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, AC, s) { \
	tmp1 = (b) ^ (c);\
	tmp1_2 = (b2) ^ (c2);\
	tmp1_3 = (b3) ^ (c3);\
	tmp1 = tmp1 ^ (d);\
	tmp1_2 = tmp1_2 ^ (d2);\
	tmp1_3 = tmp1_3 ^ (d3);\
	(a) = _mm_add_epi32(a,tmp1);         \
	(a2) = _mm_add_epi32(a2,tmp1_2);         \
	(a3) = _mm_add_epi32(a3,tmp1_3);         \
	(a) = _mm_add_epi32(a,AC);  \
	(a2) = _mm_add_epi32(a2,AC);  \
	(a3) = _mm_add_epi32(a3,AC);  \
	tmp1 = _mm_slli_epi32((a), (s));\
	tmp1_2 = _mm_slli_epi32((a2), (s));\
	tmp1_3 = _mm_slli_epi32((a3), (s));\
	(a) = _mm_srli_epi32((a), (32-s));\
	(a2) = _mm_srli_epi32((a2), (32-s));\
	(a3) = _mm_srli_epi32((a3), (32-s));\
	(a) = tmp1 | (a);\
	(a2) = tmp1_2 | (a2);\
	(a3) = tmp1_3 | (a3);\
	(a) = _mm_add_epi32(a,b);                       \
	(a2) = _mm_add_epi32(a2,b2);                       \
	(a3) = _mm_add_epi32(a3,b3);                       \
}

#define MD5STEP_ROUND4(f, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, AC, x, x2, x3, s) { \
	tmp1 = _mm_andnot_si128(d, mOne);\
	tmp1_2 = _mm_andnot_si128(d2, mOne);\
	tmp1_3 = _mm_andnot_si128(d3, mOne);\
	tmp1 = b | tmp1;\
	tmp1_2 = b2 | tmp1_2;\
	tmp1_3 = b3 | tmp1_3;\
	tmp1 = tmp1 ^ c;\
	tmp1_2 = tmp1_2 ^ c2;\
	tmp1_3 = tmp1_3 ^ c3;\
	(a) = _mm_add_epi32(a,tmp1);         \
	(a2) = _mm_add_epi32(a2,tmp1_2);         \
	(a3) = _mm_add_epi32(a3,tmp1_3);         \
	(a) = _mm_add_epi32(a,AC);  \
	(a2) = _mm_add_epi32(a2,AC);  \
	(a3) = _mm_add_epi32(a3,AC);  \
	(a) = _mm_add_epi32(a,x);  \
	(a2) = _mm_add_epi32(a2,x2);  \
	(a3) = _mm_add_epi32(a3,x3);  \
	tmp1 = _mm_slli_epi32((a), (s));\
	tmp1_2 = _mm_slli_epi32((a2), (s));\
	tmp1_3 = _mm_slli_epi32((a3), (s));\
	(a) = _mm_srli_epi32((a), (32-s));\
	(a2) = _mm_srli_epi32((a2), (32-s));\
	(a3) = _mm_srli_epi32((a3), (32-s));\
	(a) = tmp1 | (a);\
	(a2) = tmp1_2 | (a2);\
	(a3) = tmp1_3 | (a3);\
	(a) = _mm_add_epi32(a,b);                       \
	(a2) = _mm_add_epi32(a2,b2);                       \
	(a3) = _mm_add_epi32(a3,b3);                       \
}

#define MD5STEP_ROUND4_NULL(f, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, AC, s) { \
	tmp1 = _mm_andnot_si128(d, mOne);\
	tmp1_2 = _mm_andnot_si128(d2, mOne);\
	tmp1_3 = _mm_andnot_si128(d3, mOne);\
	tmp1 = b | tmp1;\
	tmp1_2 = b2 | tmp1_2;\
	tmp1_3 = b3 | tmp1_3;\
	tmp1 = tmp1 ^ c;\
	tmp1_2 = tmp1_2 ^ c2;\
	tmp1_3 = tmp1_3 ^ c3;\
	(a) = _mm_add_epi32(a,tmp1);         \
	(a2) = _mm_add_epi32(a2,tmp1_2);         \
	(a3) = _mm_add_epi32(a3,tmp1_3);         \
	(a) = _mm_add_epi32(a,AC);  \
	(a2) = _mm_add_epi32(a2,AC);  \
	(a3) = _mm_add_epi32(a3,AC);  \
	tmp1 = _mm_slli_epi32((a), (s));\
	tmp1_2 = _mm_slli_epi32((a2), (s));\
	tmp1_3 = _mm_slli_epi32((a3), (s));\
	(a) = _mm_srli_epi32((a), (32-s));\
	(a2) = _mm_srli_epi32((a2), (32-s));\
	(a3) = _mm_srli_epi32((a3), (32-s));\
	(a) = tmp1 | (a);\
	(a2) = tmp1_2 | (a2);\
	(a3) = tmp1_3 | (a3);\
	(a) = _mm_add_epi32(a,b);                       \
	(a2) = _mm_add_epi32(a2,b2);                       \
	(a3) = _mm_add_epi32(a3,b3);                       \
}


// The MD5 steps for round 1-3, 3xSSE2
#define MD5_STEPS_FULL() { \
	MD5STEP_ROUND1(F, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3,  mAC1, w0, w0, w0,  S11);\
	/*printf("a: %u %u %u %u\n", a.m128i_i32[0], a.m128i_i32[1], a.m128i_i32[2], a.m128i_i32[3]);\
	printf("b: %u %u %u %u\n", b.m128i_i32[0], b.m128i_i32[1], b.m128i_i32[2], b.m128i_i32[3]);\
	printf("c: %u %u %u %u\n", c.m128i_i32[0], c.m128i_i32[1], c.m128i_i32[2], c.m128i_i32[3]);\
	printf("d: %u %u %u %u\n", d.m128i_i32[0], d.m128i_i32[1], d.m128i_i32[2], d.m128i_i32[3]);*/\
	MD5STEP_ROUND1(F, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3,  mAC2, w1, w1_2, w1_3,    S12);\
	MD5STEP_ROUND1(F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3,  mAC3, w2, w2, w2,    S13);\
	MD5STEP_ROUND1(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3,  mAC4, w3, w3, w3,    S14);\
	\
	MD5STEP_ROUND1_NULL(F, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3,  mAC5,   S11);\
	MD5STEP_ROUND1_NULL(F, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3,  mAC6,   S12);\
	MD5STEP_ROUND1_NULL(F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3,  mAC7,   S13);\
	MD5STEP_ROUND1_NULL(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3,  mAC8,   S14);\
	MD5STEP_ROUND1_NULL(F, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3,  mAC9,   S11);\
	MD5STEP_ROUND1_NULL(F, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC10,   S12);\
	MD5STEP_ROUND1_NULL(F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC11,   S13);\
	MD5STEP_ROUND1_NULL(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC12,   S14);\
	MD5STEP_ROUND1_NULL(F, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC13,   S11);\
	MD5STEP_ROUND1_NULL(F, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC14,   S12);\
	MD5STEP_ROUND1     (F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC15, w14, w14, w14, S13);\
	MD5STEP_ROUND1_NULL(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC16,   S14);\
	\
	MD5STEP_ROUND2     (G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC17, w1, w1_2, w1_3,  S21);\
	MD5STEP_ROUND2_NULL(G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC18,   S22);\
	MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC19,   S23);\
	MD5STEP_ROUND2     (G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC20, w0, w0, w0,  S24);\
	MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC21,   S21);\
	MD5STEP_ROUND2_NULL(G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC22,   S22);\
	MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC23,   S23);\
	MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC24,   S24);\
	MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC25,   S21);\
	MD5STEP_ROUND2     (G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC26, w14, w14, w14, S22);\
	MD5STEP_ROUND2     (G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC27, w3, w3, w3,  S23);\
	MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC28,   S24);\
	MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC29,   S21);\
	MD5STEP_ROUND2     (G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC30, w2, w2, w2,  S22);\
	MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC31,   S23);\
	MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC32,   S24);\
	\
	MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC33,   S31);\
	MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC34,   S32);\
	MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC35,   S33);\
	MD5STEP_ROUND3     (H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC36, w14, w14, w14, S34);\
	MD5STEP_ROUND3     (H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC37, w1, w1_2, w1_3,  S31);\
	MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC38,   S32);\
	MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC39,   S33);\
	MD5STEP_ROUND3_NULL(H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC40,   S34);\
	MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC41,   S31);\
	MD5STEP_ROUND3     (H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC42, w0, w0, w0,  S32);\
	MD5STEP_ROUND3     (H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC43, w3, w3, w3,  S33);\
	MD5STEP_ROUND3_NULL(H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC44,   S34);\
	MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC45,   S31);\
	MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC46,   S32);\
	MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC47,   S33);\
	MD5STEP_ROUND3     (H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC48, w2, w2, w2, S34);\
	\
	MD5STEP_ROUND4     (I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC49, w0, w0, w0,  S41);\
	MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC50,   S42);\
	MD5STEP_ROUND4     (I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC51, w14, w14, w14,  S43);\
	MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC52,   S44);\
	MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC53,   S41);\
	MD5STEP_ROUND4     (I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC54, w3, w3, w3,  S42);\
	MD5STEP_ROUND4_NULL(I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC55,   S43);\
	MD5STEP_ROUND4     (I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC56, w1, w1_2, w1_3,  S44);\
	MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC57,   S41);\
	MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC58,   S42);\
	MD5STEP_ROUND4_NULL(I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC59,   S43);\
	MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC60,   S44);\
	MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC61,   S41);\
	MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC62,   S42);\
	MD5STEP_ROUND4     (I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC63, w2, w2, w2,  S43);\
	MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC64,   S44);\
}

// The MD5 steps for round 1-3, 3xSSE2
#define MD5_STEPS_FIRST_3_ROUNDS() { \
	MD5STEP_ROUND1(F, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3,  mAC1, w0, w0, w0,  S11);\
	MD5STEP_ROUND1(F, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3,  mAC2, w1, w1_2, w1_3,    S12);\
	MD5STEP_ROUND1(F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3,  mAC3, w2, w2, w2,    S13);\
	MD5STEP_ROUND1(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3,  mAC4, w3, w3, w3,    S14);\
	\
	MD5STEP_ROUND1_NULL(F, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3,  mAC5,   S11);\
	MD5STEP_ROUND1_NULL(F, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3,  mAC6,   S12);\
	MD5STEP_ROUND1_NULL(F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3,  mAC7,   S13);\
	MD5STEP_ROUND1_NULL(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3,  mAC8,   S14);\
	MD5STEP_ROUND1_NULL(F, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3,  mAC9,   S11);\
	MD5STEP_ROUND1_NULL(F, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC10,   S12);\
	MD5STEP_ROUND1_NULL(F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC11,   S13);\
	MD5STEP_ROUND1_NULL(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC12,   S14);\
	MD5STEP_ROUND1_NULL(F, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC13,   S11);\
	MD5STEP_ROUND1_NULL(F, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC14,   S12);\
	MD5STEP_ROUND1     (F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC15, w14, w14, w14, S13);\
	MD5STEP_ROUND1_NULL(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC16,   S14);\
	\
	MD5STEP_ROUND2     (G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC17, w1, w1_2, w1_3,  S21);\
	MD5STEP_ROUND2_NULL(G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC18,   S22);\
	MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC19,   S23);\
	MD5STEP_ROUND2     (G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC20, w0, w0, w0,  S24);\
	MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC21,   S21);\
	MD5STEP_ROUND2_NULL(G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC22,   S22);\
	MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC23,   S23);\
	MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC24,   S24);\
	MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC25,   S21);\
	MD5STEP_ROUND2     (G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC26, w14, w14, w14, S22);\
	MD5STEP_ROUND2     (G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC27, w3, w3, w3,  S23);\
	MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC28,   S24);\
	MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC29,   S21);\
	MD5STEP_ROUND2     (G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC30, w2, w2, w2,  S22);\
	MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC31,   S23);\
	MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC32,   S24);\
	\
	MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC33,   S31);\
	MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC34,   S32);\
	MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC35,   S33);\
	MD5STEP_ROUND3     (H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC36, w14, w14, w14, S34);\
	MD5STEP_ROUND3     (H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC37, w1, w1_2, w1_3,  S31);\
	MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC38,   S32);\
	MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC39,   S33);\
	MD5STEP_ROUND3_NULL(H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC40,   S34);\
	MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC41,   S31);\
	MD5STEP_ROUND3     (H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC42, w0, w0, w0,  S32);\
	MD5STEP_ROUND3     (H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC43, w3, w3, w3,  S33);\
	MD5STEP_ROUND3_NULL(H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC44,   S34);\
	MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC45,   S31);\
	MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC46,   S32);\
	MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC47,   S33);\
	MD5STEP_ROUND3     (H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC48, w2, w2, w2, S34);\
}

#define initializeVariables() \
	__m128i mAC1			 = _mm_set1_epi32(0xd76aa478);\
	__m128i mAC2			 = _mm_set1_epi32(0xe8c7b756);\
	__m128i mAC3			 = _mm_set1_epi32(0x242070db);\
	__m128i mAC4			 = _mm_set1_epi32(0xc1bdceee);\
	__m128i mAC5			 = _mm_set1_epi32(0xf57c0faf);\
	__m128i mAC6			 = _mm_set1_epi32(0x4787c62a);\
	__m128i mAC7			 = _mm_set1_epi32(0xa8304613);\
	__m128i mAC8			 = _mm_set1_epi32(0xfd469501);\
	__m128i mAC9			 = _mm_set1_epi32(0x698098d8);\
	__m128i mAC10			 = _mm_set1_epi32(0x8b44f7af);\
	__m128i mAC11			 = _mm_set1_epi32(0xffff5bb1);\
	__m128i mAC12			 = _mm_set1_epi32(0x895cd7be);\
	__m128i mAC13			 = _mm_set1_epi32(0x6b901122);\
	__m128i mAC14			 = _mm_set1_epi32(0xfd987193);\
	__m128i mAC15			 = _mm_set1_epi32(0xa679438e);\
	__m128i mAC16			 = _mm_set1_epi32(0x49b40821);\
	__m128i mAC17			 = _mm_set1_epi32(0xf61e2562);\
	__m128i mAC18			 = _mm_set1_epi32(0xc040b340);\
	__m128i mAC19			 = _mm_set1_epi32(0x265e5a51);\
	__m128i mAC20			 = _mm_set1_epi32(0xe9b6c7aa);\
	__m128i mAC21			 = _mm_set1_epi32(0xd62f105d);\
	__m128i mAC22			 = _mm_set1_epi32(0x02441453);\
	__m128i mAC23			 = _mm_set1_epi32(0xd8a1e681);\
	__m128i mAC24			 = _mm_set1_epi32(0xe7d3fbc8);\
	__m128i mAC25			 = _mm_set1_epi32(0x21e1cde6);\
	__m128i mAC26			 = _mm_set1_epi32(0xc33707d6);\
	__m128i mAC27			 = _mm_set1_epi32(0xf4d50d87);\
	__m128i mAC28			 = _mm_set1_epi32(0x455a14ed);\
	__m128i mAC29			 = _mm_set1_epi32(0xa9e3e905);\
	__m128i mAC30			 = _mm_set1_epi32(0xfcefa3f8);\
	__m128i mAC31			 = _mm_set1_epi32(0x676f02d9);\
	__m128i mAC32			 = _mm_set1_epi32(0x8d2a4c8a);\
	__m128i mAC33			 = _mm_set1_epi32(0xfffa3942);\
	__m128i mAC34			 = _mm_set1_epi32(0x8771f681);\
	__m128i mAC35			 = _mm_set1_epi32(0x6d9d6122);\
	__m128i mAC36			 = _mm_set1_epi32(0xfde5380c);\
	__m128i mAC37			 = _mm_set1_epi32(0xa4beea44);\
	__m128i mAC38			 = _mm_set1_epi32(0x4bdecfa9);\
	__m128i mAC39			 = _mm_set1_epi32(0xf6bb4b60);\
	__m128i mAC40			 = _mm_set1_epi32(0xbebfbc70);\
	__m128i mAC41			 = _mm_set1_epi32(0x289b7ec6);\
	__m128i mAC42			 = _mm_set1_epi32(0xeaa127fa);\
	__m128i mAC43			 = _mm_set1_epi32(0xd4ef3085);\
	__m128i mAC44			 = _mm_set1_epi32(0x04881d05);\
	__m128i mAC45			 = _mm_set1_epi32(0xd9d4d039);\
	__m128i mAC46			 = _mm_set1_epi32(0xe6db99e5);\
	__m128i mAC47			 = _mm_set1_epi32(0x1fa27cf8);\
	__m128i mAC48			 = _mm_set1_epi32(0xc4ac5665);\
	__m128i mAC49			 = _mm_set1_epi32(0xf4292244);\
	__m128i mAC50			 = _mm_set1_epi32(0x432aff97);\
	__m128i mAC51			 = _mm_set1_epi32(0xab9423a7);\
	__m128i mAC52			 = _mm_set1_epi32(0xfc93a039);\
	__m128i mAC53			 = _mm_set1_epi32(0x655b59c3);\
	__m128i mAC54			 = _mm_set1_epi32(0x8f0ccc92);\
	__m128i mAC55			 = _mm_set1_epi32(0xffeff47d);\
	__m128i mAC56			 = _mm_set1_epi32(0x85845dd1);\
	__m128i mAC57			 = _mm_set1_epi32(0x6fa87e4f);\
	__m128i mAC58			 = _mm_set1_epi32(0xfe2ce6e0);\
	__m128i mAC59			 = _mm_set1_epi32(0xa3014314);\
	__m128i mAC60			 = _mm_set1_epi32(0x4e0811a1);\
	__m128i mAC61			 = _mm_set1_epi32(0xf7537e82);\
	__m128i mAC62			 = _mm_set1_epi32(0xbd3af235);\
	__m128i mAC63			 = _mm_set1_epi32(0x2ad7d2bb);\
	__m128i mAC64			 = _mm_set1_epi32(0xeb86d391);\
	\
	__m128i	mCa = _mm_set1_epi32(Ca);\
	__m128i	mCb = _mm_set1_epi32(Cb);\
	__m128i	mCc = _mm_set1_epi32(Cc);\
	__m128i	mCd = _mm_set1_epi32(Cd);\
	\
	unsigned int i;\



