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

/*
 * Implementation of the MD4 message-digest algorithm, optimized for passwords with length < 32 (or NTLM < 16)
 * (see http://www.ietf.org/rfc/rfc1320.txt)
 *
 * Author: Dani�l Niggebrugge
 * License: Use and share as you wish at your own risk, please keep this header ;)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, [...] etc :p
 *
 */

/**
 * Some of code is used with permission from Neinbrucke out of his EnTibr tools.
 *
 * Permission to use his source as GPLv2 was obtained from him directly.
 *
 * The interlaced MD4 operations are his, and are integrated into the Cryptohaze
 * toolchain by Bitweasil.
 */


#include <emmintrin.h>

typedef unsigned int UINT4;

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

// Define rotations
#define S11 3
#define S12 7
#define S13 11
#define S14 19
#define S21 3
#define S22 5
#define S23 9
#define S24 13
#define S31 3
#define S32 9
#define S33 11
#define S34 15

// Define initial values
#define Ca 				0x67452301
#define Cb 				0xefcdab89
#define Cc 				0x98badcfe
#define Cd 				0x10325476

// Define functions for use in the 3 rounds of MD4
#define F(x, y, z)   (((x) & (y)) | (_mm_andnot_si128((x),(z))))
#define G(x, y, z)   ((((x) & (y)) | (z)) & ((x) | (y)))
#define H(x, y, z)   ((x) ^ (y) ^ (z))

#define ROTATE_LEFT(a,n) _mm_or_si128(_mm_slli_epi32(a, n), _mm_srli_epi32(a, (32-n)))




#define MD4STEP_ROUND1_4(a, b, c, d, x, s) {	\
	tmp1_0 = (b##_0) & (c##_0);\
	tmp1_1 = (b##_1) & (c##_1);\
	tmp1_2 = (b##_2) & (c##_2);\
	tmp1_3 = (b##_3) & (c##_3);\
	tmp2_0 = _mm_andnot_si128((b##_0),(d##_0));\
	tmp2_1 = _mm_andnot_si128((b##_1),(d##_1));\
	tmp2_2 = _mm_andnot_si128((b##_2),(d##_2));\
	tmp2_3 = _mm_andnot_si128((b##_3),(d##_3));\
	tmp1_0 = tmp1_0 | tmp2_0;\
	tmp1_1 = tmp1_1 | tmp2_1;\
	tmp1_2 = tmp1_2 | tmp2_2;\
	tmp1_3 = tmp1_3 | tmp2_3;\
        (a##_0) = _mm_add_epi32(a##_0,tmp1_0);		\
	(a##_1) = _mm_add_epi32(a##_1,tmp1_1);		\
	(a##_2) = _mm_add_epi32(a##_2,tmp1_2);		\
	(a##_3) = _mm_add_epi32(a##_3,tmp1_3);		\
	(a##_0) = _mm_add_epi32(a##_0,x##_0);				    \
	(a##_1) = _mm_add_epi32(a##_1,x##_1);				    \
	(a##_2) = _mm_add_epi32(a##_2,x##_2);				    \
	(a##_3) = _mm_add_epi32(a##_3,x##_3);				    \
	tmp1_0 = _mm_slli_epi32((a##_0), (s));\
	tmp1_1 = _mm_slli_epi32((a##_1), (s));\
	tmp1_2 = _mm_slli_epi32((a##_2), (s));\
	tmp1_3 = _mm_slli_epi32((a##_3), (s));\
	(a##_0) = _mm_srli_epi32((a##_0), (32-s));\
	(a##_1) = _mm_srli_epi32((a##_1), (32-s));\
	(a##_2) = _mm_srli_epi32((a##_2), (32-s));\
	(a##_3) = _mm_srli_epi32((a##_3), (32-s));\
	(a##_0) = tmp1_0 | (a##_0);\
	(a##_1) = tmp1_1 | (a##_1);\
	(a##_2) = tmp1_2 | (a##_2);\
	(a##_3) = tmp1_3 | (a##_3);\
}

#define MD4STEP_ROUND1_NULL_4(a, b, c, d, s) {	\
	tmp1_0 = (b##_0) & (c##_0);\
	tmp1_1 = (b##_1) & (c##_1);\
	tmp1_2 = (b##_2) & (c##_2);\
	tmp1_3 = (b##_3) & (c##_3);\
	tmp2_0 = _mm_andnot_si128((b##_0),(d##_0));\
	tmp2_1 = _mm_andnot_si128((b##_1),(d##_1));\
	tmp2_2 = _mm_andnot_si128((b##_2),(d##_2));\
	tmp2_3 = _mm_andnot_si128((b##_3),(d##_3));\
	tmp1_0 = tmp1_0 | tmp2_0;\
	tmp1_1 = tmp1_1 | tmp2_1;\
	tmp1_2 = tmp1_2 | tmp2_2;\
	tmp1_3 = tmp1_3 | tmp2_3;\
        (a##_0) = _mm_add_epi32(a##_0,tmp1_0);		\
	(a##_1) = _mm_add_epi32(a##_1,tmp1_1);		\
	(a##_2) = _mm_add_epi32(a##_2,tmp1_2);		\
	(a##_3) = _mm_add_epi32(a##_3,tmp1_3);		\
	tmp1_0 = _mm_slli_epi32((a##_0), (s));\
	tmp1_1 = _mm_slli_epi32((a##_1), (s));\
	tmp1_2 = _mm_slli_epi32((a##_2), (s));\
	tmp1_3 = _mm_slli_epi32((a##_3), (s));\
	(a##_0) = _mm_srli_epi32((a##_0), (32-s));\
	(a##_1) = _mm_srli_epi32((a##_1), (32-s));\
	(a##_2) = _mm_srli_epi32((a##_2), (32-s));\
	(a##_3) = _mm_srli_epi32((a##_3), (32-s));\
	(a##_0) = tmp1_0 | (a##_0);\
	(a##_1) = tmp1_1 | (a##_1);\
	(a##_2) = tmp1_2 | (a##_2);\
	(a##_3) = tmp1_3 | (a##_3);\
}

#define MD4STEP_ROUND2_4(a, b, c, d, x, s) {	\
	tmp1_0 = (b##_0) & (c##_0);\
	tmp1_1 = (b##_1) & (c##_1);\
	tmp1_2 = (b##_2) & (c##_2);\
	tmp1_3 = (b##_3) & (c##_3);\
	tmp1_0 = tmp1_0 | (d##_0);\
	tmp1_1 = tmp1_1 | (d##_1);\
	tmp1_2 = tmp1_2 | (d##_2);\
	tmp1_3 = tmp1_3 | (d##_3);\
	tmp2_0 = (b##_0) | (c##_0);\
	tmp2_1 = (b##_1) | (c##_1);\
	tmp2_2 = (b##_2) | (c##_2);\
	tmp2_3 = (b##_3) | (c##_3);\
	tmp1_0 = tmp1_0 & tmp2_0;\
	tmp1_1 = tmp1_1 & tmp2_1;\
	tmp1_2 = tmp1_2 & tmp2_2;\
	tmp1_3 = tmp1_3 & tmp2_3;\
        (a##_0) = _mm_add_epi32(a##_0,tmp1_0);		\
	(a##_1) = _mm_add_epi32(a##_1,tmp1_1);		\
	(a##_2) = _mm_add_epi32(a##_2,tmp1_2);		\
	(a##_3) = _mm_add_epi32(a##_3,tmp1_3);		\
	(a##_0) = _mm_add_epi32(a##_0,x##_0);				    \
	(a##_1) = _mm_add_epi32(a##_1,x##_1);				    \
	(a##_2) = _mm_add_epi32(a##_2,x##_2);				    \
	(a##_3) = _mm_add_epi32(a##_3,x##_3);				    \
	(a##_0) = _mm_add_epi32(a##_0,AC);				    \
	(a##_1) = _mm_add_epi32(a##_1,AC);				    \
	(a##_2) = _mm_add_epi32(a##_2,AC);				    \
	(a##_3) = _mm_add_epi32(a##_3,AC);				    \
	tmp1_0 = _mm_slli_epi32((a##_0), (s));\
	tmp1_1 = _mm_slli_epi32((a##_1), (s));\
	tmp1_2 = _mm_slli_epi32((a##_2), (s));\
	tmp1_3 = _mm_slli_epi32((a##_3), (s));\
	(a##_0) = _mm_srli_epi32((a##_0), (32-s));\
	(a##_1) = _mm_srli_epi32((a##_1), (32-s));\
	(a##_2) = _mm_srli_epi32((a##_2), (32-s));\
	(a##_3) = _mm_srli_epi32((a##_3), (32-s));\
	(a##_0) = tmp1_0 | (a##_0);\
	(a##_1) = tmp1_1 | (a##_1);\
	(a##_2) = tmp1_2 | (a##_2);\
	(a##_3) = tmp1_3 | (a##_3);\
}

#define MD4STEP_ROUND2_NULL_4(a, b, c, d, s) {	\
	tmp1_0 = (b##_0) & (c##_0);\
	tmp1_1 = (b##_1) & (c##_1);\
	tmp1_2 = (b##_2) & (c##_2);\
	tmp1_3 = (b##_3) & (c##_3);\
	tmp1_0 = tmp1_0 | (d##_0);\
	tmp1_1 = tmp1_1 | (d##_1);\
	tmp1_2 = tmp1_2 | (d##_2);\
	tmp1_3 = tmp1_3 | (d##_3);\
	tmp2_0 = (b##_0) | (c##_0);\
	tmp2_1 = (b##_1) | (c##_1);\
	tmp2_2 = (b##_2) | (c##_2);\
	tmp2_3 = (b##_3) | (c##_3);\
	tmp1_0 = tmp1_0 & tmp2_0;\
	tmp1_1 = tmp1_1 & tmp2_1;\
	tmp1_2 = tmp1_2 & tmp2_2;\
	tmp1_3 = tmp1_3 & tmp2_3;\
        (a##_0) = _mm_add_epi32(a##_0,tmp1_0);		\
	(a##_1) = _mm_add_epi32(a##_1,tmp1_1);		\
	(a##_2) = _mm_add_epi32(a##_2,tmp1_2);		\
	(a##_3) = _mm_add_epi32(a##_3,tmp1_3);		\
	(a##_0) = _mm_add_epi32(a##_0,AC);				    \
	(a##_1) = _mm_add_epi32(a##_1,AC);				    \
	(a##_2) = _mm_add_epi32(a##_2,AC);				    \
	(a##_3) = _mm_add_epi32(a##_3,AC);				    \
	tmp1_0 = _mm_slli_epi32((a##_0), (s));\
	tmp1_1 = _mm_slli_epi32((a##_1), (s));\
	tmp1_2 = _mm_slli_epi32((a##_2), (s));\
	tmp1_3 = _mm_slli_epi32((a##_3), (s));\
	(a##_0) = _mm_srli_epi32((a##_0), (32-s));\
	(a##_1) = _mm_srli_epi32((a##_1), (32-s));\
	(a##_2) = _mm_srli_epi32((a##_2), (32-s));\
	(a##_3) = _mm_srli_epi32((a##_3), (32-s));\
	(a##_0) = tmp1_0 | (a##_0);\
	(a##_1) = tmp1_1 | (a##_1);\
	(a##_2) = tmp1_2 | (a##_2);\
	(a##_3) = tmp1_3 | (a##_3);\
}

#define MD4STEP_ROUND3_4(a, b, c, d, x, s) {	\
	tmp1_0 = (b##_0) ^ (c##_0);\
	tmp1_1 = (b##_1) ^ (c##_1);\
	tmp1_2 = (b##_2) ^ (c##_2);\
	tmp1_3 = (b##_3) ^ (c##_3);\
	tmp1_0 = (tmp1_0) ^ (d##_0);\
	tmp1_1 = (tmp1_1) ^ (d##_1);\
	tmp1_2 = (tmp1_2) ^ (d##_2);\
	tmp1_3 = (tmp1_3) ^ (d##_3);\
        (a##_0) = _mm_add_epi32(a##_0,tmp1_0);		\
	(a##_1) = _mm_add_epi32(a##_1,tmp1_1);		\
	(a##_2) = _mm_add_epi32(a##_2,tmp1_2);		\
	(a##_3) = _mm_add_epi32(a##_3,tmp1_3);		\
	(a##_0) = _mm_add_epi32(a##_0,x##_0);				    \
	(a##_1) = _mm_add_epi32(a##_1,x##_1);				    \
	(a##_2) = _mm_add_epi32(a##_2,x##_2);				    \
	(a##_3) = _mm_add_epi32(a##_3,x##_3);				    \
	(a##_0) = _mm_add_epi32(a##_0,AC2);				    \
	(a##_1) = _mm_add_epi32(a##_1,AC2);				    \
	(a##_2) = _mm_add_epi32(a##_2,AC2);				    \
	(a##_3) = _mm_add_epi32(a##_3,AC2);				    \
	tmp1_0 = _mm_slli_epi32((a##_0), (s));\
	tmp1_1 = _mm_slli_epi32((a##_1), (s));\
	tmp1_2 = _mm_slli_epi32((a##_2), (s));\
	tmp1_3 = _mm_slli_epi32((a##_3), (s));\
	(a##_0) = _mm_srli_epi32((a##_0), (32-s));\
	(a##_1) = _mm_srli_epi32((a##_1), (32-s));\
	(a##_2) = _mm_srli_epi32((a##_2), (32-s));\
	(a##_3) = _mm_srli_epi32((a##_3), (32-s));\
	(a##_0) = tmp1_0 | (a##_0);\
	(a##_1) = tmp1_1 | (a##_1);\
	(a##_2) = tmp1_2 | (a##_2);\
	(a##_3) = tmp1_3 | (a##_3);\
}

#define MD4STEP_ROUND3_NULL_4(a, b, c, d, s) {	\
	tmp1_0 = (b##_0) ^ (c##_0);\
	tmp1_1 = (b##_1) ^ (c##_1);\
	tmp1_2 = (b##_2) ^ (c##_2);\
	tmp1_3 = (b##_3) ^ (c##_3);\
	tmp1_0 = (tmp1_0) ^ (d##_0);\
	tmp1_1 = (tmp1_1) ^ (d##_1);\
	tmp1_2 = (tmp1_2) ^ (d##_2);\
	tmp1_3 = (tmp1_3) ^ (d##_3);\
        (a##_0) = _mm_add_epi32(a##_0,tmp1_0);		\
	(a##_1) = _mm_add_epi32(a##_1,tmp1_1);		\
	(a##_2) = _mm_add_epi32(a##_2,tmp1_2);		\
	(a##_3) = _mm_add_epi32(a##_3,tmp1_3);		\
	(a##_0) = _mm_add_epi32(a##_0,AC2);				    \
	(a##_1) = _mm_add_epi32(a##_1,AC2);				    \
	(a##_2) = _mm_add_epi32(a##_2,AC2);				    \
	(a##_3) = _mm_add_epi32(a##_3,AC2);				    \
	tmp1_0 = _mm_slli_epi32((a##_0), (s));\
	tmp1_1 = _mm_slli_epi32((a##_1), (s));\
	tmp1_2 = _mm_slli_epi32((a##_2), (s));\
	tmp1_3 = _mm_slli_epi32((a##_3), (s));\
	(a##_0) = _mm_srli_epi32((a##_0), (32-s));\
	(a##_1) = _mm_srli_epi32((a##_1), (32-s));\
	(a##_2) = _mm_srli_epi32((a##_2), (32-s));\
	(a##_3) = _mm_srli_epi32((a##_3), (32-s));\
	(a##_0) = tmp1_0 | (a##_0);\
	(a##_1) = tmp1_1 | (a##_1);\
	(a##_2) = tmp1_2 | (a##_2);\
	(a##_3) = tmp1_3 | (a##_3);\
}


// Interlacing 4x SSE2
#define MD4_STEPS_FULL() { \
        MD4STEP_ROUND1_4      (a, b, c, d, b0, S11); \
	MD4STEP_ROUND1_4      (d, a, b, c, b1, S12); \
	MD4STEP_ROUND1_4      (c, d, a, b, b2, S13); \
	MD4STEP_ROUND1_4      (b, c, d, a, b3, S14); \
	MD4STEP_ROUND1_4      (a, b, c, d, b4, S11); \
	MD4STEP_ROUND1_4      (d, a, b, c, b5, S12); \
	MD4STEP_ROUND1_4      (c, d, a, b, b6, S13); \
	MD4STEP_ROUND1_4      (b, c, d, a, b7, S14); \
	MD4STEP_ROUND1_NULL_4 (a, b, c, d, S11); \
	MD4STEP_ROUND1_NULL_4 (d, a, b, c, S12); \
	MD4STEP_ROUND1_NULL_4 (c, d, a, b, S13); \
	MD4STEP_ROUND1_NULL_4 (b, c, d, a, S14); \
	MD4STEP_ROUND1_NULL_4 (a, b, c, d, S11); \
	MD4STEP_ROUND1_NULL_4 (d, a, b, c, S12); \
	MD4STEP_ROUND1_4      (c, d, a, b, b14, S13); \
	MD4STEP_ROUND1_NULL_4 (b, c, d, a, S14); \
	\
	MD4STEP_ROUND2_4      (a, b, c, d, b0, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, b4, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_NULL_4 (b, c, d, a, S24);\
	MD4STEP_ROUND2_4      (a, b, c, d, b1, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, b5, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_NULL_4 (b, c, d, a, S24);\
	MD4STEP_ROUND2_4      (a, b, c, d, b2, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, b6, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_4      (b, c, d, a, b14, S24);\
	MD4STEP_ROUND2_4      (a, b, c, d, b3, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, b7, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_NULL_4 (b, c, d, a, S24);\
        \
        MD4STEP_ROUND3_4      (a, b, c, d, b0, S31);\
	MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32);\
	MD4STEP_ROUND3_4      (c, d, a, b, b4, S33);\
	MD4STEP_ROUND3_NULL_4 (b, c, d, a, S34);\
	MD4STEP_ROUND3_4      (a, b, c, d, b2, S31);\
	MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32); \
        MD4STEP_ROUND3_4      (c, d, a, b, b6, S33); \
        MD4STEP_ROUND3_4      (b, c, d, a, b14, S34);\
        MD4STEP_ROUND3_4      (a, b, c, d, b1, S31);\
	MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32);\
	MD4STEP_ROUND3_4      (c, d, a, b, b5, S33);\
	MD4STEP_ROUND3_NULL_4 (b, c, d, a, S34);\
        MD4STEP_ROUND3_4      (a, b, c, d, b3, S31);\
	MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32);\
	MD4STEP_ROUND3_4      (c, d, a, b, b7, S33);\
	MD4STEP_ROUND3_NULL_4 (b, c, d, a, S34);\
}

// Interlacing 4x SSE2
#define MD4_STEPS_FIRST2_ROUNDS() { \
        MD4STEP_ROUND1_4      (a, b, c, d, b0, S11); \
	MD4STEP_ROUND1_4      (d, a, b, c, b1, S12); \
	MD4STEP_ROUND1_4      (c, d, a, b, b2, S13); \
	MD4STEP_ROUND1_4      (b, c, d, a, b3, S14); \
	MD4STEP_ROUND1_4      (a, b, c, d, b4, S11); \
	MD4STEP_ROUND1_4      (d, a, b, c, b5, S12); \
	MD4STEP_ROUND1_4      (c, d, a, b, b6, S13); \
	MD4STEP_ROUND1_4      (b, c, d, a, b7, S14); \
	MD4STEP_ROUND1_NULL_4 (a, b, c, d, S11); \
	MD4STEP_ROUND1_NULL_4 (d, a, b, c, S12); \
	MD4STEP_ROUND1_NULL_4 (c, d, a, b, S13); \
	MD4STEP_ROUND1_NULL_4 (b, c, d, a, S14); \
	MD4STEP_ROUND1_NULL_4 (a, b, c, d, S11); \
	MD4STEP_ROUND1_NULL_4 (d, a, b, c, S12); \
	MD4STEP_ROUND1_4      (c, d, a, b, b14, S13); \
	MD4STEP_ROUND1_NULL_4 (b, c, d, a, S14); \
	\
	MD4STEP_ROUND2_4      (a, b, c, d, b0, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, b4, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_NULL_4 (b, c, d, a, S24);\
	MD4STEP_ROUND2_4      (a, b, c, d, b1, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, b5, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_NULL_4 (b, c, d, a, S24);\
	MD4STEP_ROUND2_4      (a, b, c, d, b2, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, b6, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_4      (b, c, d, a, b14, S24);\
	MD4STEP_ROUND2_4      (a, b, c, d, b3, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, b7, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_NULL_4 (b, c, d, a, S24);\
}
