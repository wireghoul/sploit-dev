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
//#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
//#define G(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define F(x, y, z)   (((x) & (y)) | (_mm_andnot_si128((x),(z))))
#define G(x, y, z)   ((((x) & (y)) | (z)) & ((x) | (y)))
#define H(x, y, z)   ((x) ^ (y) ^ (z))

#define ROTATE_LEFT(a,n) _mm_or_si128(_mm_slli_epi32(a, n), _mm_srli_epi32(a, (32-n)))
#define ROTATE_RIGHT(a,n) _mm_or_si128(_mm_srli_epi32(a, n), _mm_slli_epi32(a, (32-n)))
// Rotate right is only used in reversing code, which isn't SSE2 code
#define ROTATE_RIGHT_SMALL(x, n)	(((x) >> (n)) | ((x) << (32-(n))))



#define MD4STEP_ROUND1_4(a, b, c, d, x, s) {	\
	tmp1_0 = (b##0) & (c##0);\
	tmp1_1 = (b##1) & (c##1);\
	tmp1_2 = (b##2) & (c##2);\
	tmp1_3 = (b##3) & (c##3);\
	tmp2_0 = _mm_andnot_si128((b##0),(d##0));\
	tmp2_1 = _mm_andnot_si128((b##1),(d##1));\
	tmp2_2 = _mm_andnot_si128((b##2),(d##2));\
	tmp2_3 = _mm_andnot_si128((b##3),(d##3));\
	tmp1_0 = tmp1_0 | tmp2_0;\
	tmp1_1 = tmp1_1 | tmp2_1;\
	tmp1_2 = tmp1_2 | tmp2_2;\
	tmp1_3 = tmp1_3 | tmp2_3;\
        (a##0) = _mm_add_epi32(a##0,tmp1_0);		\
	(a##1) = _mm_add_epi32(a##1,tmp1_1);		\
	(a##2) = _mm_add_epi32(a##2,tmp1_2);		\
	(a##3) = _mm_add_epi32(a##3,tmp1_3);		\
	(a##0) = _mm_add_epi32(a##0,x##_0);				    \
	(a##1) = _mm_add_epi32(a##1,x##_1);				    \
	(a##2) = _mm_add_epi32(a##2,x##_2);				    \
	(a##3) = _mm_add_epi32(a##3,x##_3);				    \
	tmp1_0 = _mm_slli_epi32((a##0), (s));\
	tmp1_1 = _mm_slli_epi32((a##1), (s));\
	tmp1_2 = _mm_slli_epi32((a##2), (s));\
	tmp1_3 = _mm_slli_epi32((a##3), (s));\
	(a##0) = _mm_srli_epi32((a##0), (32-s));\
	(a##1) = _mm_srli_epi32((a##1), (32-s));\
	(a##2) = _mm_srli_epi32((a##2), (32-s));\
	(a##3) = _mm_srli_epi32((a##3), (32-s));\
	(a##0) = tmp1_0 | (a##0);\
	(a##1) = tmp1_1 | (a##1);\
	(a##2) = tmp1_2 | (a##2);\
	(a##3) = tmp1_3 | (a##3);\
}

#define MD4STEP_ROUND1_NULL_4(a, b, c, d, s) {	\
	tmp1_0 = (b##0) & (c##0);\
	tmp1_1 = (b##1) & (c##1);\
	tmp1_2 = (b##2) & (c##2);\
	tmp1_3 = (b##3) & (c##3);\
	tmp2_0 = _mm_andnot_si128((b##0),(d##0));\
	tmp2_1 = _mm_andnot_si128((b##1),(d##1));\
	tmp2_2 = _mm_andnot_si128((b##2),(d##2));\
	tmp2_3 = _mm_andnot_si128((b##3),(d##3));\
	tmp1_0 = tmp1_0 | tmp2_0;\
	tmp1_1 = tmp1_1 | tmp2_1;\
	tmp1_2 = tmp1_2 | tmp2_2;\
	tmp1_3 = tmp1_3 | tmp2_3;\
        (a##0) = _mm_add_epi32(a##0,tmp1_0);		\
	(a##1) = _mm_add_epi32(a##1,tmp1_1);		\
	(a##2) = _mm_add_epi32(a##2,tmp1_2);		\
	(a##3) = _mm_add_epi32(a##3,tmp1_3);		\
	tmp1_0 = _mm_slli_epi32((a##0), (s));\
	tmp1_1 = _mm_slli_epi32((a##1), (s));\
	tmp1_2 = _mm_slli_epi32((a##2), (s));\
	tmp1_3 = _mm_slli_epi32((a##3), (s));\
	(a##0) = _mm_srli_epi32((a##0), (32-s));\
	(a##1) = _mm_srli_epi32((a##1), (32-s));\
	(a##2) = _mm_srli_epi32((a##2), (32-s));\
	(a##3) = _mm_srli_epi32((a##3), (32-s));\
	(a##0) = tmp1_0 | (a##0);\
	(a##1) = tmp1_1 | (a##1);\
	(a##2) = tmp1_2 | (a##2);\
	(a##3) = tmp1_3 | (a##3);\
}

#define MD4STEP_ROUND2_4(a, b, c, d, x, s) {	\
	tmp1_0 = (b##0) & (c##0);\
	tmp1_1 = (b##1) & (c##1);\
	tmp1_2 = (b##2) & (c##2);\
	tmp1_3 = (b##3) & (c##3);\
	tmp1_0 = tmp1_0 | (d##0);\
	tmp1_1 = tmp1_1 | (d##1);\
	tmp1_2 = tmp1_2 | (d##2);\
	tmp1_3 = tmp1_3 | (d##3);\
	tmp2_0 = (b##0) | (c##0);\
	tmp2_1 = (b##1) | (c##1);\
	tmp2_2 = (b##2) | (c##2);\
	tmp2_3 = (b##3) | (c##3);\
	tmp1_0 = tmp1_0 & tmp2_0;\
	tmp1_1 = tmp1_1 & tmp2_1;\
	tmp1_2 = tmp1_2 & tmp2_2;\
	tmp1_3 = tmp1_3 & tmp2_3;\
        (a##0) = _mm_add_epi32(a##0,tmp1_0);		\
	(a##1) = _mm_add_epi32(a##1,tmp1_1);		\
	(a##2) = _mm_add_epi32(a##2,tmp1_2);		\
	(a##3) = _mm_add_epi32(a##3,tmp1_3);		\
	(a##0) = _mm_add_epi32(a##0,x##_0);				    \
	(a##1) = _mm_add_epi32(a##1,x##_1);				    \
	(a##2) = _mm_add_epi32(a##2,x##_2);				    \
	(a##3) = _mm_add_epi32(a##3,x##_3);				    \
	(a##0) = _mm_add_epi32(a##0,AC);				    \
	(a##1) = _mm_add_epi32(a##1,AC);				    \
	(a##2) = _mm_add_epi32(a##2,AC);				    \
	(a##3) = _mm_add_epi32(a##3,AC);				    \
	tmp1_0 = _mm_slli_epi32((a##0), (s));\
	tmp1_1 = _mm_slli_epi32((a##1), (s));\
	tmp1_2 = _mm_slli_epi32((a##2), (s));\
	tmp1_3 = _mm_slli_epi32((a##3), (s));\
	(a##0) = _mm_srli_epi32((a##0), (32-s));\
	(a##1) = _mm_srli_epi32((a##1), (32-s));\
	(a##2) = _mm_srli_epi32((a##2), (32-s));\
	(a##3) = _mm_srli_epi32((a##3), (32-s));\
	(a##0) = tmp1_0 | (a##0);\
	(a##1) = tmp1_1 | (a##1);\
	(a##2) = tmp1_2 | (a##2);\
	(a##3) = tmp1_3 | (a##3);\
}

#define MD4STEP_ROUND2_NULL_4(a, b, c, d, s) {	\
	tmp1_0 = (b##0) & (c##0);\
	tmp1_1 = (b##1) & (c##1);\
	tmp1_2 = (b##2) & (c##2);\
	tmp1_3 = (b##3) & (c##3);\
	tmp1_0 = tmp1_0 | (d##0);\
	tmp1_1 = tmp1_1 | (d##1);\
	tmp1_2 = tmp1_2 | (d##2);\
	tmp1_3 = tmp1_3 | (d##3);\
	tmp2_0 = (b##0) | (c##0);\
	tmp2_1 = (b##1) | (c##1);\
	tmp2_2 = (b##2) | (c##2);\
	tmp2_3 = (b##3) | (c##3);\
	tmp1_0 = tmp1_0 & tmp2_0;\
	tmp1_1 = tmp1_1 & tmp2_1;\
	tmp1_2 = tmp1_2 & tmp2_2;\
	tmp1_3 = tmp1_3 & tmp2_3;\
        (a##0) = _mm_add_epi32(a##0,tmp1_0);		\
	(a##1) = _mm_add_epi32(a##1,tmp1_1);		\
	(a##2) = _mm_add_epi32(a##2,tmp1_2);		\
	(a##3) = _mm_add_epi32(a##3,tmp1_3);		\
	(a##0) = _mm_add_epi32(a##0,AC);				    \
	(a##1) = _mm_add_epi32(a##1,AC);				    \
	(a##2) = _mm_add_epi32(a##2,AC);				    \
	(a##3) = _mm_add_epi32(a##3,AC);				    \
	tmp1_0 = _mm_slli_epi32((a##0), (s));\
	tmp1_1 = _mm_slli_epi32((a##1), (s));\
	tmp1_2 = _mm_slli_epi32((a##2), (s));\
	tmp1_3 = _mm_slli_epi32((a##3), (s));\
	(a##0) = _mm_srli_epi32((a##0), (32-s));\
	(a##1) = _mm_srli_epi32((a##1), (32-s));\
	(a##2) = _mm_srli_epi32((a##2), (32-s));\
	(a##3) = _mm_srli_epi32((a##3), (32-s));\
	(a##0) = tmp1_0 | (a##0);\
	(a##1) = tmp1_1 | (a##1);\
	(a##2) = tmp1_2 | (a##2);\
	(a##3) = tmp1_3 | (a##3);\
}

#define MD4STEP_ROUND3_4(a, b, c, d, x, s) {	\
	tmp1_0 = (b##0) ^ (c##0);\
	tmp1_1 = (b##1) ^ (c##1);\
	tmp1_2 = (b##2) ^ (c##2);\
	tmp1_3 = (b##3) ^ (c##3);\
	tmp1_0 = (tmp1_0) ^ (d##0);\
	tmp1_1 = (tmp1_1) ^ (d##1);\
	tmp1_2 = (tmp1_2) ^ (d##2);\
	tmp1_3 = (tmp1_3) ^ (d##3);\
        (a##0) = _mm_add_epi32(a##0,tmp1_0);		\
	(a##1) = _mm_add_epi32(a##1,tmp1_1);		\
	(a##2) = _mm_add_epi32(a##2,tmp1_2);		\
	(a##3) = _mm_add_epi32(a##3,tmp1_3);		\
	(a##0) = _mm_add_epi32(a##0,x##_0);				    \
	(a##1) = _mm_add_epi32(a##1,x##_1);				    \
	(a##2) = _mm_add_epi32(a##2,x##_2);				    \
	(a##3) = _mm_add_epi32(a##3,x##_3);				    \
	(a##0) = _mm_add_epi32(a##0,AC2);				    \
	(a##1) = _mm_add_epi32(a##1,AC2);				    \
	(a##2) = _mm_add_epi32(a##2,AC2);				    \
	(a##3) = _mm_add_epi32(a##3,AC2);				    \
	tmp1_0 = _mm_slli_epi32((a##0), (s));\
	tmp1_1 = _mm_slli_epi32((a##1), (s));\
	tmp1_2 = _mm_slli_epi32((a##2), (s));\
	tmp1_3 = _mm_slli_epi32((a##3), (s));\
	(a##0) = _mm_srli_epi32((a##0), (32-s));\
	(a##1) = _mm_srli_epi32((a##1), (32-s));\
	(a##2) = _mm_srli_epi32((a##2), (32-s));\
	(a##3) = _mm_srli_epi32((a##3), (32-s));\
	(a##0) = tmp1_0 | (a##0);\
	(a##1) = tmp1_1 | (a##1);\
	(a##2) = tmp1_2 | (a##2);\
	(a##3) = tmp1_3 | (a##3);\
}

#define MD4STEP_ROUND3_NULL_4(a, b, c, d, s) {	\
	tmp1_0 = (b##0) ^ (c##0);\
	tmp1_1 = (b##1) ^ (c##1);\
	tmp1_2 = (b##2) ^ (c##2);\
	tmp1_3 = (b##3) ^ (c##3);\
	tmp1_0 = (tmp1_0) ^ (d##0);\
	tmp1_1 = (tmp1_1) ^ (d##1);\
	tmp1_2 = (tmp1_2) ^ (d##2);\
	tmp1_3 = (tmp1_3) ^ (d##3);\
        (a##0) = _mm_add_epi32(a##0,tmp1_0);		\
	(a##1) = _mm_add_epi32(a##1,tmp1_1);		\
	(a##2) = _mm_add_epi32(a##2,tmp1_2);		\
	(a##3) = _mm_add_epi32(a##3,tmp1_3);		\
	(a##0) = _mm_add_epi32(a##0,AC2);				    \
	(a##1) = _mm_add_epi32(a##1,AC2);				    \
	(a##2) = _mm_add_epi32(a##2,AC2);				    \
	(a##3) = _mm_add_epi32(a##3,AC2);				    \
	tmp1_0 = _mm_slli_epi32((a##0), (s));\
	tmp1_1 = _mm_slli_epi32((a##1), (s));\
	tmp1_2 = _mm_slli_epi32((a##2), (s));\
	tmp1_3 = _mm_slli_epi32((a##3), (s));\
	(a##0) = _mm_srli_epi32((a##0), (32-s));\
	(a##1) = _mm_srli_epi32((a##1), (32-s));\
	(a##2) = _mm_srli_epi32((a##2), (32-s));\
	(a##3) = _mm_srli_epi32((a##3), (32-s));\
	(a##0) = tmp1_0 | (a##0);\
	(a##1) = tmp1_1 | (a##1);\
	(a##2) = tmp1_2 | (a##2);\
	(a##3) = tmp1_3 | (a##3);\
}


// Interlacing 4x SSE2
#define MD4_STEPS_NEW() { \
        MD4STEP_ROUND1_4      (a, b, c, d, w0, S11); \
	MD4STEP_ROUND1_4      (d, a, b, c, w1, S12); \
	MD4STEP_ROUND1_4      (c, d, a, b, w2, S13); \
	MD4STEP_ROUND1_4      (b, c, d, a, w3, S14); \
	MD4STEP_ROUND1_4      (a, b, c, d, w4, S11); \
	MD4STEP_ROUND1_4      (d, a, b, c, w5, S12); \
	MD4STEP_ROUND1_4      (c, d, a, b, w6, S13); \
	MD4STEP_ROUND1_4      (b, c, d, a, w7, S14); \
	MD4STEP_ROUND1_NULL_4 (a, b, c, d, S11); \
	MD4STEP_ROUND1_NULL_4 (d, a, b, c, S12); \
	MD4STEP_ROUND1_NULL_4 (c, d, a, b, S13); \
	MD4STEP_ROUND1_NULL_4 (b, c, d, a, S14); \
	MD4STEP_ROUND1_NULL_4 (a, b, c, d, S11); \
	MD4STEP_ROUND1_NULL_4 (d, a, b, c, S12); \
	MD4STEP_ROUND1_4      (c, d, a, b, w14, S13); \
	MD4STEP_ROUND1_NULL_4 (b, c, d, a, S14); \
	\
	MD4STEP_ROUND2_4      (a, b, c, d, w0, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, w4, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_NULL_4 (b, c, d, a, S24);\
	MD4STEP_ROUND2_4      (a, b, c, d, w1, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, w5, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_NULL_4 (b, c, d, a, S24);\
	MD4STEP_ROUND2_4      (a, b, c, d, w2, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, w6, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_4      (b, c, d, a, w14, S24);\
	MD4STEP_ROUND2_4      (a, b, c, d, w3, S21);\
	MD4STEP_ROUND2_4      (d, a, b, c, w7, S22);\
	MD4STEP_ROUND2_NULL_4 (c, d, a, b, S23);\
	MD4STEP_ROUND2_NULL_4 (b, c, d, a, S24);\
        \
        MD4STEP_ROUND3_4      (a, b, c, d, w0, S31);\
	MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32);\
	MD4STEP_ROUND3_4      (c, d, a, b, w4, S33);\
	MD4STEP_ROUND3_NULL_4 (b, c, d, a, S34);\
	MD4STEP_ROUND3_4      (a, b, c, d, w2, S31);\
	MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32); \
        MD4STEP_ROUND3_4      (c, d, a, b, w6, S33); \
        MD4STEP_ROUND3_4      (b, c, d, a, w14, S34);\
        MD4STEP_ROUND3_4      (a, b, c, d, w1, S31);\
	MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32);\
	MD4STEP_ROUND3_4      (c, d, a, b, w5, S33);\
	MD4STEP_ROUND3_NULL_4 (b, c, d, a, S34);\
        MD4STEP_ROUND3_4      (a, b, c, d, w3, S31);\
	MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32);\
	MD4STEP_ROUND3_4      (c, d, a, b, w7, S33);\
	MD4STEP_ROUND3_NULL_4 (b, c, d, a, S34);\
}
