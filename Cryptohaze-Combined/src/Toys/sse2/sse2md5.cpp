#include <stdio.h>

#include "sse2md5.h"
#include <emmintrin.h>
#include <stdint.h>
#include <string.h>

void print128(__m128i value) {
    int64_t *v64 = (int64_t*) &value;
    printf("%.16llx %.16llx\n", v64[1], v64[0]);
}
/*
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
*/

// Perform a bulk SSE2 MD5 operation on data.
// inputArrays: 16 * 12 bytes of input, [H0W0][H0W1][H0W2][H0W3][H1W0][H1W1]...
// This MUST be padded properly!
// length: The length of the input string
// outputHash: 16*12 bytes of output, same type of interleaved output.
// This will probably segfault badly if you don't do it this way...
void bulkSSE2MD5(unsigned char *inputArrays, unsigned char length, unsigned char *outputHash) {

    initializeVariables();

    // Support through len15 - 4 words.
    static __m128i w0_0, w0_1, w0_2;
    static __m128i w1_0, w1_1, w1_2;
    static __m128i w2_0, w2_1, w2_2;
    static __m128i w3_0, w3_1, w3_2;
    // For length.
    __m128i w14;

    static __m128i a, b, c, d, tmp1, tmp2;
    static __m128i a2, b2, c2, d2, tmp1_2, tmp2_2;
    static __m128i a3, b3, c3, d3, tmp1_3, tmp2_3;

    __m128i mOne = _mm_set1_epi32(0xFFFFFFFF);
    
    // Load all the initial data with data.
    w0_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  0], *(uint32_t *)&inputArrays[16 *  1],
                         *(uint32_t *)&inputArrays[16 *  2], *(uint32_t *)&inputArrays[16 *  3]);
    w0_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  4], *(uint32_t *)&inputArrays[16 *  5],
                         *(uint32_t *)&inputArrays[16 *  6], *(uint32_t *)&inputArrays[16 *  7]);
    w0_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  8], *(uint32_t *)&inputArrays[16 *  9],
                         *(uint32_t *)&inputArrays[16 * 10], *(uint32_t *)&inputArrays[16 * 11]);

    w1_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  0 + 4], *(uint32_t *)&inputArrays[16 *  1 + 4],
                         *(uint32_t *)&inputArrays[16 *  2 + 4], *(uint32_t *)&inputArrays[16 *  3 + 4]);
    w1_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  4 + 4], *(uint32_t *)&inputArrays[16 *  5 + 4],
                         *(uint32_t *)&inputArrays[16 *  6 + 4], *(uint32_t *)&inputArrays[16 *  7 + 4]);
    w1_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  8 + 4], *(uint32_t *)&inputArrays[16 *  9 + 4],
                         *(uint32_t *)&inputArrays[16 * 10 + 4], *(uint32_t *)&inputArrays[16 * 11 + 4]);

    w2_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  0 + 8], *(uint32_t *)&inputArrays[16 *  1 + 8],
                         *(uint32_t *)&inputArrays[16 *  2 + 8], *(uint32_t *)&inputArrays[16 *  3 + 8]);
    w2_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  4 + 8], *(uint32_t *)&inputArrays[16 *  5 + 8],
                         *(uint32_t *)&inputArrays[16 *  6 + 8], *(uint32_t *)&inputArrays[16 *  7 + 8]);
    w2_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  8 + 8], *(uint32_t *)&inputArrays[16 *  9 + 8],
                         *(uint32_t *)&inputArrays[16 * 10 + 8], *(uint32_t *)&inputArrays[16 * 11 + 8]);

    w3_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  0 + 12], *(uint32_t *)&inputArrays[16 *  1 + 12],
                         *(uint32_t *)&inputArrays[16 *  2 + 12], *(uint32_t *)&inputArrays[16 *  3 + 12]);
    w3_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  4 + 12], *(uint32_t *)&inputArrays[16 *  5 + 12],
                         *(uint32_t *)&inputArrays[16 *  6 + 12], *(uint32_t *)&inputArrays[16 *  7 + 12]);
    w3_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[16 *  8 + 12], *(uint32_t *)&inputArrays[16 *  9 + 12],
                         *(uint32_t *)&inputArrays[16 * 10 + 12], *(uint32_t *)&inputArrays[16 * 11 + 12]);

    // Set the length (0 for now)
    w14 = _mm_set1_epi32(length*8);

    // Load the a/b/c/d vars with the initial values.
    a = mCa; a2 = mCa; a3 = mCa;
    b = mCb; b2 = mCb; b3 = mCb;
    c = mCc; c2 = mCc; c3 = mCc;
    d = mCd; d2 = mCd; d3 = mCd;

    MD5STEP_ROUND1(F, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3,  mAC1, w0_0, w0_1, w0_2,  S11);\
    MD5STEP_ROUND1(F, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3,  mAC2, w1_0, w1_1, w1_2,  S12);\
    MD5STEP_ROUND1(F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3,  mAC3, w2_0, w2_1, w2_2,  S13);\
    MD5STEP_ROUND1(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3,  mAC4, w3_0, w3_1, w3_2,  S14);\
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

    MD5STEP_ROUND2     (G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC17, w1_0, w1_1, w1_2,  S21);\
    MD5STEP_ROUND2_NULL(G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC18,   S22);\
    MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC19,   S23);\
    MD5STEP_ROUND2     (G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC20, w0_0, w0_1, w0_2,  S24);\
    MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC21,   S21);\
    MD5STEP_ROUND2_NULL(G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC22,   S22);\
    MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC23,   S23);\
    MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC24,   S24);\
    MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC25,   S21);\
    MD5STEP_ROUND2     (G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC26, w14, w14, w14, S22);\
    MD5STEP_ROUND2     (G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC27, w3_0, w3_1, w3_2,  S23);\
    MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC28,   S24);\
    MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC29,   S21);\
    MD5STEP_ROUND2     (G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC30, w2_0, w2_1, w2_2,  S22);\
    MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC31,   S23);\
    MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC32,   S24);\

    MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC33,   S31);\
    MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC34,   S32);\
    MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC35,   S33);\
    MD5STEP_ROUND3     (H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC36, w14, w14, w14, S34);\
    MD5STEP_ROUND3     (H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC37, w1_0, w1_1, w1_2,  S31);\
    MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC38,   S32);\
    MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC39,   S33);\
    MD5STEP_ROUND3_NULL(H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC40,   S34);\
    MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC41,   S31);\
    MD5STEP_ROUND3     (H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC42, w0_0, w0_1, w0_2,  S32);\
    MD5STEP_ROUND3     (H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC43, w3_0, w3_1, w3_2,  S33);\
    MD5STEP_ROUND3_NULL(H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC44,   S34);\
    MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC45,   S31);\
    MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC46,   S32);\
    MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC47,   S33);\
    MD5STEP_ROUND3     (H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC48, w2_0, w2_1, w2_2, S34);\
    \
    MD5STEP_ROUND4     (I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC49, w0_0, w0_1, w0_2,  S41);\
    MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC50,   S42);\
    MD5STEP_ROUND4     (I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC51, w14, w14, w14,  S43);\
    MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC52,   S44);\
    MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC53,   S41);\
    MD5STEP_ROUND4     (I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC54, w3_0, w3_1, w3_2,  S42);\
    MD5STEP_ROUND4_NULL(I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC55,   S43);\
    MD5STEP_ROUND4     (I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC56, w1_0, w1_1, w1_2,  S44);\
    MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC57,   S41);\
    MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC58,   S42);\
    MD5STEP_ROUND4_NULL(I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC59,   S43);\
    MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC60,   S44);\
    MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC61,   S41);\
    MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC62,   S42);\
    MD5STEP_ROUND4     (I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC63, w2_0, w2_1, w2_2,  S43);\
    MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC64,   S44);\


    // Add the initial values in again.
    a = _mm_add_epi32(a,mCa); a2 = _mm_add_epi32(a2,mCa); a3 = _mm_add_epi32(a3,mCa);
    b = _mm_add_epi32(b,mCb); b2 = _mm_add_epi32(b2,mCb); b3 = _mm_add_epi32(b3,mCb);
    c = _mm_add_epi32(c,mCc); c2 = _mm_add_epi32(c2,mCc); c3 = _mm_add_epi32(c3,mCc);
    d = _mm_add_epi32(d,mCd); d2 = _mm_add_epi32(d2,mCd); d3 = _mm_add_epi32(d3,mCd);
/*
    printf("Output hashes: a: \n");
    print128(a); print128(a2); print128(a3);
    printf("Output hashes: b: \n");
    print128(b); print128(b2); print128(b3);
    printf("Output hashes: c: \n");
    print128(c); print128(c2); print128(c3);
    printf("Output hashes: d: \n");
    print128(d); print128(d2); print128(d3);
*/

    // Unpack the hashes back out.  These are reversed in the registers!
    uint32_t *a1_32 = (uint32_t*) &a;
    uint32_t *a2_32 = (uint32_t*) &a2;
    uint32_t *a3_32 = (uint32_t*) &a3;

    uint32_t *b1_32 = (uint32_t*) &b;
    uint32_t *b2_32 = (uint32_t*) &b2;
    uint32_t *b3_32 = (uint32_t*) &b3;

    uint32_t *c1_32 = (uint32_t*) &c;
    uint32_t *c2_32 = (uint32_t*) &c2;
    uint32_t *c3_32 = (uint32_t*) &c3;

    uint32_t *d1_32 = (uint32_t*) &d;
    uint32_t *d2_32 = (uint32_t*) &d2;
    uint32_t *d3_32 = (uint32_t*) &d3;

    uint32_t *outputHash32 = (uint32_t *)outputHash;
/*
    printf("a1_32[0]: %08x\n", a1_32[0]);
    printf("a1_32[1]: %08x\n", a1_32[1]);
    printf("a1_32[2]: %08x\n", a1_32[2]);
    printf("a1_32[3]: %08x\n", a1_32[3]);
*/
    // Output hashes - note that the order is reversed in registers.
    outputHash32[4 *  0] = a1_32[3];
    outputHash32[4 *  1] = a1_32[2];
    outputHash32[4 *  2] = a1_32[1];
    outputHash32[4 *  3] = a1_32[0];

    outputHash32[4 *  4] = a2_32[3];
    outputHash32[4 *  5] = a2_32[2];
    outputHash32[4 *  6] = a2_32[1];
    outputHash32[4 *  7] = a2_32[0];

    outputHash32[4 *  8] = a3_32[3];
    outputHash32[4 *  9] = a3_32[2];
    outputHash32[4 * 10] = a3_32[1];
    outputHash32[4 * 11] = a3_32[0];

    outputHash32[4 *  0 + 1] = b1_32[3];
    outputHash32[4 *  1 + 1] = b1_32[2];
    outputHash32[4 *  2 + 1] = b1_32[1];
    outputHash32[4 *  3 + 1] = b1_32[0];

    outputHash32[4 *  4 + 1] = b2_32[3];
    outputHash32[4 *  5 + 1] = b2_32[2];
    outputHash32[4 *  6 + 1] = b2_32[1];
    outputHash32[4 *  7 + 1] = b2_32[0];

    outputHash32[4 *  8 + 1] = b3_32[3];
    outputHash32[4 *  9 + 1] = b3_32[2];
    outputHash32[4 * 10 + 1] = b3_32[1];
    outputHash32[4 * 11 + 1] = b3_32[0];

    outputHash32[4 *  0 + 2] = c1_32[3];
    outputHash32[4 *  1 + 2] = c1_32[2];
    outputHash32[4 *  2 + 2] = c1_32[1];
    outputHash32[4 *  3 + 2] = c1_32[0];

    outputHash32[4 *  4 + 2] = c2_32[3];
    outputHash32[4 *  5 + 2] = c2_32[2];
    outputHash32[4 *  6 + 2] = c2_32[1];
    outputHash32[4 *  7 + 2] = c2_32[0];

    outputHash32[4 *  8 + 2] = c3_32[3];
    outputHash32[4 *  9 + 2] = c3_32[2];
    outputHash32[4 * 10 + 2] = c3_32[1];
    outputHash32[4 * 11 + 2] = c3_32[0];

    outputHash32[4 *  0 + 3] = d1_32[3];
    outputHash32[4 *  1 + 3] = d1_32[2];
    outputHash32[4 *  2 + 3] = d1_32[1];
    outputHash32[4 *  3 + 3] = d1_32[0];

    outputHash32[4 *  4 + 3] = d2_32[3];
    outputHash32[4 *  5 + 3] = d2_32[2];
    outputHash32[4 *  6 + 3] = d2_32[1];
    outputHash32[4 *  7 + 3] = d2_32[0];

    outputHash32[4 *  8 + 3] = d3_32[3];
    outputHash32[4 *  9 + 3] = d3_32[2];
    outputHash32[4 * 10 + 3] = d3_32[1];
    outputHash32[4 * 11 + 3] = d3_32[0];
}


int main() {

    unsigned char inputData[16 * 12];
    unsigned char outputHashes[16 * 12];

    int i, j;

    uint32_t steps;

    uint32_t result;

    // Clear the input data
    memset(inputData, 0, 16 * 12);

    for (steps = 0; steps < 10000000; steps++) {
        // Set up the SSE values - first 16 letters of the alphabet
        memset(inputData, 0, 16 * 12);
        for (i = 0; i < 12; i++) {
            for (j = 0; j < 8; j++) {
                inputData[16 * i + j] = 0x41 + i + (steps % 16); // Start from 'A';
            }
            inputData[16 * i + 8] = 0x80;
        }
        bulkSSE2MD5(inputData, 8, outputHashes);
        for (j = 0; j < 12; j++) {
            result += outputHashes[j * 16];
        }
    }

    printf("result: %lu\n", result);
    memset(inputData, 0, 16 * 12);
    // Set up the SSE values - first 16 letters of the alphabet
    for (i = 0; i < 12; i++) {
        for (j = 0; j < 8; j++) {
            inputData[16 * i + j] = 0x41 + i; // Start from 'A';
        }
        inputData[16 * i + 8] = 0x80;
    }
    bulkSSE2MD5(inputData, 8, outputHashes);

    printf("Hashing done.\n");
    
    for (i = 0; i < 12; i++) {
        printf("%s : ", &inputData[16 * i]);
        for (j = 0; j < 16; j++) {
            printf("%02x", outputHashes[i * 16 + j]);
        }
        printf("\n");
    }
}

/*
int main() {

    printf("SSE MD5 test...\n");

    initializeVariables();

    int length = 0;

    printf("Variables init'd.\n");

    // Support through len15 - 4 words.
    __m128i w0_0, w0_1, w0_2;
    __m128i w1_0, w1_1, w1_2;
    __m128i w2_0, w2_1, w2_2;
    __m128i w3_0, w3_1, w3_2;
    // For length.
    __m128i w14;

    __m128i a, b, c, d, tmp1, tmp2;
    __m128i a2, b2, c2, d2, tmp1_2, tmp2_2;
    __m128i a3, b3, c3, d3, tmp1_3, tmp2_3;

    __m128i mOne = _mm_set1_epi32(0xFFFFFFFF);

    a = mCa;
    a2 = mCa;
    a3 = mCa;

    b = mCb;
    b2 = mCb;
    b3 = mCb;

    c = mCc;
    c2 = mCc;
    c3 = mCc;

    d = mCd;
    d2 = mCd;
    d3 = mCd;

    printf("A values: \n");
    print128(a);
    printf("B values: \n");
    print128(b);
    printf("C values: \n");
    print128(c);
    printf("D values: \n");
    print128(d);
    
    // Clear all the init registers, set the padding.
    w0_0 = _mm_set1_epi32(0x80);
    w0_1 = _mm_set1_epi32(0x80);
    w0_2 = _mm_set1_epi32(0x80);
    w1_0 = _mm_set1_epi32(0);
    w1_1 = _mm_set1_epi32(0);
    w1_2 = _mm_set1_epi32(0);
    w2_0 = _mm_set1_epi32(0);
    w2_1 = _mm_set1_epi32(0);
    w2_2 = _mm_set1_epi32(0);
    w3_0 = _mm_set1_epi32(0);
    w3_1 = _mm_set1_epi32(0);
    w3_2 = _mm_set1_epi32(0);
    // Set the length (0 for now)
    w14 = _mm_set1_epi32(length*8);

    printf("Initial w0: \n");
    print128(w0_0); print128(w0_1); print128(w0_2);

    printf("Initial a: \n");
    print128(a); print128(a2); print128(a3);
    printf("Initial b: \n");
    print128(b); print128(b2); print128(b3);
    printf("Initial c: \n");
    print128(c); print128(c2); print128(c3);
    printf("Initial d: \n");
    print128(d); print128(d2); print128(d3);


    MD5STEP_ROUND1(F, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3,  mAC1, w0_0, w0_1, w0_2,  S11);\
    MD5STEP_ROUND1(F, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3,  mAC2, w1_0, w1_1, w1_2,  S12);\
    MD5STEP_ROUND1(F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3,  mAC3, w2_0, w2_1, w2_2,  S13);\
    MD5STEP_ROUND1(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3,  mAC4, w3_0, w3_1, w3_2,  S14);\
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

    MD5STEP_ROUND2     (G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC17, w1_0, w1_1, w1_2,  S21);\
    MD5STEP_ROUND2_NULL(G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC18,   S22);\
    MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC19,   S23);\
    MD5STEP_ROUND2     (G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC20, w0_0, w0_1, w0_2,  S24);\
    MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC21,   S21);\
    MD5STEP_ROUND2_NULL(G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC22,   S22);\
    MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC23,   S23);\
    MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC24,   S24);\
    MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC25,   S21);\
    MD5STEP_ROUND2     (G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC26, w14, w14, w14, S22);\
    MD5STEP_ROUND2     (G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC27, w3_0, w3_1, w3_2,  S23);\
    MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC28,   S24);\
    MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC29,   S21);\
    MD5STEP_ROUND2     (G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC30, w2_0, w2_1, w2_2,  S22);\
    MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC31,   S23);\
    MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC32,   S24);\

    MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC33,   S31);\
    MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC34,   S32);\
    MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC35,   S33);\
    MD5STEP_ROUND3     (H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC36, w14, w14, w14, S34);\
    MD5STEP_ROUND3     (H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC37, w1_0, w1_1, w1_2,  S31);\
    MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC38,   S32);\
    MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC39,   S33);\
    MD5STEP_ROUND3_NULL(H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC40,   S34);\
    MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC41,   S31);\
    MD5STEP_ROUND3     (H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC42, w0_0, w0_1, w0_2,  S32);\
    MD5STEP_ROUND3     (H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC43, w3_0, w3_1, w3_2,  S33);\
    MD5STEP_ROUND3_NULL(H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC44,   S34);\
    MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC45,   S31);\
    MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC46,   S32);\
    MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC47,   S33);\
    MD5STEP_ROUND3     (H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC48, w2_0, w2_1, w2_2, S34);\
    \
    MD5STEP_ROUND4     (I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC49, w0_0, w0_1, w0_2,  S41);\
    MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC50,   S42);\
    MD5STEP_ROUND4     (I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC51, w14, w14, w14,  S43);\
    MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC52,   S44);\
    MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC53,   S41);\
    MD5STEP_ROUND4     (I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC54, w3_0, w3_1, w3_2,  S42);\
    MD5STEP_ROUND4_NULL(I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC55,   S43);\
    MD5STEP_ROUND4     (I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC56, w1_0, w1_1, w1_2,  S44);\
    MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC57,   S41);\
    MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC58,   S42);\
    MD5STEP_ROUND4_NULL(I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC59,   S43);\
    MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC60,   S44);\
    MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC61,   S41);\
    MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC62,   S42);\
    MD5STEP_ROUND4     (I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC63, w2_0, w2_1, w2_2,  S43);\
    MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC64,   S44);\


    a = _mm_add_epi32(a,mCa);
    a2 = _mm_add_epi32(a2,mCa);
    a3 = _mm_add_epi32(a3,mCa);

    b = _mm_add_epi32(b,mCb);
    b2 = _mm_add_epi32(b2,mCb);
    b3 = _mm_add_epi32(b3,mCb);

    c = _mm_add_epi32(c,mCc);
    c2 = _mm_add_epi32(c2,mCc);
    c3 = _mm_add_epi32(c3,mCc);

    d = _mm_add_epi32(d,mCd);
    d2 = _mm_add_epi32(d2,mCd);
    d3 = _mm_add_epi32(d3,mCd);


    printf("Value post-loaded: %08x\n", a);
    printf("All values: \n");
    print128(a);
    print128(b);
    print128(c);
    print128(d);

    int32_t *v32 = (int32_t*) &a;

    printf("Val: %08x\n", v32[0]);

    printf("Hash: %02x%02x%02x%02x\n", v32[0] & 0xff, v32[0] >> 8 & 0xff, v32[0] >> 16 & 0xff, v32[0] >> 24 & 0xff);
}
*/