#include <stdio.h>

#include "sse2md4.h"
#include <emmintrin.h>
#include <stdint.h>
#include <string.h>


void print128(__m128i value) {
    int32_t *v32 = (int32_t*) &value;
    printf("%08x %08x %08x %08x\n", v32[3], v32[2], v32[1], v32[0]);
}


// Perform a bulk SSE2 MD5 operation on data.
// inputArrays: 32 * 16 bytes of input, [H0W0][H0W1][H0W2][H0W3][H1W0][H1W1]...
// This MUST be padded properly!
// length: The length of the input string
// outputHash: 16*16 bytes of output, same type of interleaved output.
// This will probably segfault badly if you don't do it this way...
void bulkSSE2MD4(unsigned char *inputArrays, unsigned char length, unsigned char *outputHash) {

    // Support through len15 - 8 words.
    static __m128i w0_0, w0_1, w0_2, w0_3;
    static __m128i w1_0, w1_1, w1_2, w1_3;
    static __m128i w2_0, w2_1, w2_2, w2_3;
    static __m128i w3_0, w3_1, w3_2, w3_3;
    static __m128i w4_0, w4_1, w4_2, w4_3;
    static __m128i w5_0, w5_1, w5_2, w5_3;
    static __m128i w6_0, w6_1, w6_2, w6_3;
    static __m128i w7_0, w7_1, w7_2, w7_3;

    // For length.
    __m128i w14, w14_0, w14_1, w14_2, w14_3;

    // Seeds
    __m128i AC, AC2;

    static __m128i a0, b0, c0, d0, tmp1_0, tmp2_0;
    static __m128i a1, b1, c1, d1, tmp1_1, tmp2_1;
    static __m128i a2, b2, c2, d2, tmp1_2, tmp2_2;
    static __m128i a3, b3, c3, d3, tmp1_3, tmp2_3;
    static __m128i mCa, mCb, mCc, mCd;

    // Initial values for a,b,c,d
    mCa = _mm_set1_epi32(Ca);
    mCb = _mm_set1_epi32(Cb);
    mCc = _mm_set1_epi32(Cc);
    mCd = _mm_set1_epi32(Cd);

    // Constants for MD4
    AC = _mm_set1_epi32(0x5a827999);
    AC2 = _mm_set1_epi32(0x6ed9eba1);

    // Load all the initial data with data.
    w0_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  0], *(uint32_t *)&inputArrays[32 *  1],
                         *(uint32_t *)&inputArrays[32 *  2], *(uint32_t *)&inputArrays[32 *  3]);
    w0_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  4], *(uint32_t *)&inputArrays[32 *  5],
                         *(uint32_t *)&inputArrays[32 *  6], *(uint32_t *)&inputArrays[32 *  7]);
    w0_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  8], *(uint32_t *)&inputArrays[32 *  9],
                         *(uint32_t *)&inputArrays[32 * 10], *(uint32_t *)&inputArrays[32 * 11]);
    w0_3 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 * 12], *(uint32_t *)&inputArrays[32 * 13],
                         *(uint32_t *)&inputArrays[32 * 14], *(uint32_t *)&inputArrays[32 * 15]);

    w1_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  0 + 4], *(uint32_t *)&inputArrays[32 *  1 + 4],
                         *(uint32_t *)&inputArrays[32 *  2 + 4], *(uint32_t *)&inputArrays[32 *  3 + 4]);
    w1_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  4 + 4], *(uint32_t *)&inputArrays[32 *  5 + 4],
                         *(uint32_t *)&inputArrays[32 *  6 + 4], *(uint32_t *)&inputArrays[32 *  7 + 4]);
    w1_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  8 + 4], *(uint32_t *)&inputArrays[32 *  9 + 4],
                         *(uint32_t *)&inputArrays[32 * 10 + 4], *(uint32_t *)&inputArrays[32 * 11 + 4]);
    w1_3 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 * 12 + 4], *(uint32_t *)&inputArrays[32 * 13 + 4],
                         *(uint32_t *)&inputArrays[32 * 14 + 4], *(uint32_t *)&inputArrays[32 * 15 + 4]);

    w2_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  0 + 8], *(uint32_t *)&inputArrays[32 *  1 + 8],
                         *(uint32_t *)&inputArrays[32 *  2 + 8], *(uint32_t *)&inputArrays[32 *  3 + 8]);
    w2_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  4 + 8], *(uint32_t *)&inputArrays[32 *  5 + 8],
                         *(uint32_t *)&inputArrays[32 *  6 + 8], *(uint32_t *)&inputArrays[32 *  7 + 8]);
    w2_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  8 + 8], *(uint32_t *)&inputArrays[32 *  9 + 8],
                         *(uint32_t *)&inputArrays[32 * 10 + 8], *(uint32_t *)&inputArrays[32 * 11 + 8]);
    w2_3 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 * 12 + 8], *(uint32_t *)&inputArrays[32 * 13 + 8],
                         *(uint32_t *)&inputArrays[32 * 14 + 8], *(uint32_t *)&inputArrays[32 * 15 + 8]);

    w3_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  0 + 12], *(uint32_t *)&inputArrays[32 *  1 + 12],
                         *(uint32_t *)&inputArrays[32 *  2 + 12], *(uint32_t *)&inputArrays[32 *  3 + 12]);
    w3_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  4 + 12], *(uint32_t *)&inputArrays[32 *  5 + 12],
                         *(uint32_t *)&inputArrays[32 *  6 + 12], *(uint32_t *)&inputArrays[32 *  7 + 12]);
    w3_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  8 + 12], *(uint32_t *)&inputArrays[32 *  9 + 12],
                         *(uint32_t *)&inputArrays[32 * 10 + 12], *(uint32_t *)&inputArrays[32 * 11 + 12]);
    w3_3 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 * 12 + 12], *(uint32_t *)&inputArrays[32 * 13 + 12],
                         *(uint32_t *)&inputArrays[32 * 14 + 12], *(uint32_t *)&inputArrays[32 * 15 + 12]);

    w4_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  0 + 16], *(uint32_t *)&inputArrays[32 *  1 + 16],
                         *(uint32_t *)&inputArrays[32 *  2 + 16], *(uint32_t *)&inputArrays[32 *  3 + 16]);
    w4_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  4 + 16], *(uint32_t *)&inputArrays[32 *  5 + 16],
                         *(uint32_t *)&inputArrays[32 *  6 + 16], *(uint32_t *)&inputArrays[32 *  7 + 16]);
    w4_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  8 + 16], *(uint32_t *)&inputArrays[32 *  9 + 16],
                         *(uint32_t *)&inputArrays[32 * 10 + 16], *(uint32_t *)&inputArrays[32 * 11 + 16]);
    w4_3 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 * 12 + 16], *(uint32_t *)&inputArrays[32 * 13 + 16],
                         *(uint32_t *)&inputArrays[32 * 14 + 16], *(uint32_t *)&inputArrays[32 * 15 + 16]);

    w5_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  0 + 20], *(uint32_t *)&inputArrays[32 *  1 + 20],
                         *(uint32_t *)&inputArrays[32 *  2 + 20], *(uint32_t *)&inputArrays[32 *  3 + 20]);
    w5_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  4 + 20], *(uint32_t *)&inputArrays[32 *  5 + 20],
                         *(uint32_t *)&inputArrays[32 *  6 + 20], *(uint32_t *)&inputArrays[32 *  7 + 20]);
    w5_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  8 + 20], *(uint32_t *)&inputArrays[32 *  9 + 20],
                         *(uint32_t *)&inputArrays[32 * 10 + 20], *(uint32_t *)&inputArrays[32 * 11 + 20]);
    w5_3 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 * 12 + 20], *(uint32_t *)&inputArrays[32 * 13 + 20],
                         *(uint32_t *)&inputArrays[32 * 14 + 20], *(uint32_t *)&inputArrays[32 * 15 + 20]);

    w6_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  0 + 24], *(uint32_t *)&inputArrays[32 *  1 + 24],
                         *(uint32_t *)&inputArrays[32 *  2 + 24], *(uint32_t *)&inputArrays[32 *  3 + 24]);
    w6_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  4 + 24], *(uint32_t *)&inputArrays[32 *  5 + 24],
                         *(uint32_t *)&inputArrays[32 *  6 + 24], *(uint32_t *)&inputArrays[32 *  7 + 24]);
    w6_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  8 + 24], *(uint32_t *)&inputArrays[32 *  9 + 24],
                         *(uint32_t *)&inputArrays[32 * 10 + 24], *(uint32_t *)&inputArrays[32 * 11 + 24]);
    w6_3 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 * 12 + 24], *(uint32_t *)&inputArrays[32 * 13 + 24],
                         *(uint32_t *)&inputArrays[32 * 14 + 24], *(uint32_t *)&inputArrays[32 * 15 + 24]);

    w7_0 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  0 + 28], *(uint32_t *)&inputArrays[32 *  1 + 28],
                         *(uint32_t *)&inputArrays[32 *  2 + 28], *(uint32_t *)&inputArrays[32 *  3 + 28]);
    w7_1 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  4 + 28], *(uint32_t *)&inputArrays[32 *  5 + 28],
                         *(uint32_t *)&inputArrays[32 *  6 + 28], *(uint32_t *)&inputArrays[32 *  7 + 28]);
    w7_2 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 *  8 + 28], *(uint32_t *)&inputArrays[32 *  9 + 28],
                         *(uint32_t *)&inputArrays[32 * 10 + 28], *(uint32_t *)&inputArrays[32 * 11 + 28]);
    w7_3 = _mm_set_epi32(*(uint32_t *)&inputArrays[32 * 12 + 28], *(uint32_t *)&inputArrays[32 * 13 + 28],
                         *(uint32_t *)&inputArrays[32 * 14 + 28], *(uint32_t *)&inputArrays[32 * 15 + 28]);

    // Set the length (0 for now)
    w14 = w14_0 = w14_1 = w14_2 = w14_3 = _mm_set1_epi32(length*8);

    // Load the a/b/c/d vars with the initial values.
    a0 = a1 = a2 = a3 = mCa;
    b0 = b1 = b2 = b3 = mCb;
    c0 = c1 = c2 = c3 = mCc;
    d0 = d1 = d2 = d3 = mCd;

    MD4_STEPS_NEW();

    // Add the initial values in again.
    a0 = _mm_add_epi32(a0,mCa); a1 = _mm_add_epi32(a1,mCa); a2 = _mm_add_epi32(a2,mCa); a3 = _mm_add_epi32(a3,mCa);
    b0 = _mm_add_epi32(b0,mCb); b1 = _mm_add_epi32(b1,mCb); b2 = _mm_add_epi32(b2,mCb); b3 = _mm_add_epi32(b3,mCb);
    c0 = _mm_add_epi32(c0,mCc); c1 = _mm_add_epi32(c1,mCc); c2 = _mm_add_epi32(c2,mCc); c3 = _mm_add_epi32(c3,mCc);
    d0 = _mm_add_epi32(d0,mCd); d1 = _mm_add_epi32(d1,mCd); d2 = _mm_add_epi32(d2,mCd); d3 = _mm_add_epi32(d3,mCd);
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
    uint32_t *a1_32 = (uint32_t*) &a0;
    uint32_t *a2_32 = (uint32_t*) &a1;
    uint32_t *a3_32 = (uint32_t*) &a2;
    uint32_t *a4_32 = (uint32_t*) &a3;

    uint32_t *b1_32 = (uint32_t*) &b0;
    uint32_t *b2_32 = (uint32_t*) &b1;
    uint32_t *b3_32 = (uint32_t*) &b2;
    uint32_t *b4_32 = (uint32_t*) &b3;

    uint32_t *c1_32 = (uint32_t*) &c0;
    uint32_t *c2_32 = (uint32_t*) &c1;
    uint32_t *c3_32 = (uint32_t*) &c2;
    uint32_t *c4_32 = (uint32_t*) &c3;

    uint32_t *d1_32 = (uint32_t*) &d0;
    uint32_t *d2_32 = (uint32_t*) &d1;
    uint32_t *d3_32 = (uint32_t*) &d2;
    uint32_t *d4_32 = (uint32_t*) &d3;

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

    outputHash32[4 * 12] = a4_32[3];
    outputHash32[4 * 13] = a4_32[2];
    outputHash32[4 * 14] = a4_32[1];
    outputHash32[4 * 15] = a4_32[0];

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

    outputHash32[4 * 12 + 1] = b4_32[3];
    outputHash32[4 * 13 + 1] = b4_32[2];
    outputHash32[4 * 14 + 1] = b4_32[1];
    outputHash32[4 * 15 + 1] = b4_32[0];

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

    outputHash32[4 * 12 + 2] = c4_32[3];
    outputHash32[4 * 13 + 2] = c4_32[2];
    outputHash32[4 * 14 + 2] = c4_32[1];
    outputHash32[4 * 15 + 2] = c4_32[0];

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

    outputHash32[4 * 12 + 3] = d4_32[3];
    outputHash32[4 * 13 + 3] = d4_32[2];
    outputHash32[4 * 14 + 3] = d4_32[1];
    outputHash32[4 * 15 + 3] = d4_32[0];
}


int main() {

    unsigned char inputData[32 * 16];
    unsigned char outputHashes[16 * 16];

    int i, j;

    uint32_t steps;

    uint32_t result;

    // Clear the input data
    memset(inputData, 0, 32 * 16);

    for (steps = 0; steps < 10; steps++) {
        // Set up the SSE values - first 16 letters of the alphabet
        memset(inputData, 0, 16 * 12);
        for (i = 0; i < 16; i++) {
            for (j = 0; j < 8; j++) {
                inputData[32 * i + j*2] = 0x41 + i; // Start from 'A';
            }
            inputData[32 * i + 16] = 0x80;
        }
        bulkSSE2MD4(inputData, 16, outputHashes);
        for (j = 0; j < 12; j++) {
            result += outputHashes[j * 16];
        }
    }

    printf("result: %lu\n", result);

    memset(inputData, 0, 32 * 16);
    // Set up the SSE values - first 16 letters of the alphabet
    for (i = 0; i < 16; i++) {
        for (j = 0; j < 8; j++) {
            inputData[32 * i + j*2] = 0x41 + i; // Start from 'A';
        }
        inputData[32 * i + 16] = 0x80;
    }
    bulkSSE2MD4(inputData, 16, outputHashes);

    printf("Hashing done.\n");

    for (i = 0; i < 16; i++) {
        printf("%s... : ", &inputData[32 * i]);
        for (j = 0; j < 16; j++) {
            printf("%02x", outputHashes[i * 16 + j]);
        }
        printf("\n");
    }

    printf("\nA... : a6c92fbb3f0d0c2fb969262e8c1feb3b\n");
    printf("P... : 39967e28e0f269916a94bec13aa70f30\n");
}
