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

#include "MFN_CPU_host/MFNHashTypePlainCPU_MD5.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_CPU_host/MFNSSE2_MD5.h"
#include "MFN_CPU_host/MFN_CPU_inc.h"
#include "CH_HashFiles/CHHashFileV.h"
#include "MFN_Common/MFNDisplay.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "MFN_Common/MFNNetworkClient.h"
#include "MFN_Common/MFNCommandLineData.h"

extern MFNClassFactory MultiforcerGlobalClassFactory;


extern struct global_commands global_interface;

// Prints a 128 bit value.
static void print128(__m128i value) {
    int64_t *v64 = (int64_t*) &value;
    printf("%.16llx %.16llx\n", v64[1], v64[0]);
}

static void printPassword128(__m128i &b0, __m128i &b1, __m128i &b2, __m128i &b3, int passLen) {
    uint32_t *b0_32 = (uint32_t *)&b0;
    uint32_t *b1_32 = (uint32_t *)&b1;
    uint32_t *b2_32 = (uint32_t *)&b2;
    uint32_t *b3_32 = (uint32_t *)&b3;
    int pass;

    for (pass = 0; pass < 4; pass++) {
        printf("Pass %d: '", pass);
        if (passLen > 0) printf("%c", (b0_32[3 - pass] >> 0) & 0xff);
        if (passLen > 1) printf("%c", (b0_32[3 - pass] >> 8) & 0xff);
        if (passLen > 2) printf("%c", (b0_32[3 - pass] >> 16) & 0xff);
        if (passLen > 3) printf("%c", (b0_32[3 - pass] >> 24) & 0xff);

        if (passLen > 4) printf("%c", (b1_32[3 - pass] >> 0) & 0xff);
        if (passLen > 5) printf("%c", (b1_32[3 - pass] >> 8) & 0xff);
        if (passLen > 6) printf("%c", (b1_32[3 - pass] >> 16) & 0xff);
        if (passLen > 7) printf("%c", (b1_32[3 - pass] >> 24) & 0xff);

        if (passLen > 8) printf("%c", (b2_32[3 - pass] >> 0) & 0xff);
        if (passLen > 9) printf("%c", (b2_32[3 - pass] >> 8) & 0xff);
        if (passLen > 10) printf("%c", (b2_32[3 - pass] >> 16) & 0xff);
        if (passLen > 11) printf("%c", (b2_32[3 - pass] >> 24) & 0xff);

        if (passLen > 12) printf("%c", (b3_32[3 - pass] >> 0) & 0xff);
        if (passLen > 13) printf("%c", (b3_32[3 - pass] >> 8) & 0xff);
        if (passLen > 14) printf("%c", (b3_32[3 - pass] >> 16) & 0xff);
        if (passLen > 15) printf("%c", (b3_32[3 - pass] >> 24) & 0xff);

        printf("'\n");
    }
}

// Deal with hashes that are little endian 32-bits.
static bool hashMD5SortPredicate(const std::vector<uint8_t> &h1, const std::vector<uint8_t> &h2) {
    long int i, j;

    for (i = 0; i < (h1.size() / 4); i++) {
        for (j = 3; j >= 0; j--) {
            if (h1[(i * 4) + j] == h2[(i * 4) + j]) {
                continue;
            } else if (h1[(i * 4) + j] > h2[(i * 4) + j]) {
                return 0;
            } else if (h1[(i * 4) + j] < h2[(i * 4) + j]) {
                return 1;
            }
        }
    }
    // Exactly equal = return 0.
    return 0;
}


// How many SSE vectors each kernel deals with.
#define SSE_KERNEL_VECTOR_WIDTH 12

#define MD5ROTATE_RIGHT(x, n) (((x) >> (n)) | ((x) << (32-(n))))

#define REV_II(a,b,c,d,data,shift,constant) \
    a = MD5ROTATE_RIGHT((a - b), shift) - data - constant - (c ^ (b | (~d)));

#define MD5I(x, y, z) ((y) ^ ((x) | (~z)))
#define MD5ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))
#define MD5II(a, b, c, d, x, s, ac) { \
 (a) += MD5I ((b), (c), (d)) + (x) + (uint32_t)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }

#define MD5S41 6
#define MD5S42 10
#define MD5S43 15
#define MD5S44 21

MFNHashTypePlainCPU_MD5::MFNHashTypePlainCPU_MD5() :  MFNHashTypePlainCPU(16) {
    trace_printf("MFNHashTypePlainCPU_MD5::MFNHashTypePlainCPU_MD5()\n");

    // Vector width of 12 - 3x interlaced SSE2
    this->VectorWidth = 12;
}

void MFNHashTypePlainCPU_MD5::launchKernel() {
    trace_printf("MFNHashTypePlainCPU_MD5::launchKernel()\n");   
}

void MFNHashTypePlainCPU_MD5::printLaunchDebugData() {
}

std::vector<uint8_t> MFNHashTypePlainCPU_MD5::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypePlainCPU_MD5::preProcessHash()\n");
    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&rawHash[0];

    /*
    printf("MFNHashTypePlainCPU_MD5::preProcessHash()\n");
    printf("Raw Hash: ");
    for (i = 0; i < rawHash.size(); i++) {
        printf("%02x", rawHash[i]);
    }
    printf("\n");
    */
    a = hash32[0];
    b = hash32[1];
    c = hash32[2];
    d = hash32[3];
    
    a -= 0x67452301;
    b -= 0xefcdab89;
    c -= 0x98badcfe;
    d -= 0x10325476;
    
    if (this->passwordLength < 8) {
        REV_II (b, c, d, a, 0x00 /*b9*/, MD5S44, 0xeb86d391); //64
        REV_II (c, d, a, b, 0x00 /*b2*/, MD5S43, 0x2ad7d2bb); //63
        REV_II (d, a, b, c, 0x00 /*b11*/, MD5S42, 0xbd3af235); //62
        REV_II (a, b, c, d, 0x00 /*b4*/, MD5S41, 0xf7537e82); //61
        REV_II (b, c, d, a, 0x00 /*b13*/, MD5S44, 0x4e0811a1); //60
        REV_II (c, d, a, b, 0x00 /*b6*/, MD5S43, 0xa3014314); //59
        REV_II (d, a, b, c, 0x00 /*b15*/, MD5S42, 0xfe2ce6e0); //58
        REV_II (a, b, c, d, 0x00 /*b8*/, MD5S41, 0x6fa87e4f); //57
    } else if (this->passwordLength == 8) {
        REV_II (b, c, d, a, 0x00 /*b9*/, MD5S44, 0xeb86d391); //64
        // Padding bit will be set
        REV_II (c, d, a, b, 0x00000080 /*b2*/, MD5S43, 0x2ad7d2bb); //63
        REV_II (d, a, b, c, 0x00 /*b11*/, MD5S42, 0xbd3af235); //62
        REV_II (a, b, c, d, 0x00 /*b4*/, MD5S41, 0xf7537e82); //61
        REV_II (b, c, d, a, 0x00 /*b13*/, MD5S44, 0x4e0811a1); //60
        REV_II (c, d, a, b, 0x00 /*b6*/, MD5S43, 0xa3014314); //59
        REV_II (d, a, b, c, 0x00 /*b15*/, MD5S42, 0xfe2ce6e0); //58
        REV_II (a, b, c, d, 0x00 /*b8*/, MD5S41, 0x6fa87e4f); //57
    }
    
    hash32[0] = a;
    hash32[1] = b;
    hash32[2] = c;
    hash32[3] = d;
    /*
    printf("Preprocessed Hash: ");
    for (i = 0; i < rawHash.size(); i++) {
        printf("%02x", rawHash[i]);
    }
    printf("\n");
    
    printf("Returning rawHash\n");
    */
    return rawHash;
}

std::vector<uint8_t> MFNHashTypePlainCPU_MD5::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypePlainCPU_MD5::postProcessHash()\n");
    
    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&processedHash[0];
    
    a = hash32[0];
    b = hash32[1];
    c = hash32[2];
    d = hash32[3];

    if (this->passwordLength < 8) {
        MD5II(a, b, c, d, 0x00, MD5S41, 0x6fa87e4f); /* 57 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xfe2ce6e0); /* 58 */
        MD5II(c, d, a, b, 0x00, MD5S43, 0xa3014314); /* 59 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0x4e0811a1); /* 60 */
        MD5II(a, b, c, d, 0x00, MD5S41, 0xf7537e82); /* 61 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xbd3af235); /* 62 */
        MD5II(c, d, a, b, 0x00, MD5S43, 0x2ad7d2bb); /* 63 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0xeb86d391); /* 64 */
    } else if (this->passwordLength == 8) {
        MD5II(a, b, c, d, 0x00, MD5S41, 0x6fa87e4f); /* 57 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xfe2ce6e0); /* 58 */
        MD5II(c, d, a, b, 0x00, MD5S43, 0xa3014314); /* 59 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0x4e0811a1); /* 60 */
        MD5II(a, b, c, d, 0x00, MD5S41, 0xf7537e82); /* 61 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xbd3af235); /* 62 */
        MD5II(c, d, a, b, 0x00000080, MD5S43, 0x2ad7d2bb); /* 63 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0xeb86d391); /* 64 */
    }
    
    a += 0x67452301;
    b += 0xefcdab89;
    c += 0x98badcfe;
    d += 0x10325476;
    
    hash32[0] = a;
    hash32[1] = b;
    hash32[2] = c;
    hash32[3] = d;

    
    return processedHash;
}

void MFNHashTypePlainCPU_MD5::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCPU_MD5::copyConstantDataToDevice()\n");
}

void MFNHashTypePlainCPU_MD5::checkAndReportHashCpuMD5(
        uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
        uint32_t a, uint32_t b, uint32_t c, uint32_t d)
 {
    trace_printf("MFNHashTypePlainCPU_MD5::checkAndReportHashCpu()\n");
    trace_printf("a: 0x%08x  b: 0x%08x  c: 0x%08x  d: %08x\n", a, b, c, d);

    std::vector<uint8_t> hashToCompare;

    // Great news!  On the CPU, we can use this->activeHashesProcessed, which is
    // a vector!  No offsets needed!

    // Copy the values into the vector.
    hashToCompare.resize(16, 0);
    memcpy(&hashToCompare[0], &a, 4);
    memcpy(&hashToCompare[4], &b, 4);
    memcpy(&hashToCompare[8], &c, 4);
    memcpy(&hashToCompare[12], &d, 4);

    // Check for the hash in the vector.
    // Note that the hashes are sorted in little endian 32-bit format.
    // Therefore, we use a custom search predicate.
    if (std::binary_search (this->activeHashesProcessed.begin(),
            this->activeHashesProcessed.end(), hashToCompare, hashMD5SortPredicate)) {

        // Lock the mutex to report the hash.
        this->MFNHashTypePlainCPUMutex.lock();

        // Create a password vector, copy data in.
        std::vector<uint8_t> foundPassword;
        foundPassword.resize(16, 0);
        memcpy(&foundPassword[0], &b0, 4);
        memcpy(&foundPassword[4], &b1, 4);
        memcpy(&foundPassword[8], &b2, 4);
        memcpy(&foundPassword[12], &b3, 4);
        // Trim to length.
        foundPassword.resize(this->passwordLength);

        this->HashFile->ReportFoundPassword(this->postProcessHash(hashToCompare), foundPassword, MFN_PASSWORD_SINGLE_MD5);
        // Report the found hash over the network.
        if (MultiforcerGlobalClassFactory.getCommandlinedataClass()->GetIsNetworkClient()) {
            MultiforcerGlobalClassFactory.getNetworkClientClass()->
                    submitFoundHash(this->postProcessHash(hashToCompare), foundPassword, MFN_PASSWORD_SINGLE_MD5);
        }

        this->Display->addCrackedPassword(foundPassword);
        //printf("====>Found password %s!\n", (char *)&foundPassword[0]);
        //fflush(stdout);
        this->MFNHashTypePlainCPUMutex.unlock();
    }

}


/**
 * This thread will be called on all the CPUs as required.  It will load the
 * initial values, and run through the entire specified number of steps to
 * run, checking the hash at each step.  This allows us to utilize the spare
 * CPU time on the host.  As the CPU threads do not need to be interrupted
 * as the GPU threads do to allow the display to update, this just runs all
 * the way through from start to finish, checking occasionally to see if it
 * should terminate and return.
 */
void MFNHashTypePlainCPU_MD5::cpuSSEThread(CPUSSEThreadData threadData) {
    trace_printf("MFNHashTypePlainCPU_MD5::cpuSSEThread()\n");

    uint64_t step, periodStep = 0;

    // Set up 32-bit word access to the initial array.
    uint32_t *initialValues32 = (uint32_t *)&this->HostStartPasswords32[0];

    // Base offset for vector loads.
    uint32_t baseOffset;

    // Offset for charset incrementors.
    uint8_t passOffset;

    // For password checking & incrementing.
    // vectorIndex: 0-2 for the SSE vector, passwordIndex 0-3 for the SSE index.
    uint32_t vectorIndex, passwordIndex;

    // Set up the initial variables for the MD5 operations.
    initializeVariables();

    // Support through len15 - 4 words.
    // Len16 would require b4.  Other kernel!
    __m128i b0_0, b0_1, b0_2;
    __m128i b1_0, b1_1, b1_2;
    __m128i b2_0, b2_1, b2_2;
    __m128i b3_0, b3_1, b3_2;
    
    // Array for pointing to b0-b3 as 32-bit words
    uint32_t *inputBlocks[4][3];

    inputBlocks[0][0] = (uint32_t *)&b0_0;
    inputBlocks[0][1] = (uint32_t *)&b0_1;
    inputBlocks[0][2] = (uint32_t *)&b0_2;
    inputBlocks[1][0] = (uint32_t *)&b1_0;
    inputBlocks[1][1] = (uint32_t *)&b1_1;
    inputBlocks[1][2] = (uint32_t *)&b1_2;
    inputBlocks[2][0] = (uint32_t *)&b2_0;
    inputBlocks[2][1] = (uint32_t *)&b2_1;
    inputBlocks[2][2] = (uint32_t *)&b2_2;
    inputBlocks[3][0] = (uint32_t *)&b3_0;
    inputBlocks[3][1] = (uint32_t *)&b3_1;
    inputBlocks[3][2] = (uint32_t *)&b3_2;


    // For length.
    __m128i b14;

    __m128i a, b, c, d, tmp1;
    __m128i a2, b2, c2, d2, tmp1_2;
    __m128i a3, b3, c3, d3, tmp1_3;


    uint32_t *outputBlocks[4][3];

    outputBlocks[0][0] = (uint32_t *)&a;
    outputBlocks[0][1] = (uint32_t *)&a2;
    outputBlocks[0][2] = (uint32_t *)&a3;
    outputBlocks[1][0] = (uint32_t *)&b;
    outputBlocks[1][1] = (uint32_t *)&b2;
    outputBlocks[1][2] = (uint32_t *)&b3;
    outputBlocks[2][0] = (uint32_t *)&c;
    outputBlocks[2][1] = (uint32_t *)&c2;
    outputBlocks[2][2] = (uint32_t *)&c3;
    outputBlocks[3][0] = (uint32_t *)&d;
    outputBlocks[3][1] = (uint32_t *)&d2;
    outputBlocks[3][2] = (uint32_t *)&d3;

    __m128i mOne = _mm_set1_epi32(0xFFFFFFFF);

    // Set up the bitmap mask based on the current bitmap size.
    uint32_t bitmapMask = this->classBitmapLookup_a.size() - 1;

    // Load the vector initial values.
//    printf("Thread %d, hostStartPasswords32 in words: %d\n", threadData.threadNumber, this->HostStartPasswords32.size() / 4);
//    printf("Thread %d, should load values for b0 starting at offset %d\n", threadData.threadNumber, threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH);
//    printf("Thread %d, should load values for b1 starting at offset %d\n", threadData.threadNumber, (1 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH));
//    printf("Thread %d, should load values for b1 starting at offset %d\n", threadData.threadNumber, (2 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH));

    // Init b0-b3 with null in case they are not used.
    b0_0 = b0_1 = b0_2 = _mm_set1_epi32(0);
    b1_0 = b1_1 = b1_2 = _mm_set1_epi32(0);
    b2_0 = b2_1 = b2_2 = _mm_set1_epi32(0);
    b3_0 = b3_1 = b3_2 = _mm_set1_epi32(0);

    // Load b0 based on threadNumber and position 0
    baseOffset = (0 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH);
    b0_0 = _mm_set_epi32(initialValues32[baseOffset + 0], initialValues32[baseOffset + 1],
                         initialValues32[baseOffset + 2], initialValues32[baseOffset + 3]);
    b0_1 = _mm_set_epi32(initialValues32[baseOffset + 4], initialValues32[baseOffset + 5],
                         initialValues32[baseOffset + 6], initialValues32[baseOffset + 7]);
    b0_2 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                         initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);

    // If passLen > 3, load b1
    if (this->passwordLength > 3) {
        baseOffset = (1 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH);
        b1_0 = _mm_set_epi32(initialValues32[baseOffset + 0], initialValues32[baseOffset + 1],
                             initialValues32[baseOffset + 2], initialValues32[baseOffset + 3]);
        b1_1 = _mm_set_epi32(initialValues32[baseOffset + 4], initialValues32[baseOffset + 5],
                             initialValues32[baseOffset + 6], initialValues32[baseOffset + 7]);
        b1_2 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                             initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);
    }

    // If passLen > 7, load b2
    if (this->passwordLength > 7) {
        baseOffset = (2 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH);
        b2_0 = _mm_set_epi32(initialValues32[baseOffset + 0], initialValues32[baseOffset + 1],
                             initialValues32[baseOffset + 2], initialValues32[baseOffset + 3]);
        b2_1 = _mm_set_epi32(initialValues32[baseOffset + 4], initialValues32[baseOffset + 5],
                             initialValues32[baseOffset + 6], initialValues32[baseOffset + 7]);
        b2_2 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                             initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);
    }

    // If passLen > 11, load b2
    if (this->passwordLength > 11) {
        baseOffset = (2 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH);
        b3_0 = _mm_set_epi32(initialValues32[baseOffset + 0], initialValues32[baseOffset + 1],
                             initialValues32[baseOffset + 2], initialValues32[baseOffset + 3]);
        b3_1 = _mm_set_epi32(initialValues32[baseOffset + 4], initialValues32[baseOffset + 5],
                             initialValues32[baseOffset + 6], initialValues32[baseOffset + 7]);
        b3_2 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                             initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);
    }

    // Set the length in all vectors.
    b14 = _mm_set1_epi32(this->passwordLength * 8);

    for (step = 0; step < threadData.numberStepsTotal; step++) {

        // If the defined period has been reached, wait on all threads.
        if (periodStep == MFNHASHTYPECPUPLAIN_STEPS_PER_PERIOD) {
            // Wait for all threads to get here.
            threadData.classBarrier->wait();

            // Check to see if we should exit, and do so.
            if (global_interface.exit) {
                return;
            }

            periodStep = 0;
        }
        periodStep++;

        // Load the a/b/c/d vars with the initial values.
        a = mCa; a2 = mCa; a3 = mCa;
        b = mCb; b2 = mCb; b3 = mCb;
        c = mCc; c2 = mCc; c3 = mCc;
        d = mCd; d2 = mCd; d3 = mCd;

        MD5STEP_ROUND1(F, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3,  mAC1, b0_0, b0_1, b0_2,  S11);\
        MD5STEP_ROUND1(F, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3,  mAC2, b1_0, b1_1, b1_2,  S12);\
        MD5STEP_ROUND1(F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3,  mAC3, b2_0, b2_1, b2_2,  S13);\
        MD5STEP_ROUND1(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3,  mAC4, b3_0, b3_1, b3_2,  S14);\
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
        MD5STEP_ROUND1     (F, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC15, b14, b14, b14, S13);\
        MD5STEP_ROUND1_NULL(F, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC16,   S14);\

        MD5STEP_ROUND2     (G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC17, b1_0, b1_1, b1_2,  S21);\
        MD5STEP_ROUND2_NULL(G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC18,   S22);\
        MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC19,   S23);\
        MD5STEP_ROUND2     (G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC20, b0_0, b0_1, b0_2,  S24);\
        MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC21,   S21);\
        MD5STEP_ROUND2_NULL(G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC22,   S22);\
        MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC23,   S23);\
        MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC24,   S24);\
        MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC25,   S21);\
        MD5STEP_ROUND2     (G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC26, b14, b14, b14, S22);\
        MD5STEP_ROUND2     (G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC27, b3_0, b3_1, b3_2,  S23);\
        MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC28,   S24);\
        MD5STEP_ROUND2_NULL(G, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC29,   S21);\
        MD5STEP_ROUND2     (G, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC30, b2_0, b2_1, b2_2,  S22);\
        MD5STEP_ROUND2_NULL(G, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC31,   S23);\
        MD5STEP_ROUND2_NULL(G, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC32,   S24);\

        MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC33,   S31);\
        MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC34,   S32);\
        MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC35,   S33);\
        MD5STEP_ROUND3     (H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC36, b14, b14, b14, S34);\
        MD5STEP_ROUND3     (H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC37, b1_0, b1_1, b1_2,  S31);\
        MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC38,   S32);\
        MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC39,   S33);\
        MD5STEP_ROUND3_NULL(H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC40,   S34);\
        MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC41,   S31);\
        MD5STEP_ROUND3     (H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC42, b0_0, b0_1, b0_2,  S32);\
        MD5STEP_ROUND3     (H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC43, b3_0, b3_1, b3_2,  S33);\
        MD5STEP_ROUND3_NULL(H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC44,   S34);\
        MD5STEP_ROUND3_NULL(H, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC45,   S31);\
        MD5STEP_ROUND3_NULL(H, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC46,   S32);\
        MD5STEP_ROUND3_NULL(H, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC47,   S33);\
        MD5STEP_ROUND3     (H, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC48, b2_0, b2_1, b2_2, S34);\
        \
        MD5STEP_ROUND4     (I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC49, b0_0, b0_1, b0_2,  S41);\
        MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC50,   S42);\
        MD5STEP_ROUND4     (I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC51, b14, b14, b14,  S43);\
        MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC52,   S44);\
        MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC53,   S41);\
        MD5STEP_ROUND4     (I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC54, b3_0, b3_1, b3_2,  S42);\
        MD5STEP_ROUND4_NULL(I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC55,   S43);\
        MD5STEP_ROUND4     (I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC56, b1_0, b1_1, b1_2,  S44);\
        
        if (this->passwordLength > 8) {
            MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC57,   S41);\
            MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC58,   S42);\
            MD5STEP_ROUND4_NULL(I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC59,   S43);\
            MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC60,   S44);\
            MD5STEP_ROUND4_NULL(I, a, a2, a3, b, b2, b3, c, c2, c3, d, d2, d3, mAC61,   S41);\
            MD5STEP_ROUND4_NULL(I, d, d2, d3, a, a2, a3, b, b2, b3, c, c2, c3, mAC62,   S42);\
            MD5STEP_ROUND4     (I, c, c2, c3, d, d2, d3, a, a2, a3, b, b2, b3, mAC63, b2_0, b2_1, b2_2,  S43);\
            MD5STEP_ROUND4_NULL(I, b, b2, b3, c, c2, c3, d, d2, d3, a, a2, a3, mAC64,   S44);\
        }

//        printPassword128(b0_0, b1_0, b2_0, b3_0, this->passwordLength);
//        printPassword128(b0_1, b1_1, b2_1, b3_1, this->passwordLength);
//        printPassword128(b0_2, b1_2, b2_2, b3_2, this->passwordLength);

        for (vectorIndex = 0; vectorIndex < 3; vectorIndex++) {
            for (passwordIndex = 0; passwordIndex < 4; passwordIndex++) {
                if ((this->classBitmapLookup_a[(outputBlocks[0][vectorIndex][passwordIndex] >> 3) & bitmapMask] >> (outputBlocks[0][vectorIndex][passwordIndex] & 0x7)) & 0x1) {
                    if (this->globalBitmap128mb_a[(outputBlocks[0][vectorIndex][passwordIndex] >> 3) & 0x07FFFFFF] >> (outputBlocks[0][vectorIndex][passwordIndex] & 0x7) & 0x1) {
                        if (this->globalBitmap128mb_b[(outputBlocks[1][vectorIndex][passwordIndex] >> 3) & 0x07FFFFFF] >> (outputBlocks[1][vectorIndex][passwordIndex] & 0x7) & 0x1) {
                            if (this->globalBitmap128mb_c[(outputBlocks[2][vectorIndex][passwordIndex] >> 3) & 0x07FFFFFF] >> (outputBlocks[2][vectorIndex][passwordIndex] & 0x7) & 0x1) {
                                if (this->globalBitmap128mb_d[(outputBlocks[3][vectorIndex][passwordIndex] >> 3) & 0x07FFFFFF] >> (outputBlocks[3][vectorIndex][passwordIndex] & 0x7) & 0x1) {
                                    //printf("Global bitmap hit: a[%d]\n", 3 - i);
                                    //printPassword128(b0_0, b1_0, b2_0, b3_0, this->passwordLength);
                                    this->checkAndReportHashCpuMD5(
                                            inputBlocks[0][vectorIndex][passwordIndex],
                                            inputBlocks[1][vectorIndex][passwordIndex],
                                            inputBlocks[2][vectorIndex][passwordIndex],
                                            inputBlocks[3][vectorIndex][passwordIndex],
                                            outputBlocks[0][vectorIndex][passwordIndex],
                                            outputBlocks[1][vectorIndex][passwordIndex],
                                            outputBlocks[2][vectorIndex][passwordIndex],
                                            outputBlocks[3][vectorIndex][passwordIndex]);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Increment the passwords.
//        printf("Pre-increment: \n");
//        printPassword128(b0_0, b1_0, b2_0, b3_0, this->passwordLength);

        // Increment the passwords as needed.
        for (vectorIndex = 0; vectorIndex < 3; vectorIndex++) {
            uint32_t w0, w1, w2, w3;

            for (passwordIndex = 0; passwordIndex < 4; passwordIndex++) {
                // Load the appropriate b0/b1 values.
                w0 = inputBlocks[0][vectorIndex][passwordIndex];
                w1 = inputBlocks[1][vectorIndex][passwordIndex];
                w2 = inputBlocks[2][vectorIndex][passwordIndex];
                w3 = inputBlocks[3][vectorIndex][passwordIndex];
                // Increment the appropriate password
                if (this->charsetLengths[1] == 0) {
                    // Single incrementors
                    switch(this->passwordLength) {
                        case 1:
                        makeMFNSingleIncrementorsCPU1(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 2:
                        makeMFNSingleIncrementorsCPU2(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 3:
                        makeMFNSingleIncrementorsCPU3(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 4:
                        makeMFNSingleIncrementorsCPU4(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 5:
                        makeMFNSingleIncrementorsCPU5(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 6:
                        makeMFNSingleIncrementorsCPU6(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 7:
                        makeMFNSingleIncrementorsCPU7(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 8:
                        makeMFNSingleIncrementorsCPU8(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 9:
                        makeMFNSingleIncrementorsCPU9(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 10:
                        makeMFNSingleIncrementorsCPU10(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 11:
                        makeMFNSingleIncrementorsCPU11(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 12:
                        makeMFNSingleIncrementorsCPU12(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 13:
                        makeMFNSingleIncrementorsCPU13(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 14:
                        makeMFNSingleIncrementorsCPU14(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 15:
                        makeMFNSingleIncrementorsCPU15(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                    }
                } else {
                    // Multi incrementors
                    switch(this->passwordLength) {
                        case 1:
                        makeMFNMultipleIncrementorsCPU1(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 2:
                        makeMFNMultipleIncrementorsCPU2(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 3:
                        makeMFNMultipleIncrementorsCPU3(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 4:
                        makeMFNMultipleIncrementorsCPU4(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 5:
                        makeMFNMultipleIncrementorsCPU5(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 6:
                        makeMFNMultipleIncrementorsCPU6(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 7:
                        makeMFNMultipleIncrementorsCPU7(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 8:
                        makeMFNMultipleIncrementorsCPU8(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 9:
                        makeMFNMultipleIncrementorsCPU9(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 10:
                        makeMFNMultipleIncrementorsCPU10(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 11:
                        makeMFNMultipleIncrementorsCPU11(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 12:
                        makeMFNMultipleIncrementorsCPU12(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 13:
                        makeMFNMultipleIncrementorsCPU13(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 14:
                        makeMFNMultipleIncrementorsCPU14(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 15:
                        makeMFNMultipleIncrementorsCPU15(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                    }
                }
                // Store the password back in the right location in the vector.
                inputBlocks[0][vectorIndex][passwordIndex] = w0;
                inputBlocks[1][vectorIndex][passwordIndex] = w1;
                inputBlocks[2][vectorIndex][passwordIndex] = w2;
                inputBlocks[3][vectorIndex][passwordIndex] = w3;
            }
        }

//        printf("Post-increment: \n");
//        printPassword128(b0_0, b1_0, b2_0, b3_0, this->passwordLength);

    }
}

