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

#include "MFN_CPU_host/MFNHashTypePlainCPU_NTLM.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_CPU_host/MFNSSE2_MD4.h"
#include "MFN_CPU_host/MFN_CPU_NTLM_inc.h"
#include "CH_HashFiles/CHHashFileV.h"
#include "MFN_Common/MFNDisplay.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "MFN_Common/MFNNetworkClient.h"
#include "MFN_Common/MFNCommandLineData.h"
#include "MFN_Common/MFNDefines.h"

extern MFNClassFactory MultiforcerGlobalClassFactory;

extern struct global_commands global_interface;

// How many SSE vectors each kernel deals with.
#define SSE_KERNEL_VECTOR_WIDTH 16


#define MD4ROTATE_RIGHT(x, n) (((x) >> (n)) | ((x) << (32-(n))))
#define MD4ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

#define MD4H(x, y, z) ((x) ^ (y) ^ (z))

#define MD4HH(a, b, c, d, x, s) { \
    (a) += MD4H ((b), (c), (d)) + (x) + (uint32_t)0x6ed9eba1; \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }

#define REV_HH(a,b,c,d,data,shift) \
    a = MD4ROTATE_RIGHT((a), shift) - data - (uint32_t)0x6ed9eba1 - (b ^ c ^ d);



#define MD4S31 3
#define MD4S32 9
#define MD4S33 11
#define MD4S34 15

// Prints a 128 bit value.
static void print128(__m128i value) {
    int64_t *v64 = (int64_t*) &value;
    printf("%.16llx %.16llx\n", v64[1], v64[0]);
}

static void printPassword128NTLM(__m128i &b0, __m128i &b1, __m128i &b2, __m128i &b3,
    __m128i &b4, __m128i &b5, __m128i &b6, __m128i &b7, int passLen) {
    uint32_t *b0_32 = (uint32_t *)&b0;
    uint32_t *b1_32 = (uint32_t *)&b1;
    uint32_t *b2_32 = (uint32_t *)&b2;
    uint32_t *b3_32 = (uint32_t *)&b3;
    uint32_t *b4_32 = (uint32_t *)&b4;
    uint32_t *b5_32 = (uint32_t *)&b5;
    uint32_t *b6_32 = (uint32_t *)&b6;
    uint32_t *b7_32 = (uint32_t *)&b7;
    int pass;

    for (pass = 0; pass < 4; pass++) {
        printf("Pass %d: '", pass);
        if (passLen > 0) printf("%c", (b0_32[3 - pass] >> 0) & 0xff);
        if (passLen > 1) printf("%c", (b0_32[3 - pass] >> 16) & 0xff);

        if (passLen > 2) printf("%c", (b1_32[3 - pass] >> 0) & 0xff);
        if (passLen > 3) printf("%c", (b1_32[3 - pass] >> 16) & 0xff);

        if (passLen > 4) printf("%c", (b2_32[3 - pass] >> 0) & 0xff);
        if (passLen > 5) printf("%c", (b2_32[3 - pass] >> 16) & 0xff);
        
        if (passLen > 6) printf("%c", (b3_32[3 - pass] >> 0) & 0xff);
        if (passLen > 7) printf("%c", (b3_32[3 - pass] >> 16) & 0xff);

        if (passLen > 8) printf("%c", (b4_32[3 - pass] >> 0) & 0xff);
        if (passLen > 9) printf("%c", (b4_32[3 - pass] >> 16) & 0xff);
        
        if (passLen > 10) printf("%c", (b5_32[3 - pass] >> 0) & 0xff);
        if (passLen > 11) printf("%c", (b5_32[3 - pass] >> 16) & 0xff);

        if (passLen > 12) printf("%c", (b6_32[3 - pass] >> 0) & 0xff);
        if (passLen > 13) printf("%c", (b6_32[3 - pass] >> 16) & 0xff);
        
        if (passLen > 14) printf("%c", (b7_32[3 - pass] >> 0) & 0xff);
        if (passLen > 15) printf("%c", (b7_32[3 - pass] >> 16) & 0xff);

        printf("'\n");
    }
}

// Deal with hashes that are little endian 32-bits.
static bool hashNTLMSortPredicate(const std::vector<uint8_t> &h1, const std::vector<uint8_t> &h2) {
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

MFNHashTypePlainCPU_NTLM::MFNHashTypePlainCPU_NTLM() :  MFNHashTypePlainCPU(16) {
    trace_printf("MFNHashTypePlainCPU_NTLM::MFNHashTypePlainCPU_NTLM()\n");

    // Vector width of 16 - 4x interlaced SSE2
    this->VectorWidth = 16;
}

void MFNHashTypePlainCPU_NTLM::launchKernel() {
    trace_printf("MFNHashTypePlainCPU_NTLM::launchKernel()\n");
}

void MFNHashTypePlainCPU_NTLM::printLaunchDebugData() {
}

std::vector<uint8_t> MFNHashTypePlainCPU_NTLM::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypePlainCPU_NTLM::preProcessHash()\n");
    
    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&rawHash[0];
    
    /*
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
    
        
    // Always unwind the final constants
    a -= 0x67452301;
    b -= 0xefcdab89;
    c -= 0x98badcfe;
    d -= 0x10325476;

    // Always unwinding b15 - length field, always 0x00
    REV_HH(b, c, d, a, 0x00, MD4S34);

    if (this->passwordLength < 6) {
        // Unwind back through b9, with b3 = 0x00
        REV_HH (c, d, a, b, 0x00, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
        REV_HH (a, b, c, d, 0x00, MD4S31);
        REV_HH (b, c, d, a, 0x00, MD4S34);
        REV_HH (c, d, a, b, 0x00, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
    } else if (this->passwordLength == 6) {
        // Unwind through b9, with b3 = 0x00000080
        REV_HH (c, d, a, b, 0x00, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
        REV_HH (a, b, c, d, 0x80, MD4S31);
        REV_HH (b, c, d, a, 0x00, MD4S34);
        REV_HH (c, d, a, b, 0x00, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
    } else if (this->passwordLength < 14) {
        // Rewind through b3 with b7 = 0x00
        REV_HH (c, d, a, b, 0x00, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
    } else if (this->passwordLength == 14) {
        // Rewind through b3 with b7 = 0x80
        REV_HH (c, d, a, b, 0x80, MD4S33);
        REV_HH (d, a, b, c, 0x00, MD4S32);
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
    return rawHash;}

std::vector<uint8_t> MFNHashTypePlainCPU_NTLM::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypePlainCPU_NTLM::postProcessHash()\n");

    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&processedHash[0];

    a = hash32[0];
    b = hash32[1];
    c = hash32[2];
    d = hash32[3];
    
    if (this->passwordLength < 6) {
        // Rewind back through b9, with b3 = 0x00
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x00, MD4S33);
        MD4HH (b, c, d, a, 0x00, MD4S34);
        MD4HH (a, b, c, d, 0x00, MD4S31);
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x00, MD4S33);
    } else if (this->passwordLength == 6) {
        // Rewind with b3 = 0x80
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x00, MD4S33);
        MD4HH (b, c, d, a, 0x00, MD4S34);
        MD4HH (a, b, c, d, 0x80, MD4S31);
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x00, MD4S33);
    } else if (this->passwordLength < 14) {
        // Rewind through b3 with b7 = 0x00
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x00, MD4S33);
    } else if (this->passwordLength == 14) {
        // Rewind through b3 with b7 = 0x80
        MD4HH (d, a, b, c, 0x00, MD4S32);
        MD4HH (c, d, a, b, 0x80, MD4S33);
    }

    // Always add b15 - will always be 0 (length field)
    MD4HH (b, c, d, a, 0x00, MD4S34);
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

void MFNHashTypePlainCPU_NTLM::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainCPU_NTLM::copyConstantDataToDevice()\n");
}

void MFNHashTypePlainCPU_NTLM::checkAndReportHashCpuNTLM(
        uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
        uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7,
        uint32_t a, uint32_t b, uint32_t c, uint32_t d)
 {
    trace_printf("MFNHashTypePlainCPU_NTLM::checkAndReportHashCpu()\n");
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
            this->activeHashesProcessed.end(), hashToCompare, hashNTLMSortPredicate)) {

        // Lock the mutex to report the hash.
        this->MFNHashTypePlainCPUMutex.lock();

        // Create a password vector, copy data in.
        std::vector<uint8_t> foundPassword;
        foundPassword.resize(16, 0);
        if (this->passwordLength >= 1) {foundPassword[0] = (uint8_t)((b0 >> 0) & 0xff);}
        if (this->passwordLength >= 2) {foundPassword[1] = (uint8_t)((b0 >> 16) & 0xff);}
        if (this->passwordLength >= 3) {foundPassword[2] = (uint8_t)((b1 >> 0) & 0xff);}
        if (this->passwordLength >= 4) {foundPassword[3] = (uint8_t)((b1 >> 16) & 0xff);}
        if (this->passwordLength >= 5) {foundPassword[4] = (uint8_t)((b2 >> 0) & 0xff);}
        if (this->passwordLength >= 6) {foundPassword[5] = (uint8_t)((b2 >> 16) & 0xff);}
        if (this->passwordLength >= 7) {foundPassword[6] = (uint8_t)((b3 >> 0) & 0xff);}
        if (this->passwordLength >= 8) {foundPassword[7] = (uint8_t)((b3 >> 16) & 0xff);}
        if (this->passwordLength >= 9) {foundPassword[8] = (uint8_t)((b4 >> 0) & 0xff);}
        if (this->passwordLength >= 10) {foundPassword[9] = (uint8_t)((b4 >> 16) & 0xff);}
        if (this->passwordLength >= 11) {foundPassword[10] = (uint8_t)((b5 >> 0) & 0xff);}
        if (this->passwordLength >= 12) {foundPassword[11] = (uint8_t)((b5 >> 16) & 0xff);}
        if (this->passwordLength >= 13) {foundPassword[12] = (uint8_t)((b6 >> 0) & 0xff);}
        if (this->passwordLength >= 14) {foundPassword[13] = (uint8_t)((b6 >> 16) & 0xff);}
        if (this->passwordLength >= 15) {foundPassword[14] = (uint8_t)((b7 >> 0) & 0xff);}
        if (this->passwordLength >= 16) {foundPassword[15] = (uint8_t)((b7 >> 16) & 0xff);}
        // Trim to length.
        foundPassword.resize(this->passwordLength);

        this->HashFile->ReportFoundPassword(this->postProcessHash(hashToCompare), foundPassword, MFN_PASSWORD_NTLM);
        if (MultiforcerGlobalClassFactory.getCommandlinedataClass()->GetIsNetworkClient()) {
            MultiforcerGlobalClassFactory.getNetworkClientClass()->
                    submitFoundHash(this->postProcessHash(hashToCompare), foundPassword, MFN_PASSWORD_NTLM);
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
void MFNHashTypePlainCPU_NTLM::cpuSSEThread(CPUSSEThreadData threadData) {
    trace_printf("MFNHashTypePlainCPU_NTLM::cpuSSEThread()\n");

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

    // Set up the initial variables for the NTLM operations.
    //initializeVariables();

    // Support through len15 - 4 words.
    // Len16 would require b4.  Other kernel!
    __m128i b0_0, b0_1, b0_2, b0_3;
    __m128i b1_0, b1_1, b1_2, b1_3;
    __m128i b2_0, b2_1, b2_2, b2_3;
    __m128i b3_0, b3_1, b3_2, b3_3;
    __m128i b4_0, b4_1, b4_2, b4_3;
    __m128i b5_0, b5_1, b5_2, b5_3;
    __m128i b6_0, b6_1, b6_2, b6_3;
    __m128i b7_0, b7_1, b7_2, b7_3;

    // Masks for loading passwords.
    __m128i mask_ff, mask_ff00, mask_ff0000, mask_ff000000;
    mask_ff = _mm_set1_epi32(0xff);
    mask_ff00 = _mm_set1_epi32(0xff00);
    mask_ff0000 = _mm_set1_epi32(0xff0000);
    mask_ff000000 = _mm_set1_epi32(0xff000000);


    // Array for pointing to b0-b3 as 32-bit words
    uint32_t *inputBlocks[8][4];

    inputBlocks[0][0] = (uint32_t *)&b0_0;
    inputBlocks[0][1] = (uint32_t *)&b0_1;
    inputBlocks[0][2] = (uint32_t *)&b0_2;
    inputBlocks[0][3] = (uint32_t *)&b0_3;

    inputBlocks[1][0] = (uint32_t *)&b1_0;
    inputBlocks[1][1] = (uint32_t *)&b1_1;
    inputBlocks[1][2] = (uint32_t *)&b1_2;
    inputBlocks[1][3] = (uint32_t *)&b1_3;

    inputBlocks[2][0] = (uint32_t *)&b2_0;
    inputBlocks[2][1] = (uint32_t *)&b2_1;
    inputBlocks[2][2] = (uint32_t *)&b2_2;
    inputBlocks[2][3] = (uint32_t *)&b2_3;

    inputBlocks[3][0] = (uint32_t *)&b3_0;
    inputBlocks[3][1] = (uint32_t *)&b3_1;
    inputBlocks[3][2] = (uint32_t *)&b3_2;
    inputBlocks[3][3] = (uint32_t *)&b3_3;

    inputBlocks[4][0] = (uint32_t *)&b4_0;
    inputBlocks[4][1] = (uint32_t *)&b4_1;
    inputBlocks[4][2] = (uint32_t *)&b4_2;
    inputBlocks[4][3] = (uint32_t *)&b4_3;

    inputBlocks[5][0] = (uint32_t *)&b5_0;
    inputBlocks[5][1] = (uint32_t *)&b5_1;
    inputBlocks[5][2] = (uint32_t *)&b5_2;
    inputBlocks[5][3] = (uint32_t *)&b5_3;

    inputBlocks[6][0] = (uint32_t *)&b6_0;
    inputBlocks[6][1] = (uint32_t *)&b6_1;
    inputBlocks[6][2] = (uint32_t *)&b6_2;
    inputBlocks[6][3] = (uint32_t *)&b6_3;

    inputBlocks[7][0] = (uint32_t *)&b7_0;
    inputBlocks[7][1] = (uint32_t *)&b7_1;
    inputBlocks[7][2] = (uint32_t *)&b7_2;
    inputBlocks[7][3] = (uint32_t *)&b7_3;
    
    // For length.
    __m128i b14_0, b14_1, b14_2, b14_3;

    __m128i a_0, b_0, c_0, d_0, tmp1_0, tmp2_0;
    __m128i a_1, b_1, c_1, d_1, tmp1_1, tmp2_1;
    __m128i a_2, b_2, c_2, d_2, tmp1_2, tmp2_2;
    __m128i a_3, b_3, c_3, d_3, tmp1_3, tmp2_3;
    __m128i mCa, mCb, mCc, mCd;


    uint32_t *outputBlocks[4][4];

    outputBlocks[0][0] = (uint32_t *)&a_0;
    outputBlocks[0][1] = (uint32_t *)&a_1;
    outputBlocks[0][2] = (uint32_t *)&a_2;
    outputBlocks[0][3] = (uint32_t *)&a_3;

    outputBlocks[1][0] = (uint32_t *)&b_0;
    outputBlocks[1][1] = (uint32_t *)&b_1;
    outputBlocks[1][2] = (uint32_t *)&b_2;
    outputBlocks[1][3] = (uint32_t *)&b_3;

    outputBlocks[2][0] = (uint32_t *)&c_0;
    outputBlocks[2][1] = (uint32_t *)&c_1;
    outputBlocks[2][2] = (uint32_t *)&c_2;
    outputBlocks[2][3] = (uint32_t *)&c_3;
    
    outputBlocks[3][0] = (uint32_t *)&d_0;
    outputBlocks[3][1] = (uint32_t *)&d_1;
    outputBlocks[3][2] = (uint32_t *)&d_2;
    outputBlocks[3][3] = (uint32_t *)&d_3;

    // Constants for MD4
    __m128i AC, AC2;
    AC = _mm_set1_epi32(0x5a827999);
    AC2 = _mm_set1_epi32(0x6ed9eba1);


    // Initial values for a,b,c,d
    mCa = _mm_set1_epi32(Ca);
    mCb = _mm_set1_epi32(Cb);
    mCc = _mm_set1_epi32(Cc);
    mCd = _mm_set1_epi32(Cd);

    // Set up the bitmap mask based on the current bitmap size.
    uint32_t bitmapMask = this->classBitmapLookup_a.size() - 1;

    // Load the vector initial values.
//    printf("Thread %d, hostStartPasswords32 in words: %d\n", threadData.threadNumber, this->HostStartPasswords32.size() / 4);
//    printf("Thread %d, should load values for b0 starting at offset %d\n", threadData.threadNumber, threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH);
//    printf("Thread %d, should load values for b1 starting at offset %d\n", threadData.threadNumber, (1 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH));
//    printf("Thread %d, should load values for b1 starting at offset %d\n", threadData.threadNumber, (2 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH));

    // Init b0-b3 with null in case they are not used.
    b0_0 = b0_1 = b0_2 = b0_3 = _mm_set1_epi32(0);
    b1_0 = b1_1 = b1_2 = b1_3 = _mm_set1_epi32(0);
    b2_0 = b2_1 = b2_2 = b2_3 = _mm_set1_epi32(0);
    b3_0 = b3_1 = b3_2 = b3_3 = _mm_set1_epi32(0);
    b4_0 = b4_1 = b4_2 = b4_3 = _mm_set1_epi32(0);
    b5_0 = b5_1 = b5_2 = b5_3 = _mm_set1_epi32(0);
    b6_0 = b6_1 = b6_2 = b6_3 = _mm_set1_epi32(0);
    b7_0 = b7_1 = b7_2 = b7_3 = _mm_set1_epi32(0);

    // Load b0 based on threadNumber and position 0
    // Load into a temp variable, and spread the data out to the other variables
    // as needed.  This lets us use the standard packed password format.  This is
    // similar to what is being done in the GPU kernels.
    baseOffset = (0 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH);
    a_0 = _mm_set_epi32(initialValues32[baseOffset + 0], initialValues32[baseOffset + 1],
                        initialValues32[baseOffset + 2], initialValues32[baseOffset + 3]);
    a_1 = _mm_set_epi32(initialValues32[baseOffset + 4], initialValues32[baseOffset + 5],
                        initialValues32[baseOffset + 6], initialValues32[baseOffset + 7]);
    a_2 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                        initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);
    a_3 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                        initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);

    b0_0 = (a_0 & mask_ff) | _mm_slli_epi32((a_0 & mask_ff00), 8);
    b1_0 = _mm_srli_epi32((a_0 & mask_ff0000), 16) | _mm_srli_epi32((a_0 & mask_ff000000), 8);

    b0_1 = (a_1 & mask_ff) | _mm_slli_epi32((a_1 & mask_ff00), 8);
    b1_1 = _mm_srli_epi32((a_1 & mask_ff0000), 16) | _mm_srli_epi32((a_1 & mask_ff000000), 8);

    b0_2 = (a_2 & mask_ff) | _mm_slli_epi32((a_2 & mask_ff00), 8);
    b1_2 = _mm_srli_epi32((a_2 & mask_ff0000), 16) | _mm_srli_epi32((a_2 & mask_ff000000), 8);

    b0_3 = (a_3 & mask_ff) | _mm_slli_epi32((a_3 & mask_ff00), 8);
    b1_3 = _mm_srli_epi32((a_3 & mask_ff0000), 16) | _mm_srli_epi32((a_3 & mask_ff000000), 8);


    // If passLen > 3, load b1
    if (this->passwordLength > 3) {
        baseOffset = (1 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH);
        a_0 = _mm_set_epi32(initialValues32[baseOffset + 0], initialValues32[baseOffset + 1],
                            initialValues32[baseOffset + 2], initialValues32[baseOffset + 3]);
        a_1 = _mm_set_epi32(initialValues32[baseOffset + 4], initialValues32[baseOffset + 5],
                            initialValues32[baseOffset + 6], initialValues32[baseOffset + 7]);
        a_2 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                            initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);
        a_3 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                            initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);

        b2_0 = (a_0 & mask_ff) | _mm_slli_epi32((a_0 & mask_ff00), 8);
        b3_0 = _mm_srli_epi32((a_0 & mask_ff0000), 16) | _mm_srli_epi32((a_0 & mask_ff000000), 8);

        b2_1 = (a_1 & mask_ff) | _mm_slli_epi32((a_1 & mask_ff00), 8);
        b3_1 = _mm_srli_epi32((a_1 & mask_ff0000), 16) | _mm_srli_epi32((a_1 & mask_ff000000), 8);

        b2_2 = (a_2 & mask_ff) | _mm_slli_epi32((a_2 & mask_ff00), 8);
        b3_2 = _mm_srli_epi32((a_2 & mask_ff0000), 16) | _mm_srli_epi32((a_2 & mask_ff000000), 8);

        b2_3 = (a_3 & mask_ff) | _mm_slli_epi32((a_3 & mask_ff00), 8);
        b3_3 = _mm_srli_epi32((a_3 & mask_ff0000), 16) | _mm_srli_epi32((a_3 & mask_ff000000), 8);
    }

    // If passLen > 7, load b2
    if (this->passwordLength > 7) {
        baseOffset = (2 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH);
        a_0 = _mm_set_epi32(initialValues32[baseOffset + 0], initialValues32[baseOffset + 1],
                            initialValues32[baseOffset + 2], initialValues32[baseOffset + 3]);
        a_1 = _mm_set_epi32(initialValues32[baseOffset + 4], initialValues32[baseOffset + 5],
                            initialValues32[baseOffset + 6], initialValues32[baseOffset + 7]);
        a_2 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                            initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);
        a_3 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                            initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);

        b4_0 = (a_0 & mask_ff) | _mm_slli_epi32((a_0 & mask_ff00), 8);
        b5_0 = _mm_srli_epi32((a_0 & mask_ff0000), 16) | _mm_srli_epi32((a_0 & mask_ff000000), 8);

        b4_1 = (a_1 & mask_ff) | _mm_slli_epi32((a_1 & mask_ff00), 8);
        b5_1 = _mm_srli_epi32((a_1 & mask_ff0000), 16) | _mm_srli_epi32((a_1 & mask_ff000000), 8);

        b4_2 = (a_2 & mask_ff) | _mm_slli_epi32((a_2 & mask_ff00), 8);
        b5_2 = _mm_srli_epi32((a_2 & mask_ff0000), 16) | _mm_srli_epi32((a_2 & mask_ff000000), 8);

        b4_3 = (a_3 & mask_ff) | _mm_slli_epi32((a_3 & mask_ff00), 8);
        b5_3 = _mm_srli_epi32((a_3 & mask_ff0000), 16) | _mm_srli_epi32((a_3 & mask_ff000000), 8);
}

    // If passLen > 11, load b2
    if (this->passwordLength > 11) {
        baseOffset = (2 * this->TotalKernelWidth) + (threadData.threadNumber * SSE_KERNEL_VECTOR_WIDTH);
        a_0 = _mm_set_epi32(initialValues32[baseOffset + 0], initialValues32[baseOffset + 1],
                            initialValues32[baseOffset + 2], initialValues32[baseOffset + 3]);
        a_1 = _mm_set_epi32(initialValues32[baseOffset + 4], initialValues32[baseOffset + 5],
                            initialValues32[baseOffset + 6], initialValues32[baseOffset + 7]);
        a_2 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                            initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);
        a_3 = _mm_set_epi32(initialValues32[baseOffset + 8], initialValues32[baseOffset + 9],
                            initialValues32[baseOffset + 10], initialValues32[baseOffset + 11]);

        b6_0 = (a_0 & mask_ff) | _mm_slli_epi32((a_0 & mask_ff00), 8);
        b7_0 = _mm_srli_epi32((a_0 & mask_ff0000), 16) | _mm_srli_epi32((a_0 & mask_ff000000), 8);

        b6_1 = (a_1 & mask_ff) | _mm_slli_epi32((a_1 & mask_ff00), 8);
        b7_1 = _mm_srli_epi32((a_1 & mask_ff0000), 16) | _mm_srli_epi32((a_1 & mask_ff000000), 8);

        b6_2 = (a_2 & mask_ff) | _mm_slli_epi32((a_2 & mask_ff00), 8);
        b7_2 = _mm_srli_epi32((a_2 & mask_ff0000), 16) | _mm_srli_epi32((a_2 & mask_ff000000), 8);

        b6_3 = (a_3 & mask_ff) | _mm_slli_epi32((a_3 & mask_ff00), 8);
        b7_3 = _mm_srli_epi32((a_3 & mask_ff0000), 16) | _mm_srli_epi32((a_3 & mask_ff000000), 8);
    }

//    printf("Initial load pass_len %d\n", this->passwordLength);
//    printPassword128NTLM(b0_0, b1_0, b2_0, b3_0, b4_0, b5_0, b6_0, b7_0, this->passwordLength);
//    printPassword128NTLM(b0_1, b1_1, b2_1, b3_1, b4_1, b5_1, b6_1, b7_1, this->passwordLength);
//    printPassword128NTLM(b0_2, b1_2, b2_2, b3_2, b4_2, b5_2, b6_2, b7_2, this->passwordLength);
//    printPassword128NTLM(b0_3, b1_3, b2_3, b3_3, b4_3, b5_3, b6_3, b7_3, this->passwordLength);

    // Set the length in all vectors.
    b14_0 = b14_1 = b14_2 = b14_3 = _mm_set1_epi32(this->passwordLength * 16);

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

        // Perform the main hash function

        // Load the a/b/c/d vars with the initial values.

        a_0 = a_1 = a_2 = a_3 = mCa;
        b_0 = b_1 = b_2 = b_3 = mCb;
        c_0 = c_1 = c_2 = c_3 = mCc;
        d_0 = d_1 = d_2 = d_3 = mCd;

        MD4_STEPS_FIRST2_ROUNDS();
        MD4STEP_ROUND3_4      (a, b, c, d, b0, S31);\
	MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32);\
	MD4STEP_ROUND3_4      (c, d, a, b, b4, S33);\
	MD4STEP_ROUND3_NULL_4 (b, c, d, a, S34);\
	MD4STEP_ROUND3_4      (a, b, c, d, b2, S31);\
	MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32); \
        MD4STEP_ROUND3_4      (c, d, a, b, b6, S33); \
        MD4STEP_ROUND3_4      (b, c, d, a, b14, S34);\
        MD4STEP_ROUND3_4      (a, b, c, d, b1, S31);\
        if (this->passwordLength > 6) { \
            MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32);\
            MD4STEP_ROUND3_4      (c, d, a, b, b5, S33);\
            MD4STEP_ROUND3_NULL_4 (b, c, d, a, S34);\
            MD4STEP_ROUND3_4      (a, b, c, d, b3, S31);\
            if (this->passwordLength > 14) { \
                MD4STEP_ROUND3_NULL_4 (d, a, b, c, S32);\
                MD4STEP_ROUND3_4      (c, d, a, b, b7, S33);\
            } \
        } \

//        a_0 = _mm_add_epi32(a_0,mCa); a_1 = _mm_add_epi32(a_1,mCa);
//        a_2 = _mm_add_epi32(a_2,mCa); a_3 = _mm_add_epi32(a_3,mCa);
//        b_0 = _mm_add_epi32(b_0,mCb); b_1 = _mm_add_epi32(b_1,mCb);
//        b_2 = _mm_add_epi32(b_2,mCb); b_3 = _mm_add_epi32(b_3,mCb);
//        c_0 = _mm_add_epi32(c_0,mCc); c_1 = _mm_add_epi32(c_1,mCc);
//        c_2 = _mm_add_epi32(c_2,mCc); c_3 = _mm_add_epi32(c_3,mCc);
//        d_0 = _mm_add_epi32(d_0,mCd); d_1 = _mm_add_epi32(d_1,mCd);
//        d_2 = _mm_add_epi32(d_2,mCd); d_3 = _mm_add_epi32(d_3,mCd);
//        printPassword128NTLM(b0_0, b1_0, b2_0, b3_0, b4_0, b5_0, b6_0, b7_0, this->passwordLength);
//        printPassword128NTLM(b0_1, b1_1, b2_1, b3_1, b4_1, b5_1, b6_1, b7_1, this->passwordLength);
//        printPassword128NTLM(b0_2, b1_2, b2_2, b3_2, b4_2, b5_2, b6_2, b7_2, this->passwordLength);
//        printPassword128NTLM(b0_3, b1_3, b2_3, b3_3, b4_3, b5_3, b6_3, b7_3, this->passwordLength);

        for (vectorIndex = 0; vectorIndex < 3; vectorIndex++) {
            for (passwordIndex = 0; passwordIndex < 4; passwordIndex++) {
                if ((this->classBitmapLookup_a[(outputBlocks[0][vectorIndex][passwordIndex] >> 3) & bitmapMask] >> (outputBlocks[0][vectorIndex][passwordIndex] & 0x7)) & 0x1) {
                    if (this->globalBitmap128mb_a[(outputBlocks[0][vectorIndex][passwordIndex] >> 3) & 0x07FFFFFF] >> (outputBlocks[0][vectorIndex][passwordIndex] & 0x7) & 0x1) {
                        if (this->globalBitmap128mb_b[(outputBlocks[1][vectorIndex][passwordIndex] >> 3) & 0x07FFFFFF] >> (outputBlocks[1][vectorIndex][passwordIndex] & 0x7) & 0x1) {
                            if (this->globalBitmap128mb_c[(outputBlocks[2][vectorIndex][passwordIndex] >> 3) & 0x07FFFFFF] >> (outputBlocks[2][vectorIndex][passwordIndex] & 0x7) & 0x1) {
                                if (this->globalBitmap128mb_d[(outputBlocks[3][vectorIndex][passwordIndex] >> 3) & 0x07FFFFFF] >> (outputBlocks[3][vectorIndex][passwordIndex] & 0x7) & 0x1) {
//                                    printf("Global bitmap hit: a[%d]\n", 3 - vectorIndex);
//                                    printPassword128NTLM(b0_0, b1_0, b2_0, b3_0, b4_0, b5_0, b6_0, b7_0, this->passwordLength);
//                                    printPassword128NTLM(b0_1, b1_1, b2_1, b3_1, b4_1, b5_1, b6_1, b7_1, this->passwordLength);
//                                    printPassword128NTLM(b0_2, b1_2, b2_2, b3_2, b4_2, b5_2, b6_2, b7_2, this->passwordLength);
//                                    printPassword128NTLM(b0_3, b1_3, b2_3, b3_3, b4_3, b5_3, b6_3, b7_3, this->passwordLength);
//                                    printf("\n\n");
                                    this->checkAndReportHashCpuNTLM(
                                            inputBlocks[0][vectorIndex][passwordIndex],
                                            inputBlocks[1][vectorIndex][passwordIndex],
                                            inputBlocks[2][vectorIndex][passwordIndex],
                                            inputBlocks[3][vectorIndex][passwordIndex],
                                            inputBlocks[4][vectorIndex][passwordIndex],
                                            inputBlocks[5][vectorIndex][passwordIndex],
                                            inputBlocks[6][vectorIndex][passwordIndex],
                                            inputBlocks[7][vectorIndex][passwordIndex],
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
            uint32_t w0, w1, w2, w3, w4, w5, w6, w7;

            for (passwordIndex = 0; passwordIndex < 4; passwordIndex++) {
                // Load the appropriate b0/b1 values.
                w0 = inputBlocks[0][vectorIndex][passwordIndex];
                w1 = inputBlocks[1][vectorIndex][passwordIndex];
                w2 = inputBlocks[2][vectorIndex][passwordIndex];
                w3 = inputBlocks[3][vectorIndex][passwordIndex];
                w4 = inputBlocks[4][vectorIndex][passwordIndex];
                w5 = inputBlocks[5][vectorIndex][passwordIndex];
                w6 = inputBlocks[6][vectorIndex][passwordIndex];
                w7 = inputBlocks[7][vectorIndex][passwordIndex];
                // Increment the appropriate password
                if (this->charsetLengths[1] == 0) {
                    // Single incrementors
                    switch(this->passwordLength) {
                        case 1:
                        makeMFNSingleIncrementorsCPU_NTLM1(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 2:
                        makeMFNSingleIncrementorsCPU_NTLM2(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 3:
                        makeMFNSingleIncrementorsCPU_NTLM3(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 4:
                        makeMFNSingleIncrementorsCPU_NTLM4(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 5:
                        makeMFNSingleIncrementorsCPU_NTLM5(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 6:
                        makeMFNSingleIncrementorsCPU_NTLM6(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 7:
                        makeMFNSingleIncrementorsCPU_NTLM7(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 8:
                        makeMFNSingleIncrementorsCPU_NTLM8(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 9:
                        makeMFNSingleIncrementorsCPU_NTLM9(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 10:
                        makeMFNSingleIncrementorsCPU_NTLM10(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 11:
                        makeMFNSingleIncrementorsCPU_NTLM11(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 12:
                        makeMFNSingleIncrementorsCPU_NTLM12(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 13:
                        makeMFNSingleIncrementorsCPU_NTLM13(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 14:
                        makeMFNSingleIncrementorsCPU_NTLM14(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 15:
                        makeMFNSingleIncrementorsCPU_NTLM15(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                    }
                } else {
                    // Multi incrementors
                    switch(this->passwordLength) {
                        case 1:
                        makeMFNMultipleIncrementorsCPU_NTLM1(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 2:
                        makeMFNMultipleIncrementorsCPU_NTLM2(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 3:
                        makeMFNMultipleIncrementorsCPU_NTLM3(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 4:
                        makeMFNMultipleIncrementorsCPU_NTLM4(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 5:
                        makeMFNMultipleIncrementorsCPU_NTLM5(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 6:
                        makeMFNMultipleIncrementorsCPU_NTLM6(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 7:
                        makeMFNMultipleIncrementorsCPU_NTLM7(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 8:
                        makeMFNMultipleIncrementorsCPU_NTLM8(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 9:
                        makeMFNMultipleIncrementorsCPU_NTLM9(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 10:
                        makeMFNMultipleIncrementorsCPU_NTLM10(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 11:
                        makeMFNMultipleIncrementorsCPU_NTLM11(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 12:
                        makeMFNMultipleIncrementorsCPU_NTLM12(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 13:
                        makeMFNMultipleIncrementorsCPU_NTLM13(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 14:
                        makeMFNMultipleIncrementorsCPU_NTLM14(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                        case 15:
                        makeMFNMultipleIncrementorsCPU_NTLM15(this->charsetForwardLookup, this->charsetReverseLookup, this->charsetLengths); break;
                    }
                }
                // Store the password back in the right location in the vector.
                inputBlocks[0][vectorIndex][passwordIndex] = w0;
                inputBlocks[1][vectorIndex][passwordIndex] = w1;
                inputBlocks[2][vectorIndex][passwordIndex] = w2;
                inputBlocks[3][vectorIndex][passwordIndex] = w3;
                inputBlocks[4][vectorIndex][passwordIndex] = w4;
                inputBlocks[5][vectorIndex][passwordIndex] = w5;
                inputBlocks[6][vectorIndex][passwordIndex] = w6;
                inputBlocks[7][vectorIndex][passwordIndex] = w7;
            }
        }

//        printf("Post-increment: \n");
//        printPassword128(b0_0, b1_0, b2_0, b3_0, this->passwordLength);

    }
}

