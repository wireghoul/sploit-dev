/*
Cryptohaze GPU Rainbow Tables
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

// CUDA NTLM kernels for table generation.

// This is here so Netbeans doesn't error-spam my IDE
#if !defined(__CUDACC__)
    // define the keywords, so that the IDE does not complain about them
    #define __global__
    #define __device__
    #define __shared__
    #define __constant__
    #define blockIdx.x 1
    #define blockDim.x 1
    #define threadIdx.x 1
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "GRT_Common/GRTCommon.h"
#include <stdio.h>


// Some CUDA variables
__device__ __constant__ unsigned char NTLM_Generate_Device_Charset_Constant[512]; // Constant space for charset
__device__ __constant__ uint32_t NTLM_Generate_Device_Charset_Length; // Character set length
__device__ __constant__ uint32_t NTLM_Generate_Device_Chain_Length; // May as well pull it from constant memory... faster.
__device__ __constant__ uint32_t NTLM_Generate_Device_Number_Of_Chains; // Same, may as well be constant.
__device__ __constant__ uint32_t NTLM_Generate_Device_Table_Index;
__device__ __constant__ uint32_t NTLM_Generate_Device_Number_Of_Threads; // It needs this, and can't easily calculate it


#include "../../inc/CUDA_Common/CUDA_MD4.h"
#include "../../inc/CUDA_Common/Hash_Common.h"
#include "../../inc/GRT_CUDA_device/CUDA_Reduction_Functions.h"
#include "../../inc/GRT_CUDA_device/CUDA_Load_Store_Registers.h"


/*
__global__ void MakeNTLMChainLen6(unsigned char *InitialPasswordArray, unsigned char *OutputHashArray,
    uint32_t PasswordSpaceOffset, uint32_t StartChainIndex, uint32_t StepsToRun, uint32_t charset_offset) {

    const int pass_length = 6;

    // Needed variables for generation
    uint32_t CurrentStep, PassCount, password_index;

    // Hash variables
    uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
    uint32_t a,b,c,d;

    // Word-width access to the arrays
    uint32_t *InitialArray32;
    uint32_t *OutputArray32;
    // 32-bit accesses to the hash arrays
    InitialArray32 = (uint32_t *)InitialPasswordArray;
    OutputArray32 = (uint32_t *)OutputHashArray;


    __shared__ char charset[512];

    // Generic "copy charset to shared memory" function
    copySingleCharsetToShared(charset, NTLM_Generate_Device_Charset_Constant);

    // Figure out which password we are working on.
    password_index = ((blockIdx.x*blockDim.x + threadIdx.x) + (PasswordSpaceOffset * NTLM_Generate_Device_Number_Of_Threads));

    // Return if this thread is working on something beyond the end of the password space
    if (password_index >= NTLM_Generate_Device_Number_Of_Chains) {
        return;
    }

    clearB0toB15(b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
    LoadNTLMRegistersFromGlobalMemory(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,
        InitialArray32, NTLM_Generate_Device_Number_Of_Chains, password_index, pass_length);

    for (PassCount = 0; PassCount < StepsToRun; PassCount++) {
        CurrentStep = PassCount + StartChainIndex;

        padMDHash(pass_length * 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        CUDA_MD4(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d);
        reduceSingleCharsetNTLM(b0, b1, b2, b3, b4, a, b, c, d, CurrentStep, charset, charset_offset, pass_length, NTLM_Generate_Device_Table_Index);

        charset_offset++;
        if (charset_offset >= NTLM_Generate_Device_Charset_Length) {
            charset_offset = 0;
        }
    }
    // Done with the number of steps we need to run

    // If we are done (or have somehow overflowed), store the result
    if (CurrentStep >= (NTLM_Generate_Device_Chain_Length - 1)) {
        OutputArray32[0 * NTLM_Generate_Device_Number_Of_Chains + password_index] = a;
        OutputArray32[1 * NTLM_Generate_Device_Number_Of_Chains + password_index] = b;
        OutputArray32[2 * NTLM_Generate_Device_Number_Of_Chains + password_index] = c;
        OutputArray32[3 * NTLM_Generate_Device_Number_Of_Chains + password_index] = d;
    }
    // Else, store the b0/b1 values back to the initial array for the next loop
    else {
        SaveNTLMRegistersIntoGlobalMemory(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,
            InitialArray32, NTLM_Generate_Device_Number_Of_Chains, password_index, pass_length);
    }
}
*/

#define CREATE_NTLM_GEN_KERNEL(length) \
__global__ void MakeNTLMChainLen##length(unsigned char *InitialPasswordArray, unsigned char *OutputHashArray, \
    uint32_t PasswordSpaceOffset, uint32_t StartChainIndex, uint32_t StepsToRun, uint32_t charset_offset) { \
    const int pass_length = length; \
    uint32_t CurrentStep, PassCount, password_index; \
    uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
    uint32_t a,b,c,d; \
    uint32_t *InitialArray32; \
    uint32_t *OutputArray32; \
    InitialArray32 = (uint32_t *)InitialPasswordArray; \
    OutputArray32 = (uint32_t *)OutputHashArray; \
    __shared__ char charset[512]; \
    copySingleCharsetToShared(charset, NTLM_Generate_Device_Charset_Constant); \
    password_index = ((blockIdx.x*blockDim.x + threadIdx.x) + (PasswordSpaceOffset * NTLM_Generate_Device_Number_Of_Threads)); \
    if (password_index >= NTLM_Generate_Device_Number_Of_Chains) { \
        return; \
    } \
    clearB0toB15(b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    LoadNTLMRegistersFromGlobalMemory(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15, \
        InitialArray32, NTLM_Generate_Device_Number_Of_Chains, password_index, pass_length); \
    for (PassCount = 0; PassCount < StepsToRun; PassCount++) { \
        CurrentStep = PassCount + StartChainIndex; \
        padMDHash(pass_length * 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        CUDA_MD4(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d); \
        reduceSingleCharsetNTLM(b0, b1, b2, b3, b4, a, b, c, d, CurrentStep, charset, charset_offset, pass_length, NTLM_Generate_Device_Table_Index); \
        charset_offset++; \
        if (charset_offset >= NTLM_Generate_Device_Charset_Length) { \
            charset_offset = 0; \
        } \
    } \
    if (CurrentStep >= (NTLM_Generate_Device_Chain_Length - 1)) { \
        OutputArray32[0 * NTLM_Generate_Device_Number_Of_Chains + password_index] = a; \
        OutputArray32[1 * NTLM_Generate_Device_Number_Of_Chains + password_index] = b; \
        OutputArray32[2 * NTLM_Generate_Device_Number_Of_Chains + password_index] = c; \
        OutputArray32[3 * NTLM_Generate_Device_Number_Of_Chains + password_index] = d; \
    } \
    else { \
        SaveNTLMRegistersIntoGlobalMemory(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15, \
            InitialArray32, NTLM_Generate_Device_Number_Of_Chains, password_index, pass_length); \
    } \
}


CREATE_NTLM_GEN_KERNEL(6)
CREATE_NTLM_GEN_KERNEL(7)
CREATE_NTLM_GEN_KERNEL(8)
CREATE_NTLM_GEN_KERNEL(9)
CREATE_NTLM_GEN_KERNEL(10)


extern "C" void LaunchGenerateKernelNTLM(int passwordLength, uint32_t CUDA_Blocks,
        uint32_t CUDA_Threads, unsigned char *DEVICE_Initial_Passwords,
        unsigned char *DEVICE_End_Hashes, uint32_t PasswordSpaceOffset,
        uint32_t CurrentChainStartOffset, uint32_t StepsPerInvocation, uint32_t CharsetOffset) {
    switch (passwordLength) {
            case 6:
                MakeNTLMChainLen6 <<< CUDA_Blocks, CUDA_Threads >>>
                    (DEVICE_Initial_Passwords, DEVICE_End_Hashes, PasswordSpaceOffset,
                    CurrentChainStartOffset, StepsPerInvocation, CharsetOffset);
                break;
            case 7:
                MakeNTLMChainLen7 <<< CUDA_Blocks, CUDA_Threads >>>
                    (DEVICE_Initial_Passwords, DEVICE_End_Hashes, PasswordSpaceOffset,
                    CurrentChainStartOffset, StepsPerInvocation, CharsetOffset);
                break;
            case 8:
                MakeNTLMChainLen8 <<< CUDA_Blocks, CUDA_Threads >>>
                    (DEVICE_Initial_Passwords, DEVICE_End_Hashes, PasswordSpaceOffset,
                    CurrentChainStartOffset, StepsPerInvocation, CharsetOffset);
                break;
            case 9:
                MakeNTLMChainLen9 <<< CUDA_Blocks, CUDA_Threads >>>
                    (DEVICE_Initial_Passwords, DEVICE_End_Hashes, PasswordSpaceOffset,
                    CurrentChainStartOffset, StepsPerInvocation, CharsetOffset);
                break;
            case 10:
                MakeNTLMChainLen10 <<< CUDA_Blocks, CUDA_Threads >>>
                    (DEVICE_Initial_Passwords, DEVICE_End_Hashes, PasswordSpaceOffset,
                    CurrentChainStartOffset, StepsPerInvocation, CharsetOffset);
                break;
            default:
                printf("Password length %d not supported!", passwordLength);
                exit(1);
        }
}

extern "C" void copyConstantsToNTLM(unsigned char *HOST_Charset, uint32_t HOST_Charset_Length,
    uint32_t HOST_Chain_Length, uint32_t HOST_Number_Of_Chains, uint32_t HOST_Table_Index,
    uint32_t HOST_Number_Of_Threads) {

    cudaMemcpyToSymbol(NTLM_Generate_Device_Charset_Constant,HOST_Charset, 512);
    cudaMemcpyToSymbol(NTLM_Generate_Device_Charset_Length, &HOST_Charset_Length, sizeof(uint32_t));

    // Copy general table parameters to constant space
    cudaMemcpyToSymbol(NTLM_Generate_Device_Chain_Length, &HOST_Chain_Length, sizeof(uint32_t));
    cudaMemcpyToSymbol(NTLM_Generate_Device_Number_Of_Chains, &HOST_Number_Of_Chains, sizeof(uint32_t));
    cudaMemcpyToSymbol(NTLM_Generate_Device_Table_Index, &HOST_Table_Index, sizeof(uint32_t));
    cudaMemcpyToSymbol(NTLM_Generate_Device_Number_Of_Threads, &HOST_Number_Of_Threads, sizeof(HOST_Number_Of_Threads));
}
