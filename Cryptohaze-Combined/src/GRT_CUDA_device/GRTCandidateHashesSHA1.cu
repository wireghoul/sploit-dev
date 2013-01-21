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
#include "CUDA_Common/CUDA_SAFE_CALL.h"



// These will be the same for all GPUs working on a hash.
__device__ __constant__ unsigned char SHA1_Candidate_Device_Charset_Constant[512]; // Constant space for charset
__device__ __constant__ uint32_t SHA1_Candidate_Device_Charset_Length; // Character set length
__device__ __constant__ uint32_t SHA1_Candidate_Device_Chain_Length; // May as well pull it from constant memory... faster when cached.
__device__ __constant__ uint32_t SHA1_Candidate_Device_Table_Index;
__device__ __constant__ uint32_t SHA1_Candidate_Device_Number_Of_Threads; // It needs this, and can't easily calculate it

// 4 32-byte words for SHA1 hashes
__device__ __constant__ uint32_t SHA1_Candidate_Device_Hash[5];


#include "../../inc/CUDA_Common/CUDA_SHA1.h"
#include "../../inc/CUDA_Common/Hash_Common.h"
#include "../../inc/GRT_CUDA_device/CUDA_Reduction_Functions.h"

// Copy the shared variables to the host
extern "C" void copySHA1CandidateDataToConstant(char *hostCharset, uint32_t hostCharsetLength,
        uint32_t hostChainLength, uint32_t hostTableIndex, uint32_t hostNumberOfThreads) {

    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Candidate_Device_Charset_Constant, hostCharset, 512));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Candidate_Device_Charset_Length, &hostCharsetLength, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Candidate_Device_Chain_Length, &hostChainLength, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Candidate_Device_Table_Index, &hostTableIndex, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Candidate_Device_Number_Of_Threads, &hostNumberOfThreads, sizeof(uint32_t)));
}


extern "C" void copySHA1HashDataToConstant(unsigned char *hash) {
    // Yes, I'm copying into a uint32_t array from an unsigned char array.  This works, though, and it makes
    // my life easier.
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Candidate_Device_Hash, hash, 20 * sizeof(unsigned char)));
}


#define CREATE_SHA1_CH_KERNEL(length) \
__global__ void GenerateSHA1CH##length(unsigned char *CandidateHashes, uint32_t ThreadSpaceOffset, uint32_t StartStep, uint32_t StepsToRun) { \
    const int pass_length = length; \
    uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
    uint32_t a,b,c,d,e; \
    uint32_t *InitialArray32; \
    uint32_t *OutputArray32; \
    InitialArray32 = (uint32_t *)SHA1_Candidate_Device_Hash; \
    OutputArray32 = (uint32_t *)CandidateHashes; \
    uint32_t i, chain_index, step_to_calculate, charset_offset, last_step_for_iteration; \
    __shared__ char charset[512]; \
    copySingleCharsetToShared(charset, SHA1_Candidate_Device_Charset_Constant); \
    chain_index = ((blockIdx.x*blockDim.x + threadIdx.x) + (ThreadSpaceOffset * SHA1_Candidate_Device_Number_Of_Threads)); \
    if ((chain_index + StartStep) > SHA1_Candidate_Device_Chain_Length) { \
        return; \
    } \
    if (StartStep == 0) { \
        a = InitialArray32[0]; \
        b = InitialArray32[1]; \
        c = InitialArray32[2]; \
        d = InitialArray32[3]; \
        e = InitialArray32[4]; \
    } else { \
        a = OutputArray32[0 * SHA1_Candidate_Device_Chain_Length + chain_index]; \
        b = OutputArray32[1 * SHA1_Candidate_Device_Chain_Length + chain_index]; \
        c = OutputArray32[2 * SHA1_Candidate_Device_Chain_Length + chain_index]; \
        d = OutputArray32[3 * SHA1_Candidate_Device_Chain_Length + chain_index]; \
        e = OutputArray32[4 * SHA1_Candidate_Device_Chain_Length + chain_index]; \
    } \
    step_to_calculate = chain_index + StartStep; \
    charset_offset = step_to_calculate % SHA1_Candidate_Device_Charset_Length; \
    clearB0toB15(b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    reduceSingleCharsetNormal(b0, b1, b2, a, b, c, d, step_to_calculate, charset, charset_offset, pass_length, SHA1_Candidate_Device_Table_Index); \
    step_to_calculate++; \
    charset_offset++; \
    if (charset_offset >= SHA1_Candidate_Device_Charset_Length) { \
        charset_offset = 0; \
    } \
    if ((step_to_calculate + StepsToRun) > SHA1_Candidate_Device_Chain_Length) { \
        last_step_for_iteration = SHA1_Candidate_Device_Chain_Length - 1; \
    } else { \
        last_step_for_iteration = (step_to_calculate + StepsToRun - 1); \
    } \
    for (i = step_to_calculate; i <= last_step_for_iteration; i++) { \
        b15 = ((pass_length * 8) & 0xff) << 24 | (((pass_length * 8) >> 8) & 0xff) << 16; \
        SetCharacterAtPosition(0x80, pass_length, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 ); \
        SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        a = reverse(a);b = reverse(b);c = reverse(c);d = reverse(d);e = reverse(e); \
        clearB0toB15(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15); \
        reduceSingleCharsetNormal(b0, b1, b2, a, b, c, d, i, charset, charset_offset, pass_length, SHA1_Candidate_Device_Table_Index); \
        charset_offset++; \
        if (charset_offset >= SHA1_Candidate_Device_Charset_Length) { \
            charset_offset = 0; \
        } \
    } \
    OutputArray32[0 * SHA1_Candidate_Device_Chain_Length + chain_index] = a; \
    OutputArray32[1 * SHA1_Candidate_Device_Chain_Length + chain_index] = b; \
    OutputArray32[2 * SHA1_Candidate_Device_Chain_Length + chain_index] = c; \
    OutputArray32[3 * SHA1_Candidate_Device_Chain_Length + chain_index] = d; \
    OutputArray32[4 * SHA1_Candidate_Device_Chain_Length + chain_index] = e; \
}

CREATE_SHA1_CH_KERNEL(6)
CREATE_SHA1_CH_KERNEL(7)
CREATE_SHA1_CH_KERNEL(8)
CREATE_SHA1_CH_KERNEL(9)
CREATE_SHA1_CH_KERNEL(10)


extern "C" void LaunchSHA1CandidateHashKernel(int PasswordLength, int CUDA_Blocks, int CUDA_Threads,
        unsigned char *DEVICE_End_Hashes, uint32_t ThreadSpaceOffset, uint32_t StartStep, uint32_t StepsToRun) {

    switch (PasswordLength) {
        case 6:
            GenerateSHA1CH6 <<< CUDA_Blocks, CUDA_Threads >>>
                (DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
            break;
        case 7:
            GenerateSHA1CH7 <<< CUDA_Blocks, CUDA_Threads >>>
                (DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
            break;
        case 8:
            GenerateSHA1CH8 <<< CUDA_Blocks, CUDA_Threads >>>
                (DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
            break;
        case 9:
            GenerateSHA1CH9 <<< CUDA_Blocks, CUDA_Threads >>>
                (DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
            break;
        case 10:
            GenerateSHA1CH10 <<< CUDA_Blocks, CUDA_Threads >>>
                (DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
            break;
        default:
            printf("Password length %d not supported!", PasswordLength);
            exit(1);
    }
}
