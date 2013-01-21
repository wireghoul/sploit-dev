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
__device__ __constant__ unsigned char MD5_Candidate_Device_Charset_Constant[512]; // Constant space for charset
__device__ __constant__ uint32_t MD5_Candidate_Device_Charset_Length; // Character set length
__device__ __constant__ uint32_t MD5_Candidate_Device_Chain_Length; // May as well pull it from constant memory... faster when cached.
__device__ __constant__ uint32_t MD5_Candidate_Device_Table_Index;
__device__ __constant__ uint32_t MD5_Candidate_Device_Number_Of_Threads; // It needs this, and can't easily calculate it

// 4 32-byte words for MD5 hashes
__device__ __constant__ uint32_t MD5_Candidate_Device_Hash[4];


#include "../../inc/CUDA_Common/CUDA_MD5.h"
#include "../../inc/CUDA_Common/Hash_Common.h"
#include "../../inc/GRT_CUDA_device/CUDA_Reduction_Functions.h"

// Copy the shared variables to the host
extern "C" void copyMD5CandidateDataToConstant(char *hostCharset, uint32_t hostCharsetLength,
        uint32_t hostChainLength, uint32_t hostTableIndex, uint32_t hostNumberOfThreads) {

    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD5_Candidate_Device_Charset_Constant, hostCharset, 512));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD5_Candidate_Device_Charset_Length, &hostCharsetLength, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD5_Candidate_Device_Chain_Length, &hostChainLength, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD5_Candidate_Device_Table_Index, &hostTableIndex, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD5_Candidate_Device_Number_Of_Threads, &hostNumberOfThreads, sizeof(uint32_t)));
}


extern "C" void copyMD5HashDataToConstant(unsigned char *hash) {
    // Yes, I'm copying into a uint32_t array from an unsigned char array.  This works, though, and it makes
    // my life easier.
    // For fuck's sake, Bitweasil, copy the HASH, not the address of the hash!
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD5_Candidate_Device_Hash, hash, 16 * sizeof(unsigned char)));
}


/*
__global__ void GenerateMD5CH10(unsigned char *CandidateHashes, uint32_t ThreadSpaceOffset, uint32_t StartStep, uint32_t StepsToRun) {

    uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
    uint32_t a,b,c,d;

    uint32_t *InitialArray32;
    uint32_t *OutputArray32;
    // 32-bit accesses to the hash arrays
    InitialArray32 = (uint32_t *)MD5_Candidate_Device_Hash;
    OutputArray32 = (uint32_t *)CandidateHashes;

    uint32_t i, chain_index, step_to_calculate, charset_offset, last_step_for_iteration;

    const int pass_length = 10;

    __shared__ char charset[512];

    // Generic "copy charset to shared memory" function
    //copySingleCharsetToShared(charset);
    copySingleCharsetToShared(charset, MD5_Candidate_Device_Charset_Constant);
    // Figure out which chain we are working on.
    chain_index = ((blockIdx.x*blockDim.x + threadIdx.x) + (ThreadSpaceOffset * MD5_Candidate_Device_Number_Of_Threads));

    // Find out if we're done with work.
    // If our index + the startstep is greater than the chain length, this thread has nothing to do.
    if ((chain_index + StartStep) > MD5_Candidate_Device_Chain_Length) {
        return;
    }

    // Load the initial hash.  This will either be from the constant or from the storage space.

    if (StartStep == 0) {
        a = InitialArray32[0];
        b = InitialArray32[1];
        c = InitialArray32[2];
        d = InitialArray32[3];
    } else {
        a = OutputArray32[0 * MD5_Candidate_Device_Chain_Length + chain_index];
        b = OutputArray32[1 * MD5_Candidate_Device_Chain_Length + chain_index];
        c = OutputArray32[2 * MD5_Candidate_Device_Chain_Length + chain_index];
        d = OutputArray32[3 * MD5_Candidate_Device_Chain_Length + chain_index];
    }

    // Figure out which step we're running.
    step_to_calculate = chain_index + StartStep;

    // Yes, modulus here is slow.  And this is a less-critical chunk of code.  So it stays here.
    charset_offset = step_to_calculate % MD5_Candidate_Device_Charset_Length;

    clearB0toB15(b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);

    reduceSingleCharsetNormal(b0, b1, b2, a, b, c, d, step_to_calculate, charset, charset_offset, pass_length, MD5_Candidate_Device_Table_Index);

    step_to_calculate++;
    charset_offset++;
    if (charset_offset >= MD5_Candidate_Device_Charset_Length) {
        charset_offset = 0;
    }
    // Figure out the last step to run - either the chain length or
    // the number of specified steps.
    if ((step_to_calculate + StepsToRun) > MD5_Candidate_Device_Chain_Length) {
        last_step_for_iteration = MD5_Candidate_Device_Chain_Length - 1;
    } else {
        last_step_for_iteration = (step_to_calculate + StepsToRun - 1); // Already run one
    }


    // We now have our (step+1) charset.
    for (i = step_to_calculate; i <= last_step_for_iteration; i++) {
        padMDHash(pass_length, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        CUDA_MD5(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d);
        reduceSingleCharsetNormal(b0, b1, b2, a, b, c, d, i, charset, charset_offset, pass_length, MD5_Candidate_Device_Table_Index);
        charset_offset++;
        if (charset_offset >= MD5_Candidate_Device_Charset_Length) {
            charset_offset = 0;
        }
    }
    // Store the hash output.
    OutputArray32[0 * MD5_Candidate_Device_Chain_Length + chain_index] = a;
    OutputArray32[1 * MD5_Candidate_Device_Chain_Length + chain_index] = b;
    OutputArray32[2 * MD5_Candidate_Device_Chain_Length + chain_index] = c;
    OutputArray32[3 * MD5_Candidate_Device_Chain_Length + chain_index] = d;
}*/


#define CREATE_MD5_CH_KERNEL(length) \
__global__ void GenerateMD5CH##length(unsigned char *CandidateHashes, uint32_t ThreadSpaceOffset, uint32_t StartStep, uint32_t StepsToRun) { \
    uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
    uint32_t a,b,c,d; \
    uint32_t *InitialArray32, *OutputArray32; \
    InitialArray32 = (uint32_t *)MD5_Candidate_Device_Hash; \
    OutputArray32 = (uint32_t *)CandidateHashes; \
    uint32_t i, chain_index, step_to_calculate, charset_offset, last_step_for_iteration; \
    const int pass_length = length; \
    __shared__ char charset[512]; \
    copySingleCharsetToShared(charset, MD5_Candidate_Device_Charset_Constant); \
    chain_index = ((blockIdx.x*blockDim.x + threadIdx.x) + (ThreadSpaceOffset * MD5_Candidate_Device_Number_Of_Threads)); \
    if ((chain_index + StartStep) > MD5_Candidate_Device_Chain_Length) { \
        return; \
    } \
    if (StartStep == 0) { \
        a = InitialArray32[0]; \
        b = InitialArray32[1]; \
        c = InitialArray32[2]; \
        d = InitialArray32[3]; \
    } else { \
        a = OutputArray32[0 * MD5_Candidate_Device_Chain_Length + chain_index]; \
        b = OutputArray32[1 * MD5_Candidate_Device_Chain_Length + chain_index]; \
        c = OutputArray32[2 * MD5_Candidate_Device_Chain_Length + chain_index]; \
        d = OutputArray32[3 * MD5_Candidate_Device_Chain_Length + chain_index]; \
    } \
    step_to_calculate = chain_index + StartStep; \
    charset_offset = step_to_calculate % MD5_Candidate_Device_Charset_Length; \
    clearB0toB15(b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    reduceSingleCharsetNormal(b0, b1, b2, a, b, c, d, step_to_calculate, charset, charset_offset, pass_length, MD5_Candidate_Device_Table_Index); \
    step_to_calculate++; \
    charset_offset++; \
    if (charset_offset >= MD5_Candidate_Device_Charset_Length) { \
        charset_offset = 0; \
    } \
    if ((step_to_calculate + StepsToRun) > MD5_Candidate_Device_Chain_Length) { \
        last_step_for_iteration = MD5_Candidate_Device_Chain_Length - 1; \
    } else { \
        last_step_for_iteration = (step_to_calculate + StepsToRun - 1); \
    } \
    for (i = step_to_calculate; i <= last_step_for_iteration; i++) { \
        padMDHash(pass_length, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        CUDA_MD5(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d); \
        reduceSingleCharsetNormal(b0, b1, b2, a, b, c, d, i, charset, charset_offset, pass_length, MD5_Candidate_Device_Table_Index); \
        charset_offset++; \
        if (charset_offset >= MD5_Candidate_Device_Charset_Length) { \
            charset_offset = 0; \
        } \
    } \
    OutputArray32[0 * MD5_Candidate_Device_Chain_Length + chain_index] = a; \
    OutputArray32[1 * MD5_Candidate_Device_Chain_Length + chain_index] = b; \
    OutputArray32[2 * MD5_Candidate_Device_Chain_Length + chain_index] = c; \
    OutputArray32[3 * MD5_Candidate_Device_Chain_Length + chain_index] = d; \
} 



CREATE_MD5_CH_KERNEL(6)
CREATE_MD5_CH_KERNEL(7)
CREATE_MD5_CH_KERNEL(8)
CREATE_MD5_CH_KERNEL(9)
CREATE_MD5_CH_KERNEL(10)


extern "C" void LaunchMD5CandidateHashKernel(int PasswordLength, int CUDA_Blocks, int CUDA_Threads,
        unsigned char *DEVICE_End_Hashes, uint32_t ThreadSpaceOffset, uint32_t StartStep, uint32_t StepsToRun) {

    switch (PasswordLength) {
        case 6:
            GenerateMD5CH6 <<< CUDA_Blocks, CUDA_Threads >>>
                (DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
            break;
        case 7:
            GenerateMD5CH7 <<< CUDA_Blocks, CUDA_Threads >>>
                (DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
            break;
        case 8:
            GenerateMD5CH8 <<< CUDA_Blocks, CUDA_Threads >>>
                (DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
            break;
        case 9:
            GenerateMD5CH9 <<< CUDA_Blocks, CUDA_Threads >>>
                (DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
            break;
        case 10:
            GenerateMD5CH10 <<< CUDA_Blocks, CUDA_Threads >>>
                (DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
            break;
        default:
            printf("Password length %d not supported!", PasswordLength);
            exit(1);
    }
}
