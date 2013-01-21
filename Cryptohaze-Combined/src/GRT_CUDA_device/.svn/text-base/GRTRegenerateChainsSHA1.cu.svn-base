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
__device__ __constant__ unsigned char SHA1_Regenerate_Device_Charset_Constant[512]; // Constant space for charset
__device__ __constant__ uint32_t SHA1_Regenerate_Device_Charset_Length; // Character set length
__device__ __constant__ uint32_t SHA1_Regenerate_Device_Chain_Length; // May as well pull it from constant memory... faster when cached.
__device__ __constant__ uint32_t SHA1_Regenerate_Device_Table_Index;
__device__ __constant__ uint32_t SHA1_Regenerate_Device_Number_Of_Threads; // It needs this, and can't easily calculate it
__device__ __constant__ uint32_t SHA1_Regenerate_Device_Number_Of_Chains_To_Regen;
__device__ __constant__ uint32_t SHA1_Regenerate_Device_Number_Of_Hashes;
__device__ __constant__ unsigned char constantBitmap[8192]; // for lookups



#include "../../inc/CUDA_Common/CUDA_SHA1.h"
#include "../../inc/CUDA_Common/Hash_Common.h"
#include "../../inc/GRT_CUDA_device/CUDA_Reduction_Functions.h"
#include "../../inc/GRT_CUDA_device/CUDA_Load_Store_Registers.h"

// Copy the shared variables to the host
extern "C" void copySHA1RegenerateDataToConstant(char *hostCharset, uint32_t hostCharsetLength,
        uint32_t hostChainLength, uint32_t hostTableIndex, uint32_t hostNumberOfThreads, unsigned char *hostBitmap,
        uint32_t hostNumberOfHashes) {
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Regenerate_Device_Charset_Constant, hostCharset, 512));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Regenerate_Device_Charset_Length, &hostCharsetLength, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Regenerate_Device_Chain_Length, &hostChainLength, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Regenerate_Device_Table_Index, &hostTableIndex, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Regenerate_Device_Number_Of_Threads, &hostNumberOfThreads, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(constantBitmap, hostBitmap, 8192));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Regenerate_Device_Number_Of_Hashes, &hostNumberOfHashes, sizeof(uint32_t)));
}

extern "C" void setSHA1RegenerateNumberOfChains(uint32_t numberOfChains) {
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(SHA1_Regenerate_Device_Number_Of_Chains_To_Regen, &numberOfChains, sizeof(uint32_t)));
}



__device__ inline void copyBitmap(unsigned char *sharedBitmap) {
  uint64_t *sharedBitmapCoalesce = (uint64_t *)sharedBitmap;
  uint64_t *deviceBitmapCoalesce = (uint64_t *)constantBitmap;

  int a;

  if (threadIdx.x == 0) {
      for (a = 0; a < (8192 / 8); a++) {
          sharedBitmapCoalesce[a] = deviceBitmapCoalesce[a];
      }
  }
  // Make sure everyone is here and done before we return.
  syncthreads();
}



#define CREATE_SHA1_REGEN_KERNEL(length) \
__global__ void RegenSHA1ChainLen##length(unsigned char *InitialPasswordArray, unsigned char *FoundPasswordArray, \
        unsigned char *DeviceHashArray, uint32_t PasswordSpaceOffset, uint32_t StartChainIndex, \
        uint32_t StepsToRun, uint32_t charset_offset, unsigned char *successArray) { \
    uint32_t CurrentStep, PassCount, password_index; \
    const int pass_length = length; \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
    uint32_t passb0, passb1, passb2, passb3; \
    uint32_t a, b, c, d, e; \
    uint32_t *InitialArray32; \
    uint32_t *DEVICE_Hashes_32; \
    uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp; \
    InitialArray32 = (uint32_t *) InitialPasswordArray; \
    DEVICE_Hashes_32 = (uint32_t *) DeviceHashArray; \
    __shared__ char charset[512]; \
    __shared__ __align__(16) unsigned char sharedBitmap[8192]; \
    copySingleCharsetToShared(charset, SHA1_Regenerate_Device_Charset_Constant); \
    copyBitmap(sharedBitmap); \
    password_index = ((blockIdx.x * blockDim.x + threadIdx.x) + (PasswordSpaceOffset * SHA1_Regenerate_Device_Number_Of_Threads)); \
    if (password_index >= SHA1_Regenerate_Device_Number_Of_Chains_To_Regen) { \
        return; \
    } \
    clearB0toB15(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    LoadMD5RegistersFromGlobalMemory(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15, \
        InitialArray32, SHA1_Regenerate_Device_Number_Of_Chains_To_Regen, password_index, pass_length); \
    for (PassCount = 0; PassCount < StepsToRun; PassCount++) { \
        CurrentStep = PassCount + StartChainIndex; \
        b15 = ((pass_length * 8) & 0xff) << 24 | (((pass_length * 8) >> 8) & 0xff) << 16; \
        SetCharacterAtPosition(0x80, pass_length, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 ); \
        passb0 = b0; passb1 = b1; passb2 = b2; passb3 = b3; \
        SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        a = reverse(a);b = reverse(b);c = reverse(c);d = reverse(d);e = reverse(e); \
        if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
            { \
                search_high = SHA1_Regenerate_Device_Number_Of_Hashes; \
                search_low = 0; \
                search_index = 0; \
                while (search_low < search_high) { \
                    search_index = search_low + (search_high - search_low) / 2; \
                    temp = DEVICE_Hashes_32[5 * search_index]; \
                    hash_order_mem = (temp & 0xff) << 24 | ((temp >> 8) & 0xff) << 16 | ((temp >> 16) & 0xff) << 8 | ((temp >> 24) & 0xff); \
                    hash_order_a = (a & 0xff) << 24 | ((a >> 8) & 0xff) << 16 | ((a >> 16) & 0xff) << 8 | ((a >> 24) & 0xff); \
                    if (hash_order_mem < hash_order_a) { \
                        search_low = search_index + 1; \
                    } else { \
                        search_high = search_index; \
                    } \
                    if ((hash_order_a == hash_order_mem) && (search_low < SHA1_Regenerate_Device_Number_Of_Hashes)) { \
                        break; \
                    } \
                } \
                if (hash_order_a != hash_order_mem) { \
                    goto next; \
                } \
                while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 5])) { \
                    search_index--; \
                } \
                while ((a == DEVICE_Hashes_32[search_index * 5])) { \
                    { \
                        if (b == DEVICE_Hashes_32[search_index * 5 + 1]) { \
                            if (c == DEVICE_Hashes_32[search_index * 5 + 2]) { \
                                if (d == DEVICE_Hashes_32[search_index * 5 + 3]) { \
                                    if (pass_length >= 1) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 0] = (passb0 >> 0) & 0xff; \
                                    if (pass_length >= 2) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 1] = (passb0 >> 8) & 0xff; \
                                    if (pass_length >= 3) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 2] = (passb0 >> 16) & 0xff; \
                                    if (pass_length >= 4) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 3] = (passb0 >> 24) & 0xff; \
                                    if (pass_length >= 5) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 4] = (passb1 >> 0) & 0xff; \
                                    if (pass_length >= 6) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 5] = (passb1 >> 8) & 0xff; \
                                    if (pass_length >= 7) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 6] = (passb1 >> 16) & 0xff; \
                                    if (pass_length >= 8) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 7] = (passb1 >> 24) & 0xff; \
                                    if (pass_length >= 9) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 8] = (passb2 >> 0) & 0xff; \
                                    if (pass_length >= 10) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 9] = (passb2 >> 8) & 0xff; \
                                    if (pass_length >= 11) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 10] = (passb2 >> 16) & 0xff; \
                                    if (pass_length >= 12) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 11] = (passb2 >> 24) & 0xff; \
                                    if (pass_length >= 13) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 12] = (passb3 >> 0) & 0xff; \
                                    if (pass_length >= 14) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 13] = (passb3 >> 8) & 0xff; \
                                    if (pass_length >= 15) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 14] = (passb3 >> 16) & 0xff; \
                                    if (pass_length >= 16) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 15] = (passb3 >> 24) & 0xff; \
                                    successArray[search_index] = (unsigned char) 1; \
                                } \
                            } \
                        } \
                    } \
                    search_index++; \
                } \
            } \
        } \
next: \
        clearB0toB15(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15); \
        reduceSingleCharsetNormal(b0, b1, b2, a, b, c, d, CurrentStep, charset, charset_offset, pass_length, SHA1_Regenerate_Device_Table_Index); \
        charset_offset++; \
        if (charset_offset >= SHA1_Regenerate_Device_Charset_Length) { \
            charset_offset = 0; \
        } \
    } \
    if (CurrentStep >= (SHA1_Regenerate_Device_Chain_Length - 1)) { \
    }  \
    else { \
        SaveMD5RegistersIntoGlobalMemory(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15, \
            InitialArray32, SHA1_Regenerate_Device_Number_Of_Chains_To_Regen, password_index, pass_length); \
    } \
}

CREATE_SHA1_REGEN_KERNEL(6)
CREATE_SHA1_REGEN_KERNEL(7)
CREATE_SHA1_REGEN_KERNEL(8)
CREATE_SHA1_REGEN_KERNEL(9)
CREATE_SHA1_REGEN_KERNEL(10)


extern "C" void LaunchSHA1RegenerateKernel(int PasswordLength, int CUDA_Blocks, int CUDA_Threads,
        unsigned char *InitialPasswordArray, unsigned char *FoundPasswordArray,
        unsigned char *DeviceHashArray, uint32_t PasswordSpaceOffset, uint32_t StartChainIndex,
        uint32_t StepsToRun, uint32_t charset_offset, unsigned char *successArray) {
        
    switch (PasswordLength) {
        case 6:
            RegenSHA1ChainLen6 <<< CUDA_Blocks, CUDA_Threads >>>
                (InitialPasswordArray, FoundPasswordArray,
                DeviceHashArray, PasswordSpaceOffset, StartChainIndex,
                StepsToRun, charset_offset, successArray);
            break;
        case 7:
            RegenSHA1ChainLen7 <<< CUDA_Blocks, CUDA_Threads >>>
                (InitialPasswordArray, FoundPasswordArray,
                DeviceHashArray, PasswordSpaceOffset, StartChainIndex,
                StepsToRun, charset_offset, successArray);
            break;
        case 8:
            RegenSHA1ChainLen8 <<< CUDA_Blocks, CUDA_Threads >>>
                (InitialPasswordArray, FoundPasswordArray,
                DeviceHashArray, PasswordSpaceOffset, StartChainIndex,
                StepsToRun, charset_offset, successArray);
            break;
        case 9:
            RegenSHA1ChainLen9 <<< CUDA_Blocks, CUDA_Threads >>>
                (InitialPasswordArray, FoundPasswordArray,
                DeviceHashArray, PasswordSpaceOffset, StartChainIndex,
                StepsToRun, charset_offset, successArray);
            break;
        case 10:
            RegenSHA1ChainLen10 <<< CUDA_Blocks, CUDA_Threads >>>
                (InitialPasswordArray, FoundPasswordArray,
                DeviceHashArray, PasswordSpaceOffset, StartChainIndex,
                StepsToRun, charset_offset, successArray);
            break;
        default:
            printf("Password length %d not supported!", PasswordLength);
            exit(1);
    }

    cudaThreadSynchronize();

    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
      {
        printf("Cuda error: %s.\n", cudaGetErrorString( err) );
        exit(1);;
      }

}
