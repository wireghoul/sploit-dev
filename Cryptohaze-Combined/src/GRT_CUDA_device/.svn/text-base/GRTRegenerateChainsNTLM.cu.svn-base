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
__device__ __constant__ unsigned char NTLM_Regenerate_Device_Charset_Constant[512]; // Constant space for charset
__device__ __constant__ uint32_t NTLM_Regenerate_Device_Charset_Length; // Character set length
__device__ __constant__ uint32_t NTLM_Regenerate_Device_Chain_Length; // May as well pull it from constant memory... faster when cached.
__device__ __constant__ uint32_t NTLM_Regenerate_Device_Table_Index;
__device__ __constant__ uint32_t NTLM_Regenerate_Device_Number_Of_Threads; // It needs this, and can't easily calculate it
__device__ __constant__ uint32_t NTLM_Regenerate_Device_Number_Of_Chains_To_Regen;
__device__ __constant__ uint32_t NTLM_Regenerate_Device_Number_Of_Hashes;
__device__ __constant__ unsigned char NTLM_Regenerate_constantBitmap[8192]; // for lookups



#include "../../inc/CUDA_Common/CUDA_MD4.h"
#include "../../inc/CUDA_Common/Hash_Common.h"
#include "../../inc/GRT_CUDA_device/CUDA_Reduction_Functions.h"
#include "../../inc/GRT_CUDA_device/CUDA_Load_Store_Registers.h"

// Copy the shared variables to the host
extern "C" void copyNTLMRegenerateDataToConstant(char *hostCharset, uint32_t hostCharsetLength,
        uint32_t hostChainLength, uint32_t hostTableIndex, uint32_t hostNumberOfThreads, unsigned char *hostBitmap,
        uint32_t hostNumberOfHashes) {
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(NTLM_Regenerate_Device_Charset_Constant, hostCharset, 512));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(NTLM_Regenerate_Device_Charset_Length, &hostCharsetLength, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(NTLM_Regenerate_Device_Chain_Length, &hostChainLength, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(NTLM_Regenerate_Device_Table_Index, &hostTableIndex, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(NTLM_Regenerate_Device_Number_Of_Threads, &hostNumberOfThreads, sizeof(uint32_t)));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(NTLM_Regenerate_constantBitmap, hostBitmap, 8192));
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(NTLM_Regenerate_Device_Number_Of_Hashes, &hostNumberOfHashes, sizeof(uint32_t)));
}

extern "C" void setNTLMRegenerateNumberOfChains(uint32_t numberOfChains) {
    CH_CUDA_SAFE_CALL(cudaMemcpyToSymbol(NTLM_Regenerate_Device_Number_Of_Chains_To_Regen, &numberOfChains, sizeof(uint32_t)));
}



__device__ inline void copyBitmap(unsigned char *sharedBitmap) {
  uint64_t *sharedBitmapCoalesce = (uint64_t *)sharedBitmap;
  uint64_t *deviceBitmapCoalesce = (uint64_t *)NTLM_Regenerate_constantBitmap;

  int a;

  if (threadIdx.x == 0) {
      for (a = 0; a < (8192 / 8); a++) {
          sharedBitmapCoalesce[a] = deviceBitmapCoalesce[a];
      }
  }
  // Make sure everyone is here and done before we return.
  syncthreads();
}



/*
__global__ void RegenNTLMChainLen7(unsigned char *InitialPasswordArray, unsigned char *FoundPasswordArray,
        unsigned char *DeviceHashArray, uint32_t PasswordSpaceOffset, uint32_t StartChainIndex,
        uint32_t StepsToRun, uint32_t charset_offset, unsigned char *successArray) {
    // Needed variables for generation
    uint32_t CurrentStep, PassCount, password_index;

    const int password_length = 7;

    // Hash variables
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
    uint32_t a, b, c, d;

    // Word-width access to the arrays
    uint32_t *InitialArray32;
    uint32_t *FoundPassword32;
    uint32_t *DEVICE_Hashes_32;

    uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp;

    // 32-bit accesses to the hash arrays
    InitialArray32 = (uint32_t *) InitialPasswordArray;
    FoundPassword32 = (uint32_t *) FoundPasswordArray;
    DEVICE_Hashes_32 = (uint32_t *) DeviceHashArray;

    __shared__ char charset[512];
    __shared__ __align__(16) unsigned char sharedBitmap[8192];

    // Generic "copy charset to shared memory" function
    copySingleCharsetToShared(charset, NTLM_Regenerate_Device_Charset_Constant);
    copyBitmap(sharedBitmap);


    // Figure out which password we are working on.
    password_index = ((blockIdx.x * blockDim.x + threadIdx.x) + (PasswordSpaceOffset * NTLM_Regenerate_Device_Number_Of_Threads));


    // Return if this thread is working on something beyond the end of the password space
    if (password_index >= NTLM_Regenerate_Device_Number_Of_Chains_To_Regen) {
        return;
    }


    clearB0toB15(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
    // Load b0/b1 out of memory
    b0 = (uint32_t) InitialArray32[0 * NTLM_Regenerate_Device_Number_Of_Chains_To_Regen + password_index];
    b1 = (uint32_t) InitialArray32[1 * NTLM_Regenerate_Device_Number_Of_Chains_To_Regen + password_index];
    // Set up the padding/size.



    for (PassCount = 0; PassCount < StepsToRun; PassCount++) {
        CurrentStep = PassCount + StartChainIndex;

        padMDHash(password_length * 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
        CUDA_MD4(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d);

        if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) {
            {
                //printf("Bitmap hit.\n");
                // Init binary search through global password space

                //printf("NTLM_Regenerate_Device_Number_Of_hashes: %d\n", NTLM_Regenerate_Device_Number_Of_Hashes);

                search_high = NTLM_Regenerate_Device_Number_Of_Hashes;
                search_low = 0;
                search_index = 0;
                while (search_low < search_high) {
                    //printf("Search_low: %d\n", search_low);
                    //printf("Search_high: %d\n", search_high);

                    // Midpoint between search_high and search_low
                    search_index = search_low + (search_high - search_low) / 2;
                    //printf("Search_index: %d\n", search_index);
                    // reorder from low endian to big endian for searching, as hashes are sorted by byte.
                    temp = DEVICE_Hashes_32[4 * search_index];
                    hash_order_mem = (temp & 0xff) << 24 | ((temp >> 8) & 0xff) << 16 | ((temp >> 16) & 0xff) << 8 | ((temp >> 24) & 0xff);
                    hash_order_a = (a & 0xff) << 24 | ((a >> 8) & 0xff) << 16 | ((a >> 16) & 0xff) << 8 | ((a >> 24) & 0xff);

                    // Adjust search_high & search_low to work through space
                    if (hash_order_mem < hash_order_a) {
                        search_low = search_index + 1;
                    } else {
                        search_high = search_index;
                    }
                    if ((hash_order_a == hash_order_mem) && (search_low < NTLM_Regenerate_Device_Number_Of_Hashes)) {
                        // Break out of the search loop - search_index is on a value
                        break;
                    }
                }

                // Yes - it's a goto.  And it speeds up performance significantly (15%).
                // It stays.  These values are already loaded.  If they are not the same,
                // there is NO point to touching global memory again.
                if (hash_order_a != hash_order_mem) {
                    goto next;
                }
                // We've broken out of the loop, search_index should be on a matching value
                // Loop while the search index is the same - linear search through this to find all possible
                // matching passwords.
                // We first need to move backwards to the beginning, as we may be in the middle of a set of matching hashes.
                // If we are index 0, do NOT subtract, as we will wrap and this goes poorly.

                while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 4])) {
                    search_index--;
                }

                //printf("Got search index of %d\n", search_index);

                while ((a == DEVICE_Hashes_32[search_index * 4])) {
                    {
                        if (b == DEVICE_Hashes_32[search_index * 4 + 1]) {
                            if (c == DEVICE_Hashes_32[search_index * 4 + 2]) {
                                if (d == DEVICE_Hashes_32[search_index * 4 + 3]) {

                                    if (password_length >= 1) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 0] = (b0 >> 0) & 0xff;
                                    if (password_length >= 2) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 1] = (b0 >> 8) & 0xff;
                                    if (password_length >= 3) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 2] = (b0 >> 16) & 0xff;
                                    if (password_length >= 4) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 3] = (b0 >> 24) & 0xff;
                                    if (password_length >= 5) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 4] = (b1 >> 0) & 0xff;
                                    if (password_length >= 6) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 5] = (b1 >> 8) & 0xff;
                                    if (password_length >= 7) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 6] = (b1 >> 16) & 0xff;
                                            if (password_length >= 8) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 7] = deviceCharset[p7 + (MAX_CHARSET_LENGTH * 7)];
                                            if (password_length >= 9) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 8] = deviceCharset[p8 + (MAX_CHARSET_LENGTH * 8)];
                                            if (password_length >= 10) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 9] = deviceCharset[p9 + (MAX_CHARSET_LENGTH * 9)];
                                            if (password_length >= 11) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 10] = deviceCharset[p10 + (MAX_CHARSET_LENGTH * 10)];
                                            if (password_length >= 12) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 11] = deviceCharset[p11 + (MAX_CHARSET_LENGTH * 11)];
                                            if (password_length >= 13) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 12] = deviceCharset[p12 + (MAX_CHARSET_LENGTH * 12)];
                                            if (password_length >= 14) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 13] = deviceCharset[p13 + (MAX_CHARSET_LENGTH * 13)];
                                            if (password_length >= 15) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 14] = deviceCharset[p14 + (MAX_CHARSET_LENGTH * 14)];
                                            if (password_length >= 16) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 15] = deviceCharset[p15 + (MAX_CHARSET_LENGTH * 15)];
                                     
                                    successArray[search_index] = (unsigned char) 1;

                                    printf("FOUND PASSWORD:");
                                    for (int i = 0; i < password_length; i++) {
                                        printf("%c", FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + i]);
                                    }
                                    printf("\n");
                                }
                            }
                        }
                    }
                    search_index++;
                }
            }
        }
        // This is where the goto goes.  Notice the skipping of all the global memory access.
next:

        reduceSingleCharsetNTLM(b0, b1, b2, b3, b4, a, b, c, d, CurrentStep, charset, charset_offset, password_length, NTLM_Regenerate_Device_Table_Index);

        charset_offset++;
        if (charset_offset >= NTLM_Regenerate_Device_Charset_Length) {
            charset_offset = 0;
        }
    }
    // Done with the number of steps we need to run

    // If we are done (or have somehow overflowed), store the result
    if (CurrentStep >= (NTLM_Regenerate_Device_Chain_Length - 1)) {
        // Do nothing.
    }        // Else, store the b0/b1 values back to the initial array for the next loop
    else {
        InitialArray32[0 * NTLM_Regenerate_Device_Number_Of_Chains_To_Regen + password_index] = b0;
        InitialArray32[1 * NTLM_Regenerate_Device_Number_Of_Chains_To_Regen + password_index] = b1;
    }
}
*/



#define CREATE_NTLM_REGEN_KERNEL(length) \
__global__ void RegenNTLMChainLen##length(unsigned char *InitialPasswordArray, unsigned char *FoundPasswordArray, \
        unsigned char *DeviceHashArray, uint32_t PasswordSpaceOffset, uint32_t StartChainIndex, \
        uint32_t StepsToRun, uint32_t charset_offset, unsigned char *successArray) { \
    uint32_t CurrentStep, PassCount, password_index; \
    const int pass_length = length; \
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
    uint32_t a, b, c, d; \
    uint32_t *InitialArray32; \
    uint32_t *DEVICE_Hashes_32; \
    uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp; \
    InitialArray32 = (uint32_t *) InitialPasswordArray; \
    DEVICE_Hashes_32 = (uint32_t *) DeviceHashArray; \
    __shared__ char charset[512]; \
    __shared__ __align__(16) unsigned char sharedBitmap[8192]; \
    copySingleCharsetToShared(charset, NTLM_Regenerate_Device_Charset_Constant); \
    copyBitmap(sharedBitmap); \
    password_index = ((blockIdx.x * blockDim.x + threadIdx.x) + (PasswordSpaceOffset * NTLM_Regenerate_Device_Number_Of_Threads)); \
    if (password_index >= NTLM_Regenerate_Device_Number_Of_Chains_To_Regen) { \
        return; \
    } \
    clearB0toB15(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    LoadNTLMRegistersFromGlobalMemory(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15, \
        InitialArray32, NTLM_Regenerate_Device_Number_Of_Chains_To_Regen, password_index, pass_length); \
    for (PassCount = 0; PassCount < StepsToRun; PassCount++) { \
        CurrentStep = PassCount + StartChainIndex; \
        padMDHash(pass_length * 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        CUDA_MD4(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d); \
        if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) { \
            { \
                search_high = NTLM_Regenerate_Device_Number_Of_Hashes; \
                search_low = 0; \
                search_index = 0; \
                while (search_low < search_high) { \
                    search_index = search_low + (search_high - search_low) / 2; \
                    temp = DEVICE_Hashes_32[4 * search_index]; \
                    hash_order_mem = (temp & 0xff) << 24 | ((temp >> 8) & 0xff) << 16 | ((temp >> 16) & 0xff) << 8 | ((temp >> 24) & 0xff); \
                    hash_order_a = (a & 0xff) << 24 | ((a >> 8) & 0xff) << 16 | ((a >> 16) & 0xff) << 8 | ((a >> 24) & 0xff); \
                    if (hash_order_mem < hash_order_a) { \
                        search_low = search_index + 1; \
                    } else { \
                        search_high = search_index; \
                    } \
                    if ((hash_order_a == hash_order_mem) && (search_low < NTLM_Regenerate_Device_Number_Of_Hashes)) { \
                        break; \
                    } \
                } \
                if (hash_order_a != hash_order_mem) { \
                    goto next; \
                } \
                while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 4])) { \
                    search_index--; \
                } \
                while ((a == DEVICE_Hashes_32[search_index * 4])) { \
                    { \
                        if (b == DEVICE_Hashes_32[search_index * 4 + 1]) { \
                            if (c == DEVICE_Hashes_32[search_index * 4 + 2]) { \
                                if (d == DEVICE_Hashes_32[search_index * 4 + 3]) { \
                                    if (pass_length >= 1) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 0] = (b0 >> 0) & 0xff; \
                                    if (pass_length >= 2) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 1] = (b0 >> 16) & 0xff; \
                                    if (pass_length >= 3) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 2] = (b1 >> 0) & 0xff; \
                                    if (pass_length >= 4) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 3] = (b1 >> 16) & 0xff; \
                                    if (pass_length >= 5) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 4] = (b2 >> 0) & 0xff; \
                                    if (pass_length >= 6) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 5] = (b2 >> 16) & 0xff; \
                                    if (pass_length >= 7) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 6] = (b3 >> 0) & 0xff; \
                                    if (pass_length >= 8) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 7] = (b3 >> 16) & 0xff; \
                                    if (pass_length >= 9) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 8] = (b4 >> 0) & 0xff; \
                                    if (pass_length >= 10) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 9] = (b4 >> 16) & 0xff; \
                                    if (pass_length >= 11) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 10] = (b5 >> 0) & 0xff; \
                                    if (pass_length >= 12) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 11] = (b5 >> 16) & 0xff; \
                                    if (pass_length >= 13) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 12] = (b6 >> 0) & 0xff; \
                                    if (pass_length >= 14) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 13] = (b6 >> 16) & 0xff; \
                                    if (pass_length >= 15) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 14] = (b7 >> 0) & 0xff; \
                                    if (pass_length >= 16) FoundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 15] = (b7 >> 16) & 0xff; \
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
        reduceSingleCharsetNTLM(b0, b1, b2, b3, b4, a, b, c, d, CurrentStep, charset, charset_offset, pass_length, NTLM_Regenerate_Device_Table_Index); \
        charset_offset++; \
        if (charset_offset >= NTLM_Regenerate_Device_Charset_Length) { \
            charset_offset = 0; \
        } \
    } \
    if (CurrentStep >= (NTLM_Regenerate_Device_Chain_Length - 1)) { \
    } \
    else { \
        SaveNTLMRegistersIntoGlobalMemory(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15, \
            InitialArray32, NTLM_Regenerate_Device_Number_Of_Chains_To_Regen, password_index, pass_length); \
     } \
}


CREATE_NTLM_REGEN_KERNEL(6);
CREATE_NTLM_REGEN_KERNEL(7);
CREATE_NTLM_REGEN_KERNEL(8);
CREATE_NTLM_REGEN_KERNEL(9);
CREATE_NTLM_REGEN_KERNEL(10);

extern "C" void LaunchNTLMRegenerateKernel(int PasswordLength, int CUDA_Blocks, int CUDA_Threads,
        unsigned char *InitialPasswordArray, unsigned char *FoundPasswordArray,
        unsigned char *DeviceHashArray, uint32_t PasswordSpaceOffset, uint32_t StartChainIndex,
        uint32_t StepsToRun, uint32_t charset_offset, unsigned char *successArray) {

    switch (PasswordLength) {
        case 6:
            //printf("Launching len6 kernel\n");
            RegenNTLMChainLen6 <<< CUDA_Blocks, CUDA_Threads >>>
                (InitialPasswordArray, FoundPasswordArray,
                DeviceHashArray, PasswordSpaceOffset, StartChainIndex,
                StepsToRun, charset_offset, successArray);
            break;
        case 7:
            RegenNTLMChainLen7 <<< CUDA_Blocks, CUDA_Threads >>>
                (InitialPasswordArray, FoundPasswordArray,
                DeviceHashArray, PasswordSpaceOffset, StartChainIndex,
                StepsToRun, charset_offset, successArray);
            break;
        case 8:
            RegenNTLMChainLen8 <<< CUDA_Blocks, CUDA_Threads >>>
                (InitialPasswordArray, FoundPasswordArray,
                DeviceHashArray, PasswordSpaceOffset, StartChainIndex,
                StepsToRun, charset_offset, successArray);
            break;
        case 9:
            RegenNTLMChainLen9 <<< CUDA_Blocks, CUDA_Threads >>>
                (InitialPasswordArray, FoundPasswordArray,
                DeviceHashArray, PasswordSpaceOffset, StartChainIndex,
                StepsToRun, charset_offset, successArray);
            break;
        case 10:
            RegenNTLMChainLen10 <<< CUDA_Blocks, CUDA_Threads >>>
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
