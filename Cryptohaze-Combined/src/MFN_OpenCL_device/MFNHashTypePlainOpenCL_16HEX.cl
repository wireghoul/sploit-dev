
// Make my UI sane...
#ifndef __OPENCL_VERSION__
    #define __kernel
    #define __global
    #define __local
    #define __private
    #define __constant
    #define get_global_id(x)
    #define get_local_id(x)
    #define restrict
    #include <vector_types.h>
#include "MFN_OpenCL_device/MFNHashTypeHeader.h"
#include "MFN_OpenCL_device/MFNHashTypeMD5.h"
#endif


#ifndef SHARED_BITMAP_SIZE
#define SHARED_BITMAP_SIZE 8
#endif

#if SHARED_BITMAP_SIZE == 8
#define SHARED_BITMAP_MASK 0x0000ffff
#elif SHARED_BITMAP_SIZE == 16
#define SHARED_BITMAP_MASK 0x0001ffff
#elif SHARED_BITMAP_SIZE == 32
#define SHARED_BITMAP_MASK 0x0003ffff
#endif

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_CHARSET_LENGTH 128

// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
#define CopyFoundPasswordToMemory(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 16: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b3.s##suffix >> 24) & 0xff; \
        case 15: \
            dfp[search_index * PASSWORD_LENGTH + 14] = (b3.s##suffix >> 16) & 0xff; \
        case 14: \
            dfp[search_index * PASSWORD_LENGTH + 13] = (b3.s##suffix >> 8) & 0xff; \
        case 13: \
            dfp[search_index * PASSWORD_LENGTH + 12] = (b3.s##suffix >> 0) & 0xff; \
        case 12: \
            dfp[search_index * PASSWORD_LENGTH + 11] = (b2.s##suffix >> 24) & 0xff; \
        case 11: \
            dfp[search_index * PASSWORD_LENGTH + 10] = (b2.s##suffix >> 16) & 0xff; \
        case 10: \
            dfp[search_index * PASSWORD_LENGTH + 9] = (b2.s##suffix >> 8) & 0xff; \
        case 9: \
            dfp[search_index * PASSWORD_LENGTH + 8] = (b2.s##suffix >> 0) & 0xff; \
        case 8: \
            dfp[search_index * PASSWORD_LENGTH + 7] = (b1.s##suffix >> 24) & 0xff; \
        case 7: \
            dfp[search_index * PASSWORD_LENGTH + 6] = (b1.s##suffix >> 16) & 0xff; \
        case 6: \
            dfp[search_index * PASSWORD_LENGTH + 5] = (b1.s##suffix >> 8) & 0xff; \
        case 5: \
            dfp[search_index * PASSWORD_LENGTH + 4] = (b1.s##suffix >> 0) & 0xff; \
        case 4: \
            dfp[search_index * PASSWORD_LENGTH + 3] = (b0.s##suffix >> 24) & 0xff; \
        case 3: \
            dfp[search_index * PASSWORD_LENGTH + 2] = (b0.s##suffix >> 16) & 0xff; \
        case 2: \
            dfp[search_index * PASSWORD_LENGTH + 1] = (b0.s##suffix >> 8) & 0xff; \
        case 1: \
            dfp[search_index * PASSWORD_LENGTH + 0] = (b0.s##suffix >> 0) & 0xff; \
    } \
    deviceGlobalFoundPasswordFlagsPlain16HEX[search_index] = (unsigned char) 1; \
}


#define CheckPassword128LE(dgh, dfp, dfpf, dnh, suffix) { \
    search_high = dnh; \
    search_low = 0; \
    while (search_low < search_high) { \
        search_index = search_low + (search_high - search_low) / 2; \
        current_hash_value = dgh[4 * search_index]; \
        if (current_hash_value < a.s##suffix) { \
            search_low = search_index + 1; \
        } else { \
            search_high = search_index; \
        } \
        if ((a.s##suffix == current_hash_value) && (search_low < dnh)) { \
            break; \
        } \
    } \
    if (a.s##suffix == current_hash_value) { \
        while (search_index && (a.s##suffix == dgh[(search_index - 1) * 4])) { \
            search_index--; \
        } \
        while ((a.s##suffix == dgh[search_index * 4])) { \
            if (b.s##suffix == dgh[search_index * 4 + 1]) { \
                if (c.s##suffix == dgh[search_index * 4 + 2]) { \
                    if (d.s##suffix == dgh[search_index * 4 + 3]) { \
                    /*printf("YEHAA!\n");*/ \
                    CopyFoundPasswordToMemory(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_16HEX(
    __constant unsigned char const * restrict deviceCharsetPlain16HEX, /* 0 */
    __constant unsigned char const * restrict deviceReverseCharsetPlain16HEX, /* 1 */
    __constant unsigned char const * restrict charsetLengthsPlain16HEX, /* 2 */
    __constant unsigned char const * restrict constantBitmapAPlain16HEX, /* 3 */
        
    __private unsigned long const numberOfHashesPlain16HEX, /* 4 */
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlain16HEX, /* 5 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlain16HEX, /* 6 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlain16HEX, /* 7 */
        
    __global   unsigned char const * restrict deviceGlobalBitmapAPlain16HEX, /* 8 */
    __global   unsigned char const * restrict deviceGlobalBitmapBPlain16HEX, /* 9 */
    __global   unsigned char const * restrict deviceGlobalBitmapCPlain16HEX, /* 10 */
    __global   unsigned char const * restrict deviceGlobalBitmapDPlain16HEX, /* 11 */
        
    __global   unsigned char *deviceGlobalStartPointsPlain16HEX, /* 12 */
    __private unsigned long const deviceNumberThreadsPlain16HEX, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlain16HEX, /* 14 */
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlain16HEX, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlain16HEX /* 16 */
) {
    // Start the kernel.
    __local unsigned char sharedCharsetPlain16HEX[MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    __local unsigned char sharedReverseCharsetPlain16HEX[MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    __local unsigned char sharedCharsetLengthsPlain16HEX[PASSWORD_LENGTH];
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;

#if CPU_DEBUG
    //printf("Kernel start, global id %d\n", get_global_id(0));
    
    if (get_global_id(0) == 0) {
        printf("Charset forward: %c %c %c ...\n", 
                deviceCharsetPlain16HEX[0], deviceCharsetPlain16HEX[1], deviceCharsetPlain16HEX[2]);
        printf("Charset lengths: %d %d %d...\n", charsetLengthsPlain16HEX[0], 
                charsetLengthsPlain16HEX[1], charsetLengthsPlain16HEX[2]);
        printf("Number hashes: %d\n", numberOfHashesPlain16HEX);
        printf("Bitmap A: %lu\n", deviceGlobalBitmapAPlain16HEX);
        printf("Bitmap B: %lu\n", deviceGlobalBitmapBPlain16HEX);
        printf("Bitmap C: %lu\n", deviceGlobalBitmapCPlain16HEX);
        printf("Bitmap D: %lu\n", deviceGlobalBitmapDPlain16HEX);
        printf("Number threads: %lu\n", deviceNumberThreadsPlain16HEX);
        printf("Steps to run: %u\n", deviceNumberStepsToRunPlain16HEX);
        printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        
        int i, j;
        
        //for (i = 0; i < (deviceNumberThreadsPlain16HEX * PASSWORD_LENGTH); i++) {
        //    printf("%c", deviceGlobalStartPointsPlain16HEX[i]);
        //}
        
        vector_type data0;
        
        //data0 = char0;
        
        //printf("data0.s1: %02x\n", data0.s1);
        //printf("data0.s2: %02x\n", data0.s2);
        //printf("data0.s3: %02x\n", data0.s3);
    }
#endif
    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlain16HEX[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < (MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_CHARSET_LENGTH * PASSWORD_LENGTH); counter++) {
            sharedCharsetPlain16HEX[counter] = deviceCharsetPlain16HEX[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < (MFN_HASH_TYPE_PLAIN_CUDA_16HEX_MAX_CHARSET_LENGTH * PASSWORD_LENGTH); counter++) {
            sharedReverseCharsetPlain16HEX[counter] = deviceReverseCharsetPlain16HEX[counter];
        }
        for (counter = 0; counter < PASSWORD_LENGTH; counter++) {
            sharedCharsetLengthsPlain16HEX[counter] = charsetLengthsPlain16HEX[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    b14 = (vector_type) (PASSWORD_LENGTH * 8);
    a = b = c = d = (vector_type) 0;

    b0 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlain16HEX[0]);
    if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlain16HEX[1 * deviceNumberThreadsPlain16HEX]);}
    if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlain16HEX[2 * deviceNumberThreadsPlain16HEX]);}
    if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlain16HEX[3 * deviceNumberThreadsPlain16HEX]);}
        
    while (password_count < deviceNumberStepsToRunPlain16HEX) {
        MD5_FULL_HASH();
        // If the password length is <= 8, we can start the early out process 
        // at this point and perform the final 3 checks internally.
            OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlain16HEX, 
                deviceGlobalBitmapBPlain16HEX, deviceGlobalBitmapCPlain16HEX, 
                deviceGlobalBitmapDPlain16HEX, deviceGlobalHashlistAddressPlain16HEX, 
                deviceGlobalFoundPasswordsPlain16HEX, deviceGlobalFoundPasswordFlagsPlain16HEX,
                numberOfHashesPlain16HEX, deviceGlobal256kbBitmapAPlain16HEX);
//        printf(".s0 pass: %c%c%c%c%c hash: %08x%08x%08x%08x\n",
//                (b0.s0 >> 0) & 0xff, (b0.s0 >> 8) & 0xff,
//                (b0.s0 >> 16) & 0xff, (b0.s0 >> 24) & 0xff,
//                (b1.s0 >> 0) & 0xff,
//                a.s0, b.s0, c.s0, d.s0);
//        printf(".s1 pass: %c%c%c%c%c hash: %08x%08x%08x%08x\n",
//                (b0.s1 >> 0) & 0xff, (b0.s1 >> 8) & 0xff,
//                (b0.s1 >> 16) & 0xff, (b0.s1 >> 24) & 0xff,
//                (b1.s1 >> 0) & 0xff,
//                a.s1, b.s1, c.s1, d.s1);
//        printf(".s2 pass: %c%c%c%c%c hash: %08x%08x%08x%08x\n",
//                (b0.s2 >> 0) & 0xff, (b0.s2 >> 8) & 0xff,
//                (b0.s2 >> 16) & 0xff, (b0.s2 >> 24) & 0xff,
//                (b1.s2 >> 0) & 0xff,
//                a.s2, b.s2, c.s2, d.s2);
//        printf(".s3 pass: %c%c%c%c%c hash: %08x%08x%08x%08x\n",
//                (b0.s3 >> 0) & 0xff, (b0.s3 >> 8) & 0xff,
//                (b0.s3 >> 16) & 0xff, (b0.s3 >> 24) & 0xff,
//                (b1.s3 >> 0) & 0xff,
//                a.s3, b.s3, c.s3, d.s3);

        OpenCLPasswordIncrementorLE(sharedCharsetPlain16HEX, sharedReverseCharsetPlain16HEX, sharedCharsetLengthsPlain16HEX);

        password_count++; 
    }
    vstore_type(b0, get_global_id(0), &deviceGlobalStartPasswordsPlain16HEX[0]);
    if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_global_id(0), &deviceGlobalStartPasswordsPlain16HEX[1 * deviceNumberThreadsPlain16HEX]);}
    if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_global_id(0), &deviceGlobalStartPasswordsPlain16HEX[2 * deviceNumberThreadsPlain16HEX]);}
    if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_global_id(0), &deviceGlobalStartPasswordsPlain16HEX[3 * deviceNumberThreadsPlain16HEX]);}
  
}
