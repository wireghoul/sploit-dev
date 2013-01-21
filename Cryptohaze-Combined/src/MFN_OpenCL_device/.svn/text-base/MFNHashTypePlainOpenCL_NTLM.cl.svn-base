



// Things we should define in the calling code...
#define CPU_DEBUG 0
//#define BITALIGN 1
//#define NVIDIA_HACKS
//#define PASSWORD_LENGTH 6

// Make my UI sane...
#ifndef __OPENCL_VERSION__
    #define __kernel
    #define __global
    #define __local
    #define __private
    #define __constant
    #define get_global_id(x)
    #define restrict
    #include <vector_types.h>
#endif

#ifndef VECTOR_WIDTH
    //#error "VECTOR_WIDTH must be defined for compile!"
    #define VECTOR_WIDTH 4
#endif

#ifndef PASSWORD_LENGTH
    #define PASSWORD_LENGTH 5
#endif

#if VECTOR_WIDTH == 1
    #define vector_type uint
    #define vload_type vload1
    #define vstore_type vstore1
    #define grt_vector_2 1
    #define vload1(offset, p) (offset + *p) 
    #define grt_vector_1 1
    #undef E0
    #define E0
#elif VECTOR_WIDTH == 2
    #define vector_type uint2
    #define vload_type vload2
    #define vstore_type vstore2
    #define grt_vector_2 1
#elif VECTOR_WIDTH == 4
    #define vector_type uint4
    #define vload_type vload4
    #define vstore_type vstore4
    #define grt_vector_4 1
#elif VECTOR_WIDTH == 8
    #define vector_type uint8
    #define vload_type vload8
    #define vstore_type vstore8
    #define grt_vector_8 1
#elif VECTOR_WIDTH == 16
    #define vector_type uint16
    #define vload_type vload16
    #define vstore_type vstore16
    #define grt_vector_16 1
#else
    #error "Vector width not specified or invalid vector width specified!"
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

#if CPU_DEBUG
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif


#if BITALIGN
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#define MD4ROTATE_LEFT(x, y) rotate((vector_type)x, (uint)y)
#define MD4FF(a,b,c,d,x,s) { (a) = (a)+x+(amd_bytealign((b),(c),(d))); (a) = MD4ROTATE_LEFT((a), (s)); }
#define MD4H(x, y, z) ((x) ^ (y) ^ (z))
#define MD4GG(a,b,c,d,x,s) {(a) = (a) + (vector_type)0x5a827999 + (amd_bytealign((b), ((d) | (c)), ((c) & (d)))) +x  ; (a) = MD4ROTATE_LEFT((a), (s)); }
#else
#define MD4ROTATE_LEFT(x, y) rotate((vector_type)x, (uint)y)
//#define MD4F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD4F(x, y, z) bitselect((z), (y), (x))
//#define MD4G(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define MD4G(x, y, z) bitselect((y),(x),((z)^(y))) 
#define MD4H(x, y, z) ((x) ^ (y) ^ (z))
#define MD4FF(a, b, c, d, x, s) { \
    (a) += MD4F ((b), (c), (d)) + (x); \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#define MD4GG(a, b, c, d, x, s) { \
    (a) += MD4G ((b), (c), (d)) + (x) + (vector_type)0x5a827999; \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#endif

#define MD4HH(a, b, c, d, x, s) { \
    (a) += MD4H ((b), (c), (d)) + (x) + (vector_type)0x6ed9eba1; \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#define MD4S11 3
#define MD4S12 7
#define MD4S13 11
#define MD4S14 19
#define MD4S21 3
#define MD4S22 5
#define MD4S23 9
#define MD4S24 13
#define MD4S31 3
#define MD4S32 9
#define MD4S33 11
#define MD4S34 15
/* End MD4 Defines */


#define MD4_FIRST_2_ROUNDS() { \
a = (vector_type)0x67452301; \
b = (vector_type)0xefcdab89; \
c = (vector_type)0x98badcfe; \
d = (vector_type)0x10325476; \
MD4FF (a, b, c, d, b0, MD4S11); \
MD4FF (d, a, b, c, b1, MD4S12); \
MD4FF (c, d, a, b, b2, MD4S13); \
MD4FF (b, c, d, a, b3, MD4S14); \
MD4FF (a, b, c, d, b4, MD4S11); \
MD4FF (d, a, b, c, b5, MD4S12); \
MD4FF (c, d, a, b, b6, MD4S13); \
MD4FF (b, c, d, a, b7, MD4S14); \
MD4FF (a, b, c, d, b8, MD4S11); \
MD4FF (d, a, b, c, b9, MD4S12); \
MD4FF (c, d, a, b, b10, MD4S13); \
MD4FF (b, c, d, a, b11, MD4S14); \
MD4FF (a, b, c, d, b12, MD4S11); \
MD4FF (d, a, b, c, b13, MD4S12); \
MD4FF (c, d, a, b, b14, MD4S13); \
MD4FF (b, c, d, a, b15, MD4S14); \
MD4GG (a, b, c, d, b0, MD4S21); \
MD4GG (d, a, b, c, b4, MD4S22); \
MD4GG (c, d, a, b, b8, MD4S23); \
MD4GG (b, c, d, a, b12, MD4S24); \
MD4GG (a, b, c, d, b1, MD4S21); \
MD4GG (d, a, b, c, b5, MD4S22); \
MD4GG (c, d, a, b, b9, MD4S23); \
MD4GG (b, c, d, a, b13, MD4S24); \
MD4GG (a, b, c, d, b2, MD4S21); \
MD4GG (d, a, b, c, b6, MD4S22); \
MD4GG (c, d, a, b, b10, MD4S23); \
MD4GG (b, c, d, a, b14, MD4S24); \
MD4GG (a, b, c, d, b3, MD4S21); \
MD4GG (d, a, b, c, b7, MD4S22); \
MD4GG (c, d, a, b, b11, MD4S23); \
MD4GG (b, c, d, a, b15, MD4S24); \
}

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_CHARSET_LENGTH 128

// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
#define CopyFoundPasswordToMemoryNTLM(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 27: \
            dfp[search_index * PASSWORD_LENGTH + 26] = (b13.s##suffix >> 0) & 0xff; \
        case 26: \
            dfp[search_index * PASSWORD_LENGTH + 25] = (b12.s##suffix >> 16) & 0xff; \
        case 25: \
            dfp[search_index * PASSWORD_LENGTH + 24] = (b12.s##suffix >> 0) & 0xff; \
        case 24: \
            dfp[search_index * PASSWORD_LENGTH + 23] = (b11.s##suffix >> 16) & 0xff; \
        case 23: \
            dfp[search_index * PASSWORD_LENGTH + 22] = (b11.s##suffix >> 0) & 0xff; \
        case 22: \
            dfp[search_index * PASSWORD_LENGTH + 21] = (b10.s##suffix >> 16) & 0xff; \
        case 21: \
            dfp[search_index * PASSWORD_LENGTH + 20] = (b10.s##suffix >> 0) & 0xff; \
        case 20: \
            dfp[search_index * PASSWORD_LENGTH + 19] = (b9.s##suffix >> 16) & 0xff; \
        case 19: \
            dfp[search_index * PASSWORD_LENGTH + 18] = (b9.s##suffix >> 0) & 0xff; \
        case 18: \
            dfp[search_index * PASSWORD_LENGTH + 17] = (b8.s##suffix >> 16) & 0xff; \
        case 17: \
            dfp[search_index * PASSWORD_LENGTH + 16] = (b8.s##suffix >> 0) & 0xff; \
        case 16: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b7.s##suffix >> 16) & 0xff; \
        case 15: \
            dfp[search_index * PASSWORD_LENGTH + 14] = (b7.s##suffix >> 0) & 0xff; \
        case 14: \
            dfp[search_index * PASSWORD_LENGTH + 13] = (b6.s##suffix >> 16) & 0xff; \
        case 13: \
            dfp[search_index * PASSWORD_LENGTH + 12] = (b6.s##suffix >> 0) & 0xff; \
        case 12: \
            dfp[search_index * PASSWORD_LENGTH + 11] = (b5.s##suffix >> 16) & 0xff; \
        case 11: \
            dfp[search_index * PASSWORD_LENGTH + 10] = (b5.s##suffix >> 0) & 0xff; \
        case 10: \
            dfp[search_index * PASSWORD_LENGTH + 9] = (b4.s##suffix >> 16) & 0xff; \
        case 9: \
            dfp[search_index * PASSWORD_LENGTH + 8] = (b4.s##suffix >> 0) & 0xff; \
        case 8: \
            dfp[search_index * PASSWORD_LENGTH + 7] = (b3.s##suffix >> 16) & 0xff; \
        case 7: \
            dfp[search_index * PASSWORD_LENGTH + 6] = (b3.s##suffix >> 0) & 0xff; \
        case 6: \
            dfp[search_index * PASSWORD_LENGTH + 5] = (b2.s##suffix >> 16) & 0xff; \
        case 5: \
            dfp[search_index * PASSWORD_LENGTH + 4] = (b2.s##suffix >> 0) & 0xff; \
        case 4: \
            dfp[search_index * PASSWORD_LENGTH + 3] = (b1.s##suffix >> 16) & 0xff; \
        case 3: \
            dfp[search_index * PASSWORD_LENGTH + 2] = (b1.s##suffix >> 0) & 0xff; \
        case 2: \
            dfp[search_index * PASSWORD_LENGTH + 1] = (b0.s##suffix >> 16) & 0xff; \
        case 1: \
            dfp[search_index * PASSWORD_LENGTH + 0] = (b0.s##suffix >> 0) & 0xff; \
    } \
    deviceGlobalFoundPasswordFlagsPlainNTLM[search_index] = (unsigned char) 1; \
}


#define CheckPassword128LENTLM(dgh, dfp, dfpf, dnh, suffix) { \
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
                    CopyFoundPasswordToMemoryNTLM(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}


// sb: shared bitmap a
// gb{a-d}: global bitmap a-d
// dgh: Device Global hashlist
// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
// dnh: Device number hashes
#define checkPasswordNTLM(sb, gba, gbb, gbc, gbd, dgh, dfp, dfpf, dnh, suffix) { \
    if ((sb[(a.s##suffix & 0x0000ffff) >> 3] >> (a.s##suffix & 0x00000007)) & 0x00000001) { \
        if (!(gba) || ((gba[(a.s##suffix >> 3) & 0x07FFFFFF] >> (a.s##suffix & 0x7)) & 0x1)) { \
            if (!gbb || ((gbb[(b.s##suffix >> 3) & 0x07FFFFFF] >> (b.s##suffix & 0x7)) & 0x1)) { \
                if (!gbc || ((gbc[(c.s##suffix >> 3) & 0x07FFFFFF] >> (c.s##suffix & 0x7)) & 0x1)) { \
                    if (!gbd || ((gbd[(d.s##suffix >> 3) & 0x07FFFFFF] >> (d.s##suffix & 0x7)) & 0x1)) { \
                        /*printf("Bitmap HIT!\n");*/ \
                        CheckPassword128LENTLM(dgh, dfp, dfpf, dnh, suffix); \
                    } \
                } \
            } \
        } \
    } \
}




__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_NTLM(
    __constant unsigned char const * restrict deviceCharsetPlainNTLM, /* 0 */
    __constant unsigned char const * restrict deviceReverseCharsetPlainNTLM, /* 1 */
    __constant unsigned char const * restrict charsetLengthsPlainNTLM, /* 2 */
    __constant unsigned char const * restrict constantBitmapAPlainNTLM, /* 3 */
        
    __private unsigned long const numberOfHashesPlainNTLM, /* 4 */
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainNTLM, /* 5 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlainNTLM, /* 6 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainNTLM, /* 7 */
        
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainNTLM, /* 8 */
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainNTLM, /* 9 */
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainNTLM, /* 10 */
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainNTLM, /* 11 */
        
    __global   unsigned char *deviceGlobalStartPointsPlainNTLM, /* 12 */
    __private unsigned long const deviceNumberThreadsPlainNTLM, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainNTLM, /* 14 */
    __global   unsigned int * deviceGlobalStartPasswordsPlainNTLM, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainNTLM /* 16 */
) {
    // Start the kernel.
    //__local unsigned char sharedCharsetPlainNTLM[MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    //__local unsigned char sharedReverseCharsetPlainNTLM[MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    //__local unsigned char sharedCharsetLengthsPlainNTLM[PASSWORD_LENGTH];
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local unsigned char padding[1];
    
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
                deviceCharsetPlainNTLM[0], deviceCharsetPlainNTLM[1], deviceCharsetPlainNTLM[2]);
        printf("Charset lengths: %d %d %d...\n", charsetLengthsPlainNTLM[0], 
                charsetLengthsPlainNTLM[1], charsetLengthsPlainNTLM[2]);
        printf("Number hashes: %d\n", numberOfHashesPlainNTLM);
        printf("Bitmap A: %lu\n", deviceGlobalBitmapAPlainNTLM);
        printf("Bitmap B: %lu\n", deviceGlobalBitmapBPlainNTLM);
        printf("Bitmap C: %lu\n", deviceGlobalBitmapCPlainNTLM);
        printf("Bitmap D: %lu\n", deviceGlobalBitmapDPlainNTLM);
        printf("Number threads: %lu\n", deviceNumberThreadsPlainNTLM);
        printf("Steps to run: %u\n", deviceNumberStepsToRunPlainNTLM);
        printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        
        int i, j;
        
        //for (i = 0; i < (deviceNumberThreadsPlainNTLM * PASSWORD_LENGTH); i++) {
        //    printf("%c", deviceGlobalStartPointsPlainNTLM[i]);
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
            sharedBitmap[counter] = constantBitmapAPlainNTLM[counter];
        }/*
#pragma unroll 128
        for (counter = 0; counter < (MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_CHARSET_LENGTH * PASSWORD_LENGTH); counter++) {
            sharedCharsetPlainNTLM[counter] = deviceCharsetPlainNTLM[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < (MFN_HASH_TYPE_PLAIN_CUDA_NTLM_MAX_CHARSET_LENGTH * PASSWORD_LENGTH); counter++) {
            sharedReverseCharsetPlainNTLM[counter] = deviceReverseCharsetPlainNTLM[counter];
        }
        for (counter = 0; counter < PASSWORD_LENGTH; counter++) {
            sharedCharsetLengthsPlainNTLM[counter] = charsetLengthsPlainNTLM[counter];
        }*/
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    a = b = c = d = (vector_type) 0;

    // Load and "spread" NTLM data - we expand from 4 characters per word to the UTF16-LE
    // Reuse b14 as it's used anyway.
    b14 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[0]);
#if CPU_DEBUG
    printf("loading state b14: %08x %08x %08x %08x\n", b14.s0, b14.s1, b14.s2, b14.s3);
#endif
    b0 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
#if PASSWORD_LENGTH > 1
    b1 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
#endif
    
#if PASSWORD_LENGTH > 3
    b14 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[1 * deviceNumberThreadsPlainNTLM]);
    b2 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 5
    b3 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
#endif
    
#if PASSWORD_LENGTH > 7
    b14 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[2 * deviceNumberThreadsPlainNTLM]);
    b4 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 9
    b5 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
#endif
    
#if PASSWORD_LENGTH > 11
    b14 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[3 * deviceNumberThreadsPlainNTLM]);
    b6 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 13
    b7 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
#endif

#if PASSWORD_LENGTH > 15
    b14 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[4 * deviceNumberThreadsPlainNTLM]);
    b8 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 17
    b9 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
#endif

#if PASSWORD_LENGTH > 19
    b14 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[5 * deviceNumberThreadsPlainNTLM]);
    b10 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 21
    b11 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
#endif
    
#if PASSWORD_LENGTH > 23
    b14 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[6 * deviceNumberThreadsPlainNTLM]);
    b12 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 25
    b13 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
#endif

    // Set the length field.
    b14 = (vector_type) (PASSWORD_LENGTH * 8 * 2);
    
    while (password_count < deviceNumberStepsToRunPlainNTLM) {
#if CPU_DEBUG
        printf("Thread %d Step %d\n", get_global_id(0), password_count);
#endif
        MD4_FIRST_2_ROUNDS();
        // Split by password lengths for easier early-out.
        if (PASSWORD_LENGTH <= 6) {
            MD4HH (a, b, c, d, b0, MD4S31);
            MD4HH (d, a, b, c, b8, MD4S32);
            MD4HH (c, d, a, b, b4, MD4S33);
            MD4HH (b, c, d, a, b12, MD4S34);
            MD4HH (a, b, c, d, b2, MD4S31);
            MD4HH (d, a, b, c, b10, MD4S32);
            OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
                    deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
                    deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
                    deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
                    numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);
        } else if (PASSWORD_LENGTH <= 14) {
            MD4HH (a, b, c, d, b0, MD4S31);
            MD4HH (d, a, b, c, b8, MD4S32);
            MD4HH (c, d, a, b, b4, MD4S33);
            MD4HH (b, c, d, a, b12, MD4S34);
            MD4HH (a, b, c, d, b2, MD4S31);
            MD4HH (d, a, b, c, b10, MD4S32);
            MD4HH (c, d, a, b, b6, MD4S33);
            MD4HH (b, c, d, a, b14, MD4S34);
            MD4HH (a, b, c, d, b1, MD4S31);
            MD4HH (d, a, b, c, b9, MD4S32);
            OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
                    deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
                    deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
                    deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
                    numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);
        } else {
            MD4HH (a, b, c, d, b0, MD4S31);
            MD4HH (d, a, b, c, b8, MD4S32);
            MD4HH (c, d, a, b, b4, MD4S33);
            MD4HH (b, c, d, a, b12, MD4S34);
            MD4HH (a, b, c, d, b2, MD4S31);
            MD4HH (d, a, b, c, b10, MD4S32);
            MD4HH (c, d, a, b, b6, MD4S33);
            MD4HH (b, c, d, a, b14, MD4S34);
            MD4HH (a, b, c, d, b1, MD4S31);
            MD4HH (d, a, b, c, b9, MD4S32);
            MD4HH (c, d, a, b, b5, MD4S33);
            MD4HH (b, c, d, a, b13, MD4S34);
            OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
                    deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
                    deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
                    deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
                    numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);
        }

        
#if CPU_DEBUG
        printf(".s0 pass: %c%c%c%c%c%c hash: %08x%08x%08x%08x\n",
                (b0.s0 >> 0) & 0xff, (b0.s0 >> 16) & 0xff,
                (b1.s0 >> 0) & 0xff, (b1.s0 >> 16) & 0xff,
                (b2.s0 >> 0) & 0xff, (b2.s0 >> 16) & 0xff,
                a.s0, b.s0, c.s0, d.s0);
        printf(".s0 words: %08x %08x %08x %08x %08x %08x\n", 
                b0.s0, b1.s0, b2.s0, b3.s0, b4.s0, b5.s0);
        printf(".s1 pass: %c%c%c%c%c%c hash: %08x%08x%08x%08x\n",
                (b0.s1 >> 0) & 0xff, (b0.s1 >> 16) & 0xff,
                (b1.s1 >> 0) & 0xff, (b1.s1 >> 16) & 0xff,
                (b2.s1 >> 0) & 0xff, (b2.s1 >> 16) & 0xff,
                a.s1, b.s1, c.s1, d.s1);
        printf(".s1 words: %08x %08x %08x %08x %08x %08x\n", 
                b0.s1, b1.s1, b2.s1, b3.s1, b4.s1, b5.s1);
        printf(".s2 pass: %c%c%c%c%c%c hash: %08x%08x%08x%08x\n",
                (b0.s2 >> 0) & 0xff, (b0.s2 >> 16) & 0xff,
                (b1.s2 >> 0) & 0xff, (b1.s2 >> 16) & 0xff,
                (b2.s2 >> 0) & 0xff, (b2.s2 >> 16) & 0xff,
                a.s2, b.s2, c.s2, d.s2);
        printf(".s2 words: %08x %08x %08x %08x %08x %08x\n", 
                b0.s2, b1.s2, b2.s2, b3.s2, b4.s2, b5.s2);
        printf(".s3 pass: %c%c%c%c%c%c hash: %08x%08x%08x%08x\n",
                (b0.s3 >> 0) & 0xff, (b0.s3 >> 16) & 0xff,
                (b1.s3 >> 0) & 0xff, (b1.s3 >> 16) & 0xff,
                (b2.s3 >> 0) & 0xff, (b2.s3 >> 16) & 0xff,
                a.s3, b.s3, c.s3, d.s3);
        printf(".s3 words: %08x %08x %08x %08x %08x %08x\n", 
                b0.s3, b1.s3, b2.s3, b3.s3, b4.s3, b5.s3);
#endif   
        // Defines created by calling code.
//        OpenCLPasswordCheck128(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
//                deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
//                deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
//                deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
//                numberOfHashesPlainNTLM);
        
        //OpenCLPasswordIncrementorNTLM(sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM);
        OpenCLNoMemPasswordIncrementorLE();

        password_count++; 
    }
    
    // Compress the NTLM hash back down into the MD5-style packed password
    b14 =  (b0 & 0xff) | ((b0 & 0xff0000) >> 8);
#if PASSWORD_LENGTH > 1
    b14 |= (b1 & 0xff) << 16 | ((b1 & 0xff0000) << 8);
#endif
    vstore_type(b14, get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[0]);
    
#if CPU_DEBUG
    printf("storing state b14: %08x %08x %08x %08x\n", b14.s0, b14.s1, b14.s2, b14.s3);
#endif
    
#if PASSWORD_LENGTH > 3
    b14 =  (b2 & 0xff) | ((b2 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 5
    b14 |= (b3 & 0xff) << 16 | ((b3 & 0xff0000) << 8);

    #endif
    vstore_type(b14, get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[1 * deviceNumberThreadsPlainNTLM]);
#endif

#if PASSWORD_LENGTH > 7
    b14 =  (b4 & 0xff) | ((b4 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 9
    b14 |= (b5 & 0xff) << 16 | ((b5 & 0xff0000) << 8);
    #endif
    vstore_type(b14, get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[2 * deviceNumberThreadsPlainNTLM]);
#endif

#if PASSWORD_LENGTH > 11
    b14 =  (b6 & 0xff) | ((b6 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 13
    b14 |= (b7 & 0xff) << 16 | ((b7 & 0xff0000) << 8);
    #endif
    vstore_type(b14, get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[3 * deviceNumberThreadsPlainNTLM]);
#endif

#if PASSWORD_LENGTH > 15
    b14 =  (b8 & 0xff) | ((b8 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 17
    b14 |= (b9 & 0xff) << 16 | ((b9 & 0xff0000) << 8);
    #endif
    vstore_type(b14, get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[4 * deviceNumberThreadsPlainNTLM]);
#endif

#if PASSWORD_LENGTH > 19
    b14 =  (b10 & 0xff) | ((b10 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 21
    b14 |= (b11 & 0xff) << 16 | ((b11 & 0xff0000) << 8);
    #endif
    vstore_type(b14, get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[5 * deviceNumberThreadsPlainNTLM]);
#endif

#if PASSWORD_LENGTH > 23
    b14 =  (b12 & 0xff) | ((b12 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 25
    b14 |= (b13 & 0xff) << 16 | ((b13 & 0xff0000) << 8);
    #endif
    vstore_type(b14, get_global_id(0), &deviceGlobalStartPasswordsPlainNTLM[6 * deviceNumberThreadsPlainNTLM]);
#endif
    
}

