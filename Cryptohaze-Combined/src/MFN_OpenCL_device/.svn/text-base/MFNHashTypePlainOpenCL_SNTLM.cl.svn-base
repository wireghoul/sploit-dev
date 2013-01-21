



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



#if CPU_DEBUG
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif


#define MD4ROTATE_LEFT(x, y) rotate((vector_type)x, (uint)y)
#define MD4F(x, y, z) bitselect((z), (y), (x))
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

#define MD4ROTATE_RIGHT(x, n) (((x) >> (n)) | ((x) << (32-(n))))

#define REV_HH(a,b,c,d,data,shift) \
    a = MD4ROTATE_RIGHT((a), shift) - data - (vector_type)0x6ed9eba1 - (b ^ c ^ d);

#define REV_GG(a,b,c,d,data,shift) \
    a = MD4ROTATE_RIGHT((a), shift) - data - (vector_type)0x5a827999 - MD4G(b, c, d);


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
/*MD4GG (d, a, b, c, b7, MD4S22); \
MD4GG (c, d, a, b, b11, MD4S23); \
MD4GG (b, c, d, a, b15, MD4S24); \
MD4HH (a, b, c, d, b0, MD4S31); \
MD4HH (d, a, b, c, b8, MD4S32); \
MD4HH (c, d, a, b, b4, MD4S33); \
MD4HH (b, c, d, a, b12, MD4S34); \
MD4HH (a, b, c, d, b2, MD4S31); \
MD4HH (d, a, b, c, b10, MD4S32); \
MD4HH (c, d, a, b, b6, MD4S33); \
MD4HH (b, c, d, a, b14, MD4S34); \
MD4HH (a, b, c, d, b1, MD4S31); \
MD4HH (d, a, b, c, b9, MD4S32); \
MD4HH (c, d, a, b, b5, MD4S33); \
MD4HH (b, c, d, a, b13, MD4S34); \
MD4HH (a, b, c, d, b3, MD4S31); \
MD4HH (d, a, b, c, b11, MD4S32); \
MD4HH (c, d, a, b, b7, MD4S33); \
MD4HH (b, c, d, a, b15, MD4S34);*/ \
}

#define REVERSE() { \
    a = (vector_type) searchHash[0]; \
    b = (vector_type) searchHash[1]; \
    c = (vector_type) searchHash[2]; \
    d = (vector_type) searchHash[3]; \
    REV_HH(b, c, d, a, 0, MD4S34); \
    REV_HH(c, d, a, b, b7, MD4S33); \
    REV_HH(d, a, b, c, b11, MD4S32); \
    REV_HH(a, b, c, d, b3, MD4S31); \
    REV_HH(b, c, d, a, b13, MD4S34); \
    REV_HH(c, d, a, b, b5, MD4S33); \
    REV_HH(d, a, b, c, b9, MD4S32); \
    REV_HH(a, b, c, d, b1, MD4S31); \
    REV_HH(b, c, d, a, b14, MD4S34); \
    REV_HH(c, d, a, b, b6, MD4S33); \
    REV_HH(d, a, b, c, b10, MD4S32); \
    REV_HH(a, b, c, d, b2, MD4S31); \
    REV_HH(b, c, d, a, b12, MD4S34); \
    REV_HH(c, d, a, b, b4, MD4S33); \
    REV_HH(d, a, b, c, b8, MD4S32); \
    REV_HH(a, b, c, d, 0, MD4S31); \
    a_t = a; \
    b_t = b; \
    c_t = c; \
    d_t = d; \
}

// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
#define CopyFoundPasswordToMemoryNTLM(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 15: \
            dfp[14] = (b7.s##suffix >> 0) & 0xff; \
        case 14: \
            dfp[13] = (b6.s##suffix >> 16) & 0xff; \
        case 13: \
            dfp[12] = (b6.s##suffix >> 0) & 0xff; \
        case 12: \
            dfp[11] = (b5.s##suffix >> 16) & 0xff; \
        case 11: \
            dfp[10] = (b5.s##suffix >> 0) & 0xff; \
        case 10: \
            dfp[9] = (b4.s##suffix >> 16) & 0xff; \
        case 9: \
            dfp[8] = (b4.s##suffix >> 0) & 0xff; \
        case 8: \
            dfp[7] = (b3.s##suffix >> 16) & 0xff; \
        case 7: \
            dfp[6] = (b3.s##suffix >> 0) & 0xff; \
        case 6: \
            dfp[5] = (b2.s##suffix >> 16) & 0xff; \
        case 5: \
            dfp[4] = (b2.s##suffix >> 0) & 0xff; \
        case 4: \
            dfp[3] = (b1.s##suffix >> 16) & 0xff; \
        case 3: \
            dfp[2] = (b1.s##suffix >> 0) & 0xff; \
        case 2: \
            dfp[1] = (b0.s##suffix >> 16) & 0xff; \
        case 1: \
            dfp[0] = (b0.s##suffix >> 0) & 0xff; \
    } \
    dfpf[0] = (unsigned char) 1; \
}


#define PASSWORD_TEST(dfp, dfpf, suffix) { \
if ( ((a.s##suffix + b0.s##suffix) == a_t.s##suffix) && \
     (b.s##suffix == b_t.s##suffix) && \
     (c.s##suffix == c_t.s##suffix) && \
     (d.s##suffix == d_t.s##suffix)) { \
     CopyFoundPasswordToMemoryNTLM(dfp, dfpf, suffix); \
} }


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_SNTLM(
    __constant unsigned char const * restrict deviceCharsetPlainSNTLM, /* 0 */
    __constant unsigned char const * restrict deviceReverseCharsetPlainSNTLM, /* 1 */
    __constant unsigned char const * restrict charsetLengthsPlainSNTLM, /* 2 */
    __constant unsigned char const * restrict constantBitmapAPlainSNTLM, /* 3 */
        
    __private unsigned long const numberOfHashesPlainSNTLM, /* 4 */
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainSNTLM, /* 5 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlainSNTLM, /* 6 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainSNTLM, /* 7 */
        
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainSNTLM, /* 8 */
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainSNTLM, /* 9 */
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainSNTLM, /* 10 */
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainSNTLM, /* 11 */
        
    __global   unsigned char *deviceGlobalStartPointsPlainSNTLM, /* 12 */
    __private unsigned long const deviceNumberThreadsPlainSNTLM, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainSNTLM, /* 14 */
    __global   unsigned int * deviceGlobalStartPasswordsPlainSNTLM, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainSNTLM /* 16 */
) {
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d;
    vector_type a_t, b_t, c_t, d_t;
    unsigned long password_count = 0;
    
    __local unsigned int searchHash[4];

    if (get_local_id(0) == 0) {
        searchHash[0] = deviceGlobalHashlistAddressPlainSNTLM[0];
        searchHash[1] = deviceGlobalHashlistAddressPlainSNTLM[1];
        searchHash[2] = deviceGlobalHashlistAddressPlainSNTLM[2];
        searchHash[3] = deviceGlobalHashlistAddressPlainSNTLM[3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    a = b = c = d = (vector_type) 0;

    // Load and "spread" NTLM data - we expand from 4 characters per word to the UTF16-LE
    // Reuse b14 as it's used anyway.
    b14 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSNTLM[0]);
#if CPU_DEBUG
    printf("loading state b14: %08x %08x %08x %08x\n", b14.s0, b14.s1, b14.s2, b14.s3);
#endif
    b0 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
#if PASSWORD_LENGTH > 1
    b1 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
#endif
    
#if PASSWORD_LENGTH > 3
    b14 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSNTLM[1 * deviceNumberThreadsPlainSNTLM]);
    b2 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 5
    b3 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
#endif
    
#if PASSWORD_LENGTH > 7
    b14 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSNTLM[2 * deviceNumberThreadsPlainSNTLM]);
    b4 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 9
    b5 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
#endif
    
#if PASSWORD_LENGTH > 11
    b14 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSNTLM[3 * deviceNumberThreadsPlainSNTLM]);
    b6 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 13
    b7 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
#endif

    // Set the length field.
    b14 = (vector_type) (PASSWORD_LENGTH * 8 * 2);
    
    REVERSE();

    while (password_count < deviceNumberStepsToRunPlainSNTLM) {
        MD4_FIRST_2_ROUNDS();
        
        if (any((a + b0) == a_t)) {
            MD4GG (d, a, b, c, b7, MD4S22);
            if (any(d == d_t)) {
                MD4GG (c, d, a, b, b11, MD4S23);
                if (any(c == c_t)) {
                    MD4GG (b, c, d, a, b15, MD4S24);
                    if (any(b == b_t)) {
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, 0);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, 1);
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, 2);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, 3);
#endif
#if grt_vector_8 || grt_vector_16
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, 4);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, 5);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, 6);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, 7);
#endif
#if grt_vector_16
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, 8);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, 9);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, A);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, B);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, C);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, D);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, E);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSNTLM, deviceGlobalFoundPasswordFlagsPlainSNTLM, F);
#endif
                    }
                }
            }
        }
        
        OpenCLNoMemPasswordIncrementorLE();

        password_count++; 
    }
    
    // Compress the NTLM hash back down into the MD5-style packed password
    b14 =  (b0 & 0xff) | ((b0 & 0xff0000) >> 8);
#if PASSWORD_LENGTH > 1
    b14 |= (b1 & 0xff) << 16 | ((b1 & 0xff0000) << 8);
#endif
    vstore_type(b14, get_global_id(0), &deviceGlobalStartPasswordsPlainSNTLM[0]);
    
#if CPU_DEBUG
    printf("storing state b14: %08x %08x %08x %08x\n", b14.s0, b14.s1, b14.s2, b14.s3);
#endif
    
#if PASSWORD_LENGTH > 3
    b14 =  (b2 & 0xff) | ((b2 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 5
    b14 |= (b3 & 0xff) << 16 | ((b3 & 0xff0000) << 8);

    #endif
    vstore_type(b14, get_global_id(0), &deviceGlobalStartPasswordsPlainSNTLM[1 * deviceNumberThreadsPlainSNTLM]);
#endif

#if PASSWORD_LENGTH > 7
    b14 =  (b4 & 0xff) | ((b4 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 9
    b14 |= (b5 & 0xff) << 16 | ((b5 & 0xff0000) << 8);
    #endif
    vstore_type(b14, get_global_id(0), &deviceGlobalStartPasswordsPlainSNTLM[2 * deviceNumberThreadsPlainSNTLM]);
#endif

#if PASSWORD_LENGTH > 11
    b14 =  (b6 & 0xff) | ((b6 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 13
    b14 |= (b7 & 0xff) << 16 | ((b7 & 0xff0000) << 8);
    #endif
    vstore_type(b14, get_global_id(0), &deviceGlobalStartPasswordsPlainSNTLM[3 * deviceNumberThreadsPlainSNTLM]);
#endif

}

