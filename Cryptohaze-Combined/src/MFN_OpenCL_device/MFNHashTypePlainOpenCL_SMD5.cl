
// Things we should define in the calling code...
#define CPU_DEBUG 0
//#define BITALIGN 1
//#define PASSWORD_LENGTH 6



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
    #define vload1(offset, p) *(offset + p) 
    #define vstore1(val, offset, p) *(offset + p) = val 
    #define grt_vector_1 1
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

// Hash defines

#define MD5S11 7
#define MD5S12 12
#define MD5S13 17
#define MD5S14 22
#define MD5S21 5
#define MD5S22 9
#define MD5S23 14
#define MD5S24 20
#define MD5S31 4
#define MD5S32 11
#define MD5S33 16
#define MD5S34 23
#define MD5S41 6
#define MD5S42 10
#define MD5S43 15
#define MD5S44 21

/* F, G, H and I are basic MD5 functions.
 */

//#define MD5F(x, y, z) (((x) & (y)) | ((~x) & (z)))
//#define MD5G(x, y, z) (((x) & (z)) | ((y) & (~z)))

// Define F and G with bitselect.  If bfi_int is not being used, this works
// properly, and I understand the 7970 natively understands this function.
#define MD5F(x, y, z) bitselect((z), (y), (x))
#define MD5G(x, y, z) bitselect((y), (x), (z))

#define MD5H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5I(x, y, z) ((y) ^ ((x) | (~z)))



#ifdef BITALIGN
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
// Use bitalign if we are making use of ATI GPUs.
//#define MD5ROTATE_LEFT(x, y) amd_bitalign(x, x, (uint)(32 - y))
#define MD5ROTATE_LEFT(x, y) rotate((vector_type)x, (uint)y)
#define MD5FF(a, b, c, d, x, s, ac) { \
 (a) += amd_bytealign((b),(c),(d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
// HERP DERP LOOK I CAN ACCELERATE GG TOO!
#define MD5GG(a, b, c, d, x, s, ac) { \
 (a) += amd_bytealign((d),(b),(c)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#else
// Doing this with OpenCL common types for other GPUs.
#define MD5ROTATE_LEFT(x, y) rotate((vector_type)x, (uint)y)
#define MD5FF(a, b, c, d, x, s, ac) { \
 (a) += MD5F ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5GG(a, b, c, d, x, s, ac) { \
 (a) += MD5G ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#endif


// We can't easily optimize these.
#define MD5HH(a, b, c, d, x, s, ac) { \
 (a) += MD5H ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5II(a, b, c, d, x, s, ac) { \
 (a) += MD5I ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }




#define MD5_FIRST_3_ROUNDS() { \
a = (vector_type)0x67452301; \
b = (vector_type)0xefcdab89; \
c = (vector_type)0x98badcfe; \
d = (vector_type)0x10325476; \
MD5FF(a, b, c, d, b0, MD5S11, 0xd76aa478); \
MD5FF(d, a, b, c, b1, MD5S12, 0xe8c7b756); \
MD5FF(c, d, a, b, b2, MD5S13, 0x242070db); \
MD5FF(b, c, d, a, b3, MD5S14, 0xc1bdceee); \
MD5FF(a, b, c, d, 0, MD5S11, 0xf57c0faf); \
MD5FF(d, a, b, c, 0, MD5S12, 0x4787c62a); \
MD5FF(c, d, a, b, 0, MD5S13, 0xa8304613); \
MD5FF(b, c, d, a, 0, MD5S14, 0xfd469501); \
MD5FF(a, b, c, d, 0, MD5S11, 0x698098d8); \
MD5FF(d, a, b, c, 0, MD5S12, 0x8b44f7af); \
MD5FF(c, d, a, b, 0, MD5S13, 0xffff5bb1); \
MD5FF(b, c, d, a, 0, MD5S14, 0x895cd7be); \
MD5FF(a, b, c, d, 0, MD5S11, 0x6b901122); \
MD5FF(d, a, b, c, 0, MD5S12, 0xfd987193); \
MD5FF(c, d, a, b, (PASSWORD_LENGTH * 8), MD5S13, 0xa679438e); \
MD5FF(b, c, d, a, 0, MD5S14, 0x49b40821); \
MD5GG(a, b, c, d, b1, MD5S21, 0xf61e2562); \
MD5GG(d, a, b, c, 0, MD5S22, 0xc040b340); \
MD5GG(c, d, a, b, 0, MD5S23, 0x265e5a51); \
MD5GG(b, c, d, a, b0, MD5S24, 0xe9b6c7aa); \
MD5GG(a, b, c, d, 0, MD5S21, 0xd62f105d); \
MD5GG(d, a, b, c, 0, MD5S22, 0x2441453); \
MD5GG(c, d, a, b, 0, MD5S23, 0xd8a1e681); \
MD5GG(b, c, d, a, 0, MD5S24, 0xe7d3fbc8); \
MD5GG(a, b, c, d, 0, MD5S21, 0x21e1cde6); \
MD5GG(d, a, b, c, (PASSWORD_LENGTH * 8), MD5S22, 0xc33707d6); \
MD5GG(c, d, a, b, b3, MD5S23, 0xf4d50d87); \
MD5GG(b, c, d, a, 0, MD5S24, 0x455a14ed); \
MD5GG(a, b, c, d, 0, MD5S21, 0xa9e3e905); \
MD5GG(d, a, b, c, b2, MD5S22, 0xfcefa3f8); \
MD5GG(c, d, a, b, 0, MD5S23, 0x676f02d9); \
MD5GG(b, c, d, a, 0, MD5S24, 0x8d2a4c8a); \
MD5HH(a, b, c, d, 0, MD5S31, 0xfffa3942); \
MD5HH(d, a, b, c, 0, MD5S32, 0x8771f681); \
MD5HH(c, d, a, b, 0, MD5S33, 0x6d9d6122); \
MD5HH(b, c, d, a, (PASSWORD_LENGTH * 8), MD5S34, 0xfde5380c); \
MD5HH(a, b, c, d, b1, MD5S31, 0xa4beea44); \
MD5HH(d, a, b, c, 0, MD5S32, 0x4bdecfa9); \
MD5HH(c, d, a, b, 0, MD5S33, 0xf6bb4b60); \
MD5HH(b, c, d, a, 0, MD5S34, 0xbebfbc70); \
MD5HH(a, b, c, d, 0, MD5S31, 0x289b7ec6); \
MD5HH(d, a, b, c, b0, MD5S32, 0xeaa127fa); \
MD5HH(c, d, a, b, b3, MD5S33, 0xd4ef3085); \
MD5HH(b, c, d, a, 0, MD5S34, 0x4881d05); \
MD5HH(a, b, c, d, 0, MD5S31, 0xd9d4d039); \
}


#define PASSWORD_TEST(dfp, dfpf, suffix) { \
if ( ((a.s##suffix + b0.s##suffix) == a_t.s##suffix) && \
     (b.s##suffix == b_t.s##suffix) && \
     (c.s##suffix == c_t.s##suffix) && \
     (d.s##suffix == d_t.s##suffix)) { \
     CopyFoundPasswordToMemory(dfp, dfpf, suffix); \
} }

// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
#define CopyFoundPasswordToMemory(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 16: \
            dfp[15] = (b3.s##suffix >> 24) & 0xff; \
        case 15: \
            dfp[14] = (b3.s##suffix >> 16) & 0xff; \
        case 14: \
            dfp[13] = (b3.s##suffix >> 8) & 0xff; \
        case 13: \
            dfp[12] = (b3.s##suffix >> 0) & 0xff; \
        case 12: \
            dfp[11] = (b2.s##suffix >> 24) & 0xff; \
        case 11: \
            dfp[10] = (b2.s##suffix >> 16) & 0xff; \
        case 10: \
            dfp[9] = (b2.s##suffix >> 8) & 0xff; \
        case 9: \
            dfp[8] = (b2.s##suffix >> 0) & 0xff; \
        case 8: \
            dfp[7] = (b1.s##suffix >> 24) & 0xff; \
        case 7: \
            dfp[6] = (b1.s##suffix >> 16) & 0xff; \
        case 6: \
            dfp[5] = (b1.s##suffix >> 8) & 0xff; \
        case 5: \
            dfp[4] = (b1.s##suffix >> 0) & 0xff; \
        case 4: \
            dfp[3] = (b0.s##suffix >> 24) & 0xff; \
        case 3: \
            dfp[2] = (b0.s##suffix >> 16) & 0xff; \
        case 2: \
            dfp[1] = (b0.s##suffix >> 8) & 0xff; \
        case 1: \
            dfp[0] = (b0.s##suffix >> 0) & 0xff; \
    } \
    dfpf[0] = (unsigned char) 1; \
}

#define MD5ROTATE_RIGHT(x, n) (((x) >> (n)) | ((x) << (32-(n))))

#define REV_II(a,b,c,d,data,shift,constant) \
    a = MD5ROTATE_RIGHT((a - b), shift) - data - constant - (c ^ (b | (~d)));

#if PASSWORD_LENGTH <= 8
#define REVERSE() { \
    a = (vector_type) searchHash[0]; \
    b = (vector_type) searchHash[1]; \
    c = (vector_type) searchHash[2]; \
    d = (vector_type) searchHash[3]; \
    REV_II (b, c, d, a, b1, MD5S44, 0x85845dd1); \
    REV_II (c, d, a, b, 0, MD5S43, 0xffeff47d); \
    REV_II (d, a, b, c, b3, MD5S42, 0x8f0ccc92); \
    REV_II (a, b, c, d, 0, MD5S41, 0x655b59c3); \
    REV_II (b, c, d, a, 0, MD5S44, 0xfc93a039); \
    REV_II (c, d, a, b, (PASSWORD_LENGTH * 8), MD5S43, 0xab9423a7); \
    REV_II (d, a, b, c, 0, MD5S42, 0x432aff97); \
    REV_II (a, b, c, d, 0, MD5S41, 0xf4292244); \
    a_t = a; \
    b_t = b; \
    c_t = c; \
    d_t = d; \
}
#else
#define REVERSE() { \
    a = (vector_type) searchHash[0]; \
    b = (vector_type) searchHash[1]; \
    c = (vector_type) searchHash[2]; \
    d = (vector_type) searchHash[3]; \
    REV_II (b, c, d, a, b9, MD5S44, 0xeb86d391); \
    REV_II (c, d, a, b, b2, MD5S43, 0x2ad7d2bb); \
    REV_II (d, a, b, c, b11, MD5S42, 0xbd3af235); \
    REV_II (a, b, c, d, b4, MD5S41, 0xf7537e82); \
    REV_II (b, c, d, a, b13, MD5S44, 0x4e0811a1); \
    REV_II (c, d, a, b, b6, MD5S43, 0xa3014314); \
    REV_II (d, a, b, c, b15, MD5S42, 0xfe2ce6e0); \
    REV_II (a, b, c, d, b8, MD5S41, 0x6fa87e4f); \
    REV_II (b, c, d, a, b1, MD5S44, 0x85845dd1); \
    REV_II (c, d, a, b, 0, MD5S43, 0xffeff47d); \
    REV_II (d, a, b, c, b3, MD5S42, 0x8f0ccc92); \
    REV_II (a, b, c, d, 0, MD5S41, 0x655b59c3); \
    REV_II (b, c, d, a, 0, MD5S44, 0xfc93a039); \
    REV_II (c, d, a, b, (PASSWORD_LENGTH * 8), MD5S43, 0xab9423a7); \
    REV_II (d, a, b, c, 0, MD5S42, 0x432aff97); \
    REV_II (a, b, c, d, 0, MD5S41, 0xf4292244); \
    a_t = a; \
    b_t = b; \
    c_t = c; \
    d_t = d; \
}
#endif

__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_SMD5(
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainSMD5, /* 0 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlainSMD5, /* 1 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainSMD5, /* 2 */

    __private unsigned long const deviceNumberThreadsPlainSMD5, /* 3 */
    __private unsigned int const deviceNumberStepsToRunPlainSMD5, /* 4 */
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlainSMD5 /* 5 */
) {
    // Start the kernel.
    vector_type b0, b1, b2, b3, a, b, c, d;
    vector_type a_t, b_t, c_t, d_t;

    unsigned long password_count = 0;
    __local unsigned int searchHash[4];

    if (get_local_id(0) == 0) {
        searchHash[0] = deviceGlobalHashlistAddressPlainSMD5[0];
        searchHash[1] = deviceGlobalHashlistAddressPlainSMD5[1];
        searchHash[2] = deviceGlobalHashlistAddressPlainSMD5[2];
        searchHash[3] = deviceGlobalHashlistAddressPlainSMD5[3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    b0 = b1 = b2 = b3 = (vector_type)0;
    a = b = c = d = (vector_type) 0;

    b0 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSMD5[0]);
    if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSMD5[1 * deviceNumberThreadsPlainSMD5]);}
    if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSMD5[2 * deviceNumberThreadsPlainSMD5]);}
    if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSMD5[3 * deviceNumberThreadsPlainSMD5]);}

    REVERSE();
    
    while (password_count < deviceNumberStepsToRunPlainSMD5) {
        MD5_FIRST_3_ROUNDS();
        // All the lengths need these.
        if (any((a + b0) == a_t)) {
            MD5HH(d, a, b, c, 0, MD5S32, 0xe6db99e5);
            if (any(d == d_t)) {
                MD5HH(c, d, a, b, 0, MD5S33, 0x1fa27cf8);
                if (any(c == c_t)) {
                    MD5HH(b, c, d, a, b2, MD5S34, 0xc4ac5665);
                    if (any(b == b_t)) {
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, 0);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, 1);
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, 2);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, 3);
#endif
#if grt_vector_8 || grt_vector_16
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, 4);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, 5);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, 6);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, 7);
#endif
#if grt_vector_16
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, 8);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, 9);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, A);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, B);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, C);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, D);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, E);
                    PASSWORD_TEST(deviceGlobalFoundPasswordsPlainSMD5, deviceGlobalFoundPasswordFlagsPlainSMD5, F);
#endif
        } } } }

        OpenCLNoMemPasswordIncrementorLE();

        password_count++; 
    }
    vstore_type(b0, get_global_id(0), &deviceGlobalStartPasswordsPlainSMD5[0]);
    if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_global_id(0), &deviceGlobalStartPasswordsPlainSMD5[1 * deviceNumberThreadsPlainSMD5]);}
    if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_global_id(0), &deviceGlobalStartPasswordsPlainSMD5[2 * deviceNumberThreadsPlainSMD5]);}
    if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_global_id(0), &deviceGlobalStartPasswordsPlainSMD5[3 * deviceNumberThreadsPlainSMD5]);}
  
}
