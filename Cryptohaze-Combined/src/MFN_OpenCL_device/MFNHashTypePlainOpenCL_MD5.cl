
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
MD5FF(a, b, c, d, b4, MD5S11, 0xf57c0faf); \
MD5FF(d, a, b, c, b5, MD5S12, 0x4787c62a); \
MD5FF(c, d, a, b, b6, MD5S13, 0xa8304613); \
MD5FF(b, c, d, a, b7, MD5S14, 0xfd469501); \
MD5FF(a, b, c, d, b8, MD5S11, 0x698098d8); \
MD5FF(d, a, b, c, b9, MD5S12, 0x8b44f7af); \
MD5FF(c, d, a, b, b10, MD5S13, 0xffff5bb1); \
MD5FF(b, c, d, a, b11, MD5S14, 0x895cd7be); \
MD5FF(a, b, c, d, b12, MD5S11, 0x6b901122); \
MD5FF(d, a, b, c, b13, MD5S12, 0xfd987193); \
MD5FF(c, d, a, b, b14, MD5S13, 0xa679438e); \
MD5FF(b, c, d, a, b15, MD5S14, 0x49b40821); \
MD5GG(a, b, c, d, b1, MD5S21, 0xf61e2562); \
MD5GG(d, a, b, c, b6, MD5S22, 0xc040b340); \
MD5GG(c, d, a, b, b11, MD5S23, 0x265e5a51); \
MD5GG(b, c, d, a, b0, MD5S24, 0xe9b6c7aa); \
MD5GG(a, b, c, d, b5, MD5S21, 0xd62f105d); \
MD5GG(d, a, b, c, b10, MD5S22, 0x2441453); \
MD5GG(c, d, a, b, b15, MD5S23, 0xd8a1e681); \
MD5GG(b, c, d, a, b4, MD5S24, 0xe7d3fbc8); \
MD5GG(a, b, c, d, b9, MD5S21, 0x21e1cde6); \
MD5GG(d, a, b, c, b14, MD5S22, 0xc33707d6); \
MD5GG(c, d, a, b, b3, MD5S23, 0xf4d50d87); \
MD5GG(b, c, d, a, b8, MD5S24, 0x455a14ed); \
MD5GG(a, b, c, d, b13, MD5S21, 0xa9e3e905); \
MD5GG(d, a, b, c, b2, MD5S22, 0xfcefa3f8); \
MD5GG(c, d, a, b, b7, MD5S23, 0x676f02d9); \
MD5GG(b, c, d, a, b12, MD5S24, 0x8d2a4c8a); \
MD5HH(a, b, c, d, b5, MD5S31, 0xfffa3942); \
MD5HH(d, a, b, c, b8, MD5S32, 0x8771f681); \
MD5HH(c, d, a, b, b11, MD5S33, 0x6d9d6122); \
MD5HH(b, c, d, a, b14, MD5S34, 0xfde5380c); \
MD5HH(a, b, c, d, b1, MD5S31, 0xa4beea44); \
MD5HH(d, a, b, c, b4, MD5S32, 0x4bdecfa9); \
MD5HH(c, d, a, b, b7, MD5S33, 0xf6bb4b60); \
MD5HH(b, c, d, a, b10, MD5S34, 0xbebfbc70); \
MD5HH(a, b, c, d, b13, MD5S31, 0x289b7ec6); \
MD5HH(d, a, b, c, b0, MD5S32, 0xeaa127fa); \
MD5HH(c, d, a, b, b3, MD5S33, 0xd4ef3085); \
MD5HH(b, c, d, a, b6, MD5S34, 0x4881d05); \
MD5HH(a, b, c, d, b9, MD5S31, 0xd9d4d039); \
MD5HH(d, a, b, c, b12, MD5S32, 0xe6db99e5); \
MD5HH(c, d, a, b, b15, MD5S33, 0x1fa27cf8); \
MD5HH(b, c, d, a, b2, MD5S34, 0xc4ac5665); \
}


/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH 128

// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
#ifdef grt_vector_1
#define CopyFoundPasswordToMemory(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 55: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b13 >> 16) & 0xff; \
        case 54: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b13 >> 8) & 0xff; \
        case 53: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b13 >> 0) & 0xff; \
        case 52: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b12 >> 24) & 0xff; \
        case 51: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b12 >> 16) & 0xff; \
        case 50: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b12 >> 8) & 0xff; \
        case 49: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b12 >> 0) & 0xff; \
        case 48: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b11 >> 24) & 0xff; \
        case 47: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b11 >> 16) & 0xff; \
        case 46: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b11 >> 8) & 0xff; \
        case 45: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b11 >> 0) & 0xff; \
        case 44: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b10 >> 24) & 0xff; \
        case 43: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b10 >> 16) & 0xff; \
        case 42: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b10 >> 8) & 0xff; \
        case 41: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b10 >> 0) & 0xff; \
        case 40: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b9 >> 24) & 0xff; \
        case 39: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b9 >> 16) & 0xff; \
        case 38: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b9 >> 8) & 0xff; \
        case 37: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b9 >> 0) & 0xff; \
        case 36: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b8 >> 24) & 0xff; \
        case 35: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b8 >> 16) & 0xff; \
        case 34: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b8 >> 8) & 0xff; \
        case 33: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b8 >> 0) & 0xff; \
        case 32: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b7 >> 24) & 0xff; \
        case 31: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b7 >> 16) & 0xff; \
        case 30: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b7 >> 8) & 0xff; \
        case 29: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b7 >> 0) & 0xff; \
        case 28: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b6 >> 24) & 0xff; \
        case 27: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b6 >> 16) & 0xff; \
        case 26: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b6 >> 8) & 0xff; \
        case 25: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b6 >> 0) & 0xff; \
        case 24: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b5 >> 24) & 0xff; \
        case 23: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b5 >> 16) & 0xff; \
        case 22: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b5 >> 8) & 0xff; \
        case 21: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b5 >> 0) & 0xff; \
        case 20: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b4 >> 24) & 0xff; \
        case 19: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b4 >> 16) & 0xff; \
        case 18: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b4 >> 8) & 0xff; \
        case 17: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b4 >> 0) & 0xff; \
        case 16: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b3 >> 24) & 0xff; \
        case 15: \
            dfp[search_index * PASSWORD_LENGTH + 14] = (b3 >> 16) & 0xff; \
        case 14: \
            dfp[search_index * PASSWORD_LENGTH + 13] = (b3 >> 8) & 0xff; \
        case 13: \
            dfp[search_index * PASSWORD_LENGTH + 12] = (b3 >> 0) & 0xff; \
        case 12: \
            dfp[search_index * PASSWORD_LENGTH + 11] = (b2 >> 24) & 0xff; \
        case 11: \
            dfp[search_index * PASSWORD_LENGTH + 10] = (b2 >> 16) & 0xff; \
        case 10: \
            dfp[search_index * PASSWORD_LENGTH + 9] = (b2 >> 8) & 0xff; \
        case 9: \
            dfp[search_index * PASSWORD_LENGTH + 8] = (b2 >> 0) & 0xff; \
        case 8: \
            dfp[search_index * PASSWORD_LENGTH + 7] = (b1 >> 24) & 0xff; \
        case 7: \
            dfp[search_index * PASSWORD_LENGTH + 6] = (b1 >> 16) & 0xff; \
        case 6: \
            dfp[search_index * PASSWORD_LENGTH + 5] = (b1 >> 8) & 0xff; \
        case 5: \
            dfp[search_index * PASSWORD_LENGTH + 4] = (b1 >> 0) & 0xff; \
        case 4: \
            dfp[search_index * PASSWORD_LENGTH + 3] = (b0 >> 24) & 0xff; \
        case 3: \
            dfp[search_index * PASSWORD_LENGTH + 2] = (b0 >> 16) & 0xff; \
        case 2: \
            dfp[search_index * PASSWORD_LENGTH + 1] = (b0 >> 8) & 0xff; \
        case 1: \
            dfp[search_index * PASSWORD_LENGTH + 0] = (b0 >> 0) & 0xff; \
    } \
    deviceGlobalFoundPasswordFlagsPlainMD5[search_index] = (unsigned char) 1; \
}
#else
#define CopyFoundPasswordToMemory(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 55: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b13.s##suffix >> 16) & 0xff; \
        case 54: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b13.s##suffix >> 8) & 0xff; \
        case 53: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b13.s##suffix >> 0) & 0xff; \
        case 52: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b12.s##suffix >> 24) & 0xff; \
        case 51: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b12.s##suffix >> 16) & 0xff; \
        case 50: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b12.s##suffix >> 8) & 0xff; \
        case 49: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b12.s##suffix >> 0) & 0xff; \
        case 48: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b11.s##suffix >> 24) & 0xff; \
        case 47: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b11.s##suffix >> 16) & 0xff; \
        case 46: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b11.s##suffix >> 8) & 0xff; \
        case 45: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b11.s##suffix >> 0) & 0xff; \
        case 44: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b10.s##suffix >> 24) & 0xff; \
        case 43: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b10.s##suffix >> 16) & 0xff; \
        case 42: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b10.s##suffix >> 8) & 0xff; \
        case 41: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b10.s##suffix >> 0) & 0xff; \
        case 40: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b9.s##suffix >> 24) & 0xff; \
        case 39: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b9.s##suffix >> 16) & 0xff; \
        case 38: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b9.s##suffix >> 8) & 0xff; \
        case 37: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b9.s##suffix >> 0) & 0xff; \
        case 36: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b8.s##suffix >> 24) & 0xff; \
        case 35: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b8.s##suffix >> 16) & 0xff; \
        case 34: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b8.s##suffix >> 8) & 0xff; \
        case 33: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b8.s##suffix >> 0) & 0xff; \
        case 32: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b7.s##suffix >> 24) & 0xff; \
        case 31: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b7.s##suffix >> 16) & 0xff; \
        case 30: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b7.s##suffix >> 8) & 0xff; \
        case 29: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b7.s##suffix >> 0) & 0xff; \
        case 28: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b6.s##suffix >> 24) & 0xff; \
        case 27: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b6.s##suffix >> 16) & 0xff; \
        case 26: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b6.s##suffix >> 8) & 0xff; \
        case 25: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b6.s##suffix >> 0) & 0xff; \
        case 24: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b5.s##suffix >> 24) & 0xff; \
        case 23: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b5.s##suffix >> 16) & 0xff; \
        case 22: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b5.s##suffix >> 8) & 0xff; \
        case 21: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b5.s##suffix >> 0) & 0xff; \
        case 20: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b4.s##suffix >> 24) & 0xff; \
        case 19: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b4.s##suffix >> 16) & 0xff; \
        case 18: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b4.s##suffix >> 8) & 0xff; \
        case 17: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b4.s##suffix >> 0) & 0xff; \
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
    deviceGlobalFoundPasswordFlagsPlainMD5[search_index] = (unsigned char) 1; \
}
#endif


#ifdef grt_vector_1
#define CheckPassword128LE(dgh, dfp, dfpf, dnh, suffix) { \
    search_high = dnh; \
    search_low = 0; \
    while (search_low < search_high) { \
        search_index = search_low + (search_high - search_low) / 2; \
        current_hash_value = dgh[4 * search_index]; \
        if (current_hash_value < a) { \
            search_low = search_index + 1; \
        } else { \
            search_high = search_index; \
        } \
        if ((a == current_hash_value) && (search_low < dnh)) { \
            break; \
        } \
    } \
    if (a == current_hash_value) { \
        while (search_index && (a == dgh[(search_index - 1) * 4])) { \
            search_index--; \
        } \
        while ((a == dgh[search_index * 4])) { \
            if (b == dgh[search_index * 4 + 1]) { \
                if (c == dgh[search_index * 4 + 2]) { \
                    if (d == dgh[search_index * 4 + 3]) { \
                    CopyFoundPasswordToMemory(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}
#else
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
                    CopyFoundPasswordToMemory(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}
#endif


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_MD5(
    __constant unsigned char const * restrict deviceCharsetPlainMD5, /* 0 */
    __constant unsigned char const * restrict deviceReverseCharsetPlainMD5, /* 1 */
    __constant unsigned char const * restrict charsetLengthsPlainMD5, /* 2 */
    __constant unsigned char const * restrict constantBitmapAPlainMD5, /* 3 */
        
    __private unsigned long const numberOfHashesPlainMD5, /* 4 */
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainMD5, /* 5 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlainMD5, /* 6 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainMD5, /* 7 */
        
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainMD5, /* 8 */
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainMD5, /* 9 */
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainMD5, /* 10 */
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainMD5, /* 11 */
        
    __global   unsigned char *deviceGlobalStartPointsPlainMD5, /* 12 */
    __private unsigned long const deviceNumberThreadsPlainMD5, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainMD5, /* 14 */
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlainMD5, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainMD5 /* 16 */
) {
    // Start the kernel.
    //__local unsigned char sharedCharsetPlainMD5[MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    //__local unsigned char sharedReverseCharsetPlainMD5[MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    //__local unsigned char sharedCharsetLengthsPlainMD5[PASSWORD_LENGTH];
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
                deviceCharsetPlainMD5[0], deviceCharsetPlainMD5[1], deviceCharsetPlainMD5[2]);
        printf("Charset lengths: %d %d %d...\n", charsetLengthsPlainMD5[0], 
                charsetLengthsPlainMD5[1], charsetLengthsPlainMD5[2]);
        printf("Number hashes: %d\n", numberOfHashesPlainMD5);
        printf("Bitmap A: %lu\n", deviceGlobalBitmapAPlainMD5);
        printf("Bitmap B: %lu\n", deviceGlobalBitmapBPlainMD5);
        printf("Bitmap C: %lu\n", deviceGlobalBitmapCPlainMD5);
        printf("Bitmap D: %lu\n", deviceGlobalBitmapDPlainMD5);
        printf("Number threads: %lu\n", deviceNumberThreadsPlainMD5);
        printf("Steps to run: %u\n", deviceNumberStepsToRunPlainMD5);
        printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        
        int i, j;
        
        //for (i = 0; i < (deviceNumberThreadsPlainMD5 * PASSWORD_LENGTH); i++) {
        //    printf("%c", deviceGlobalStartPointsPlainMD5[i]);
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
            sharedBitmap[counter] = constantBitmapAPlainMD5[counter];
        }/*
#pragma unroll 128
        for (counter = 0; counter < (MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * PASSWORD_LENGTH); counter++) {
            sharedCharsetPlainMD5[counter] = deviceCharsetPlainMD5[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < (MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * PASSWORD_LENGTH); counter++) {
            sharedReverseCharsetPlainMD5[counter] = deviceReverseCharsetPlainMD5[counter];
        }
        for (counter = 0; counter < PASSWORD_LENGTH; counter++) {
            sharedCharsetLengthsPlainMD5[counter] = charsetLengthsPlainMD5[counter];
        }*/
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    b14 = (vector_type) (PASSWORD_LENGTH * 8);
    a = b = c = d = (vector_type) 0;

    b0 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[0]);
    if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[1 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[2 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[3 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 15) {b4 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[4 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 19) {b5 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[5 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 23) {b6 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[6 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 27) {b7 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[7 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 31) {b8 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[8 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 35) {b9 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[9 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 39) {b10 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[10 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 43) {b11 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[11 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 47) {b12 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[12 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 51) {b13 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[13 * deviceNumberThreadsPlainMD5]);}
        
    while (password_count < deviceNumberStepsToRunPlainMD5) {
        MD5_FIRST_3_ROUNDS();
        // All the lengths need these.
        MD5II(a, b, c, d, b0, MD5S41, 0xf4292244);
        MD5II(d, a, b, c, b7, MD5S42, 0x432aff97);
        MD5II(c, d, a, b, b14, MD5S43, 0xab9423a7);
        MD5II(b, c, d, a, b5, MD5S44, 0xfc93a039);
        MD5II(a, b, c, d, b12, MD5S41, 0x655b59c3);
        // If the password length is <= 8, we can start the early out process 
        // at this point and perform the final 3 checks internally.
        if (PASSWORD_LENGTH <= 8) {
            OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainMD5, 
                deviceGlobalBitmapBPlainMD5, deviceGlobalBitmapCPlainMD5, 
                deviceGlobalBitmapDPlainMD5, deviceGlobalHashlistAddressPlainMD5, 
                deviceGlobalFoundPasswordsPlainMD5, deviceGlobalFoundPasswordFlagsPlainMD5,
                numberOfHashesPlainMD5, deviceGlobal256kbBitmapAPlainMD5);
        } else {
            // Longer than 8 - need to keep on rolling.
            MD5II (d, a, b, c, b3, MD5S42, 0x8f0ccc92); 
            MD5II (c, d, a, b, b10, MD5S43, 0xffeff47d);
            MD5II (b, c, d, a, b1, MD5S44, 0x85845dd1);
            MD5II (a, b, c, d, b8, MD5S41, 0x6fa87e4f);
            MD5II (d, a, b, c, b15, MD5S42, 0xfe2ce6e0);
            MD5II (c, d, a, b, b6, MD5S43, 0xa3014314);
            MD5II (b, c, d, a, b13, MD5S44, 0x4e0811a1);
            MD5II (a, b, c, d, b4, MD5S41, 0xf7537e82); 
            OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainMD5, 
                deviceGlobalBitmapBPlainMD5, deviceGlobalBitmapCPlainMD5, 
                deviceGlobalBitmapDPlainMD5, deviceGlobalHashlistAddressPlainMD5, 
                deviceGlobalFoundPasswordsPlainMD5, deviceGlobalFoundPasswordFlagsPlainMD5,
                numberOfHashesPlainMD5, deviceGlobal256kbBitmapAPlainMD5);
        }
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
        
        
        OpenCLNoMemPasswordIncrementorLE();
        
        password_count++; 
    }
    vstore_type(b0, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[0]);
    if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[1 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[2 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[3 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 15) {vstore_type(b4, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[4 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 19) {vstore_type(b5, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[5 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 23) {vstore_type(b6, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[6 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 27) {vstore_type(b7, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[7 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 31) {vstore_type(b8, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[8 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 35) {vstore_type(b9, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[9 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 39) {vstore_type(b10, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[10 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 43) {vstore_type(b11, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[11 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 47) {vstore_type(b12, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[12 * deviceNumberThreadsPlainMD5]);}
    if (PASSWORD_LENGTH > 51) {vstore_type(b13, get_global_id(0), &deviceGlobalStartPasswordsPlainMD5[13 * deviceNumberThreadsPlainMD5]);}
  
}
