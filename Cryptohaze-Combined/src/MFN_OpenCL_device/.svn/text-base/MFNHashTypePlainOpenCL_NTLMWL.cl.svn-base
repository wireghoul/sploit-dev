
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
    #define convert_type 
#elif VECTOR_WIDTH == 2
    #define vector_type uint2
    #define vload_type vload2
    #define vstore_type vstore2
    #define grt_vector_2 1
    #define convert_type convert_uint2
#elif VECTOR_WIDTH == 4
    #define vector_type uint4
    #define vload_type vload4
    #define vstore_type vstore4
    #define grt_vector_4 1
    #define convert_type convert_uint4
#elif VECTOR_WIDTH == 8
    #define vector_type uint8
    #define vload_type vload8
    #define vstore_type vstore8
    #define grt_vector_8 1
    #define convert_type convert_uint8
#elif VECTOR_WIDTH == 16
    #define vector_type uint16
    #define vload_type vload16
    #define vstore_type vstore16
    #define grt_vector_16 1
    #define convert_type convert_uint16
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


#define MFN_HASH_TYPE_WORDLIST_MAX_PASSLEN 128

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
#define MD4G(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
//#define MD4G(d, c, b) bitselect((b), ((d) | (c)), ((c) & (d)))
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


#define MD4_FIRST_ROUND() { \
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
MD4HH (b, c, d, a, b15, MD4S34); \
a += (vector_type)0x67452301; \
b += (vector_type)0xefcdab89; \
c += (vector_type)0x98badcfe; \
d += (vector_type)0x10325476; \
}

#define MD4_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d) { \
MD4FF (a, b, c, d, 0, MD4S11); \
MD4FF (d, a, b, c, 0, MD4S12); \
MD4FF (c, d, a, b, 0, MD4S13); \
MD4FF (b, c, d, a, 0, MD4S14); \
MD4FF (a, b, c, d, 0, MD4S11); \
MD4FF (d, a, b, c, 0, MD4S12); \
MD4FF (c, d, a, b, 0, MD4S13); \
MD4FF (b, c, d, a, 0, MD4S14); \
MD4FF (a, b, c, d, 0, MD4S11); \
MD4FF (d, a, b, c, 0, MD4S12); \
MD4FF (c, d, a, b, 0, MD4S13); \
MD4FF (b, c, d, a, 0, MD4S14); \
MD4FF (a, b, c, d, 0, MD4S11); \
MD4FF (d, a, b, c, 0, MD4S12); \
MD4FF (c, d, a, b, b14, MD4S13); \
MD4FF (b, c, d, a, 0, MD4S14); \
MD4GG (a, b, c, d, 0, MD4S21); \
MD4GG (d, a, b, c, 0, MD4S22); \
MD4GG (c, d, a, b, 0, MD4S23); \
MD4GG (b, c, d, a, 0, MD4S24); \
MD4GG (a, b, c, d, 0, MD4S21); \
MD4GG (d, a, b, c, 0, MD4S22); \
MD4GG (c, d, a, b, 0, MD4S23); \
MD4GG (b, c, d, a, 0, MD4S24); \
MD4GG (a, b, c, d, 0, MD4S21); \
MD4GG (d, a, b, c, 0, MD4S22); \
MD4GG (c, d, a, b, 0, MD4S23); \
MD4GG (b, c, d, a, b14, MD4S24); \
MD4GG (a, b, c, d, 0, MD4S21); \
MD4GG (d, a, b, c, 0, MD4S22); \
MD4GG (c, d, a, b, 0, MD4S23); \
MD4GG (b, c, d, a, 0, MD4S24); \
MD4HH (a, b, c, d, 0, MD4S31); \
MD4HH (d, a, b, c, 0, MD4S32); \
MD4HH (c, d, a, b, 0, MD4S33); \
MD4HH (b, c, d, a, 0, MD4S34); \
MD4HH (a, b, c, d, 0, MD4S31); \
MD4HH (d, a, b, c, 0, MD4S32); \
MD4HH (c, d, a, b, 0, MD4S33); \
MD4HH (b, c, d, a, b14, MD4S34); \
MD4HH (a, b, c, d, 0, MD4S31); \
MD4HH (d, a, b, c, 0, MD4S32); \
MD4HH (c, d, a, b, 0, MD4S33); \
MD4HH (b, c, d, a, 0, MD4S34); \
MD4HH (a, b, c, d, 0, MD4S31); \
MD4HH (d, a, b, c, 0, MD4S32); \
MD4HH (c, d, a, b, 0, MD4S33); \
MD4HH (b, c, d, a, 0, MD4S34); \
a += (vector_type)prev_a; \
b += (vector_type)prev_b; \
c += (vector_type)prev_c; \
d += (vector_type)prev_d; \
}

#define MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d) { \
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
MD4HH (b, c, d, a, b15, MD4S34); \
a += (vector_type)prev_a; \
b += (vector_type)prev_b; \
c += (vector_type)prev_c; \
d += (vector_type)prev_d; \
}


// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
#ifdef grt_vector_1
#define CopyFoundPasswordToMemoryFromWordlist(dfp, dfpf, suffix) { \
    vector_type passwordLength = convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreadsPlainNTLM])); \
    vector_type wordlistBlock; \
    for (uint i = 0; i < passwordLength; i++) { \
        if ((i % 4) == 0) { \
            wordlistBlock = vload_type(get_global_id(0), \
                &deviceWordlistBlocks[(i / 4) * deviceNumberWords + \
                passwordStep * deviceNumberThreadsPlainNTLM]); \
        } \
        dfp[search_index * MFN_HASH_TYPE_WORDLIST_MAX_PASSLEN + i] = (wordlistBlock >> ((i % 4) * 8)) & 0xff; \
    } \
    deviceGlobalFoundPasswordFlagsPlainNTLM[search_index] = (unsigned char) 1; \
}
#else
#define CopyFoundPasswordToMemoryFromWordlist(dfp, dfpf, suffix) { \
    vector_type passwordLength = convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreadsPlainNTLM])); \
    vector_type wordlistBlock; \
    for (uint i = 0; i < passwordLength.s##suffix; i++) { \
        if ((i % 4) == 0) { \
            wordlistBlock = vload_type(get_global_id(0), \
                &deviceWordlistBlocks[(i / 4) * deviceNumberWords + \
                passwordStep * deviceNumberThreadsPlainNTLM]); \
        } \
        dfp[search_index * MFN_HASH_TYPE_WORDLIST_MAX_PASSLEN + i] = (wordlistBlock.s##suffix >> ((i % 4) * 8)) & 0xff; \
    } \
    deviceGlobalFoundPasswordFlagsPlainNTLM[search_index] = (unsigned char) 1; \
}
#endif


#ifdef grt_vector_1
#define CheckWordlistPassword128LE(dgh, dfp, dfpf, dnh, suffix) { \
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
                    CopyFoundPasswordToMemoryFromWordlist(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}
#else
#define CheckWordlistPassword128LE(dgh, dfp, dfpf, dnh, suffix) { \
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
                    CopyFoundPasswordToMemoryFromWordlist(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}
#endif

// Loads a full NTLM block into registers, starting at offset start_offset.
#define LOAD_NTLM_FULL_BLOCK(start_offset) { \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[(start_offset + 0) * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]); \
    b0 = (b15 & 0xff) | ((b15 & 0xff00) << 8); \
    b1 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8); \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[(start_offset + 1) * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]); \
    b2 = (b15 & 0xff) | ((b15 & 0xff00) << 8); \
    b3 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8); \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[(start_offset + 2) * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]); \
    b4 = (b15 & 0xff) | ((b15 & 0xff00) << 8); \
    b5 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8); \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[(start_offset + 3) * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]); \
    b6 = (b15 & 0xff) | ((b15 & 0xff00) << 8); \
    b7 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8); \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[(start_offset + 4) * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]); \
    b8 = (b15 & 0xff) | ((b15 & 0xff00) << 8); \
    b9 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8); \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[(start_offset + 5) * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]); \
    b10 = (b15 & 0xff) | ((b15 & 0xff00) << 8); \
    b11 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8); \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[(start_offset + 6) * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]); \
    b12 = (b15 & 0xff) | ((b15 & 0xff00) << 8); \
    b13 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8); \
    b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[(start_offset + 7) * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]); \
    b14 = (b15 & 0xff) | ((b15 & 0xff00) << 8); \
    b15 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8); \
}

// Define these here.  It should make the subsequent code a LOT cleaner.
#define __OPENCL_NTLMWL_KERNEL_ARGS__ \
    __constant unsigned char const * restrict deviceCharsetPlainNTLM, /* 0 */ \
    __constant unsigned char const * restrict deviceReverseCharsetPlainNTLM, /* 1 */ \
    __constant unsigned char const * restrict charsetLengthsPlainNTLM, /* 2 */ \
    __constant unsigned char const * restrict constantBitmapAPlainNTLM, /* 3 */ \
\
    __private unsigned long const numberOfHashesPlainNTLM, /* 4 */ \
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainNTLM, /* 5 */ \
    __global   unsigned char *deviceGlobalFoundPasswordsPlainNTLM, /* 6 */ \
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainNTLM, /* 7 */ \
\
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainNTLM, /* 8 */ \
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainNTLM, /* 9 */ \
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainNTLM, /* 10 */ \
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainNTLM, /* 11 */ \
\
    __global   unsigned char *deviceGlobalStartPointsPlainNTLM, /* 12 */ \
    __private unsigned long const deviceNumberThreadsPlainNTLM, /* 13 */ \
    __private unsigned int const deviceNumberStepsToRunPlainNTLM, /* 14 */ \
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlainNTLM, /* 15 */ \
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainNTLM, /* 16 */ \
    __global   unsigned char const * restrict deviceWordlistLengths, /* 17 */ \
    __global   unsigned int const * restrict deviceWordlistBlocks, /* 18 */ \
    __private unsigned int const deviceNumberWords, /* 19 */ \
    __private unsigned int const deviceStartStep, /* 20 */ \
    __private unsigned char const deviceNumberBlocksPerWord /* 21 */


// For 1-4 input blocks (passwords length 0-8)
__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_NTLMWL_B1_4(
    __OPENCL_NTLMWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

#if CPU_DEBUG 
    printf("Kernel start, global id %d\n", get_global_id(0));
    
    if (get_global_id(0) == 0) {
        printf("Number hashes: %d\n", numberOfHashesPlainNTLM);
        printf("Number threads: %lu\n", deviceNumberThreadsPlainNTLM);
        printf("Steps to run: %u\n", deviceNumberStepsToRunPlainNTLM);
        printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        printf("Start Step: %d\n", deviceStartStep);
        printf("Blocks per word: %d\n", deviceNumberBlocksPerWord);
        printf("Number words: %d\n", deviceNumberWords);
    }
#endif
    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainNTLM[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainNTLM) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
#if CPU_DEBUG 
        printf("Thread %d loading b0 from block %d\n", 
                get_global_id(0), 
                get_global_id(0) + 0 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM);
#endif        
        b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
        b0 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
        b1 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
        
        if (deviceNumberBlocksPerWord >= 2) {
            b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b2 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
            b3 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 3) {
            b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b4 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
            b5 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 4) {
            b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b6 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
            b7 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
        }
        // Load length.
        b14 = 16 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreadsPlainNTLM]));
        
        MD4_FIRST_ROUND();
        
        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
            deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
            deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
            deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
            numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);

        password_count++;
        passwordStep++;
    }
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_NTLMWL_B5_7(
    __OPENCL_NTLMWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

#if CPU_DEBUG 
    printf("Kernel start, global id %d\n", get_global_id(0));
    
    if (get_global_id(0) == 0) {
        printf("Number hashes: %d\n", numberOfHashesPlainNTLM);
        printf("Number threads: %lu\n", deviceNumberThreadsPlainNTLM);
        printf("Steps to run: %u\n", deviceNumberStepsToRunPlainNTLM);
        printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        printf("Start Step: %d\n", deviceStartStep);
        printf("Blocks per word: %d\n", deviceNumberBlocksPerWord);
        printf("Number words: %d\n", deviceNumberWords);
    }
#endif
    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainNTLM[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainNTLM) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
#if CPU_DEBUG 
        printf("Thread %d loading b0 from block %d\n", 
                get_global_id(0), 
                get_global_id(0) + 0 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM);
#endif        
        b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[0 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
        b0 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
        b1 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
        b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[1 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
        b2 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
        b3 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
        b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[2 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
        b4 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
        b5 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
        b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[3 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
        b6 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
        b7 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);

        if (deviceNumberBlocksPerWord >= 5) {
            b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[4 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b8 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
            b9 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 6) {
            b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[5 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b10 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
            b11 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 7) {
            b14 = vload_type(get_global_id(0), &deviceWordlistBlocks[6 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b12 = (b14 & 0xff) | ((b14 & 0xff00) << 8);
            b13 = ((b14 & 0xff0000) >> 16) | ((b14 & 0xff000000) >> 8);
        }
            
        // Load length.
        b14 = 16 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreadsPlainNTLM]));
        
        MD4_FIRST_ROUND();
        
        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
            deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
            deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
            deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
            numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);

        password_count++;
        passwordStep++;
    }
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_NTLMWL_B8(
    __OPENCL_NTLMWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

#if CPU_DEBUG 
    printf("Kernel start, global id %d\n", get_global_id(0));
    
    if (get_global_id(0) == 0) {
        printf("Number hashes: %d\n", numberOfHashesPlainNTLM);
        printf("Number threads: %lu\n", deviceNumberThreadsPlainNTLM);
        printf("Steps to run: %u\n", deviceNumberStepsToRunPlainNTLM);
        printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
        printf("VECTOR_WIDTH: %d\n", VECTOR_WIDTH);
        printf("Start Step: %d\n", deviceStartStep);
        printf("Blocks per word: %d\n", deviceNumberBlocksPerWord);
        printf("Number words: %d\n", deviceNumberWords);
    }
#endif
    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainNTLM[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainNTLM) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
#if CPU_DEBUG 
        printf("Thread %d loading b0 from block %d\n", 
                get_global_id(0), 
                get_global_id(0) + 0 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM);
#endif        
        LOAD_NTLM_FULL_BLOCK(0);
        
        MD4_FIRST_ROUND();
        
        prev_a = a; prev_b = b; prev_c = c; prev_d = d;

        // Load length.
        b14 = 16 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreadsPlainNTLM]));

        MD4_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);
        
        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
            deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
            deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
            deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
            numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);

        password_count++;
        passwordStep++;
    }
}



__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_NTLMWL_B9_15(
    __OPENCL_NTLMWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainNTLM[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainNTLM) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
#if CPU_DEBUG 
        printf("Thread %d loading b0 from block %d\n", 
                get_global_id(0), 
                get_global_id(0) + 0 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM);
#endif
        LOAD_NTLM_FULL_BLOCK(0);

        MD4_FIRST_ROUND();

        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;

        if (deviceNumberBlocksPerWord >= 9) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[8 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b0 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b1 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 10) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[9 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b2 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b3 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 11) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[10 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b4 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b5 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 12) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[11 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b6 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b7 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 13) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[12 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b8 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b9 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 14) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[13 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b10 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b11 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 15) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[14 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b12 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b13 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        
        b14 = 16 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreadsPlainNTLM]));
        b15 = 0;
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);

        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
            deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
            deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
            deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
            numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);

        password_count++;
        passwordStep++;
    }
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_NTLMWL_B16(
    __OPENCL_NTLMWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainNTLM[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainNTLM) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
#if CPU_DEBUG 
        printf("Thread %d loading b0 from block %d\n", 
                get_global_id(0), 
                get_global_id(0) + 0 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM);
#endif
        LOAD_NTLM_FULL_BLOCK(0);

        MD4_FIRST_ROUND();

        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;

        LOAD_NTLM_FULL_BLOCK(8);
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);

        prev_a = a; prev_b = b; prev_c = c; prev_d = d;

        // Load length.
        b14 = 16 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreadsPlainNTLM]));

        MD4_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);

        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
            deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
            deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
            deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
            numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);

        password_count++;
        passwordStep++;
    }
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_NTLMWL_B17_23(
    __OPENCL_NTLMWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainNTLM[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainNTLM) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
#if CPU_DEBUG 
        printf("Thread %d loading b0 from block %d\n", 
                get_global_id(0), 
                get_global_id(0) + 0 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM);
#endif
        LOAD_NTLM_FULL_BLOCK(0);

        MD4_FIRST_ROUND();

        prev_a = a; prev_b = b; prev_c = c; prev_d = d;

        LOAD_NTLM_FULL_BLOCK(8);
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
        
        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;

        if (deviceNumberBlocksPerWord >= 17) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[16 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b0 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b1 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 18) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[17 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b2 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b3 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 19) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[18 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b4 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b5 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 20) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[19 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b6 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b7 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 21) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[20 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b8 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b9 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 22) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[21 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b10 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b11 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 23) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[22 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b12 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b13 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        
        b14 = 16 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreadsPlainNTLM]));
        b15 = 0;
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);

        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
            deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
            deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
            deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
            numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);

        password_count++;
        passwordStep++;
    }
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_NTLMWL_B24(
    __OPENCL_NTLMWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainNTLM[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainNTLM) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
#if CPU_DEBUG 
        printf("Thread %d loading b0 from block %d\n", 
                get_global_id(0), 
                get_global_id(0) + 0 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM);
#endif
        LOAD_NTLM_FULL_BLOCK(0);

        MD4_FIRST_ROUND();

        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;

        LOAD_NTLM_FULL_BLOCK(8);
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);

        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;

        LOAD_NTLM_FULL_BLOCK(16);
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);


        prev_a = a; prev_b = b; prev_c = c; prev_d = d;

        // Load length.
        b14 = 16 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreadsPlainNTLM]));

        MD4_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);

        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
            deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
            deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
            deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
            numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);

        password_count++;
        passwordStep++;
    }
}



__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_NTLMWL_B25_31(
    __OPENCL_NTLMWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainNTLM[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainNTLM) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
#if CPU_DEBUG 
        printf("Thread %d loading b0 from block %d\n", 
                get_global_id(0), 
                get_global_id(0) + 0 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM);
#endif
        LOAD_NTLM_FULL_BLOCK(0);

        MD4_FIRST_ROUND();

        prev_a = a; prev_b = b; prev_c = c; prev_d = d;

        LOAD_NTLM_FULL_BLOCK(8);
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);

        prev_a = a; prev_b = b; prev_c = c; prev_d = d;

        LOAD_NTLM_FULL_BLOCK(16);
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
        
        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;

        if (deviceNumberBlocksPerWord >= 25) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[24 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b0 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b1 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 26) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[25 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b2 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b3 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 27) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[26 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b4 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b5 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 28) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[27 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b6 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b7 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 29) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[28 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b8 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b9 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 30) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[29 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b10 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b11 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        if (deviceNumberBlocksPerWord >= 31) {
            b15 = vload_type(get_global_id(0), &deviceWordlistBlocks[30 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM]);
            b12 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
            b13 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
        }
        
        b14 = 16 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreadsPlainNTLM]));
        b15 = 0;
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);

        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
            deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
            deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
            deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
            numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);

        password_count++;
        passwordStep++;
    }
}


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_NTLMWL_B32(
    __OPENCL_NTLMWL_KERNEL_ARGS__
) {
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, bitmap_index;
    vector_type prev_a, prev_b, prev_c, prev_d;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    uint passwordStep = deviceStartStep;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainNTLM[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop until the number of requested steps are completed.
    while (password_count < deviceNumberStepsToRunPlainNTLM) {
        // If the password step is beyond the number of loaded, exit.
        // This should be checked.

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        // Load passwords
#if CPU_DEBUG 
        printf("Thread %d loading b0 from block %d\n", 
                get_global_id(0), 
                get_global_id(0) + 0 * deviceNumberWords + passwordStep * deviceNumberThreadsPlainNTLM);
#endif
        LOAD_NTLM_FULL_BLOCK(0);

        MD4_FIRST_ROUND();

        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;

        LOAD_NTLM_FULL_BLOCK(8);
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);

        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;

        LOAD_NTLM_FULL_BLOCK(16);
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);

        prev_a = a; prev_b = b; prev_c = c; prev_d = d;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;

        LOAD_NTLM_FULL_BLOCK(24);
        
        MD4_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);


        prev_a = a; prev_b = b; prev_c = c; prev_d = d;

        // Load length.
        b14 = 16 * convert_type(vload_type(get_global_id(0), &deviceWordlistLengths[passwordStep * deviceNumberThreadsPlainNTLM]));

        MD4_ZERO_ROUND(prev_a, prev_b, prev_c, prev_d);

        OpenCLPasswordCheckWordlist128(sharedBitmap, deviceGlobalBitmapAPlainNTLM, 
            deviceGlobalBitmapBPlainNTLM, deviceGlobalBitmapCPlainNTLM, 
            deviceGlobalBitmapDPlainNTLM, deviceGlobalHashlistAddressPlainNTLM, 
            deviceGlobalFoundPasswordsPlainNTLM, deviceGlobalFoundPasswordFlagsPlainNTLM,
            numberOfHashesPlainNTLM, deviceGlobal256kbBitmapAPlainNTLM);

        password_count++;
        passwordStep++;
    }
}

