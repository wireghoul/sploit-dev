
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
    #define vector_type_uchar uchar2
    #define vload_type vload2
    #define vstore_type vstore2
    #define grt_vector_2 1
#elif VECTOR_WIDTH == 4
    #define vector_type uint4
    #define vector_type_uchar uchar4
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


/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_LOTUS_MAX_CHARSET_LENGTH 128

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
    deviceGlobalFoundPasswordFlagsPlainLOTUS[search_index] = (unsigned char) 1; \
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
    deviceGlobalFoundPasswordFlagsPlainLOTUS[search_index] = (unsigned char) 1; \
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



__constant unsigned char S_BOX[] = {
  0xBD,0x56,0xEA,0xF2,0xA2,0xF1,0xAC,0x2A,
  0xB0,0x93,0xD1,0x9C,0x1B,0x33,0xFD,0xD0,
  0x30,0x04,0xB6,0xDC,0x7D,0xDF,0x32,0x4B,
  0xF7,0xCB,0x45,0x9B,0x31,0xBB,0x21,0x5A,
  0x41,0x9F,0xE1,0xD9,0x4A,0x4D,0x9E,0xDA,
  0xA0,0x68,0x2C,0xC3,0x27,0x5F,0x80,0x36,
  0x3E,0xEE,0xFB,0x95,0x1A,0xFE,0xCE,0xA8,
  0x34,0xA9,0x13,0xF0,0xA6,0x3F,0xD8,0x0C,
  0x78,0x24,0xAF,0x23,0x52,0xC1,0x67,0x17,
  0xF5,0x66,0x90,0xE7,0xE8,0x07,0xB8,0x60,
  0x48,0xE6,0x1E,0x53,0xF3,0x92,0xA4,0x72,
  0x8C,0x08,0x15,0x6E,0x86,0x00,0x84,0xFA,
  0xF4,0x7F,0x8A,0x42,0x19,0xF6,0xDB,0xCD,
  0x14,0x8D,0x50,0x12,0xBA,0x3C,0x06,0x4E,
  0xEC,0xB3,0x35,0x11,0xA1,0x88,0x8E,0x2B,
  0x94,0x99,0xB7,0x71,0x74,0xD3,0xE4,0xBF,
  0x3A,0xDE,0x96,0x0E,0xBC,0x0A,0xED,0x77,
  0xFC,0x37,0x6B,0x03,0x79,0x89,0x62,0xC6,
  0xD7,0xC0,0xD2,0x7C,0x6A,0x8B,0x22,0xA3,
  0x5B,0x05,0x5D,0x02,0x75,0xD5,0x61,0xE3,
  0x18,0x8F,0x55,0x51,0xAD,0x1F,0x0B,0x5E,
  0x85,0xE5,0xC2,0x57,0x63,0xCA,0x3D,0x6C,
  0xB4,0xC5,0xCC,0x70,0xB2,0x91,0x59,0x0D,
  0x47,0x20,0xC8,0x4F,0x58,0xE0,0x01,0xE2,
  0x16,0x38,0xC4,0x6F,0x3B,0x0F,0x65,0x46,
  0xBE,0x7E,0x2D,0x7B,0x82,0xF9,0x40,0xB5,
  0x1D,0x73,0xF8,0xEB,0x26,0xC7,0x87,0x97,
  0x25,0x54,0xB1,0x28,0xAA,0x98,0x9D,0xA5,
  0x64,0x6D,0x7A,0xD4,0x10,0x81,0x44,0xEF,
  0x49,0xD6,0xAE,0x2E,0xDD,0x76,0x5C,0x2F,
  0xA7,0x1C,0xC9,0x09,0x69,0x9A,0x83,0xCF,
  0x29,0x39,0xB9,0xE9,0x4C,0xFF,0x43,0xAB
};

#define lotusCLSharedIndex(offset) (((offset) % 4) + (((offset) / 4) * THREADSPERBLOCK * 4) + (get_local_id(0) * 4))
#define lotusCLSharedIndex32(offset) ((offset) * THREADSPERBLOCK + get_local_id(0))

#ifdef grt_vector_1
#define KeyWordSbox() { \
    KeyWord = (KeyWord << 8) | S_S_BOX[sboxIndex]; \
}
#define KeyWordLoad() { \
    KeyWord = S_S_BOX[PwWord & 0x000000ff]; \
}
#define SBoxLookup(offset) { \
    sBoxValue = S_S_BOX[((offset - (4 * i)) + X) & 0xff]; \
}
#elif grt_vector_2
#define KeyWordSbox() { \
    KeyWord.s0 = (KeyWord.s0 << 8) | S_S_BOX[sboxIndex.s0]; \
    KeyWord.s1 = (KeyWord.s1 << 8) | S_S_BOX[sboxIndex.s1]; \
}
#define KeyWordLoad() { \
    KeyWord.s0 = S_S_BOX[PwWord.s0 & 0x000000ff]; \
    KeyWord.s1 = S_S_BOX[PwWord.s1 & 0x000000ff]; \
}
#define SBoxLookup(offset) { \
    sBoxValue.s0 = S_S_BOX[((offset - (4 * i)) + X.s0) & 0xff]; \
    sBoxValue.s1 = S_S_BOX[((offset - (4 * i)) + X.s1) & 0xff]; \
}
#elif grt_vector_4
#define KeyWordSbox() { \
    KeyWord.s0 = (KeyWord.s0 << 8) | S_S_BOX[sboxIndex.s0]; \
    KeyWord.s1 = (KeyWord.s1 << 8) | S_S_BOX[sboxIndex.s1]; \
    KeyWord.s2 = (KeyWord.s2 << 8) | S_S_BOX[sboxIndex.s2]; \
    KeyWord.s3 = (KeyWord.s3 << 8) | S_S_BOX[sboxIndex.s3]; \
}
#define KeyWordLoad() { \
    KeyWord.s0 = S_S_BOX[PwWord.s0 & 0x000000ff]; \
    KeyWord.s1 = S_S_BOX[PwWord.s1 & 0x000000ff]; \
    KeyWord.s2 = S_S_BOX[PwWord.s2 & 0x000000ff]; \
    KeyWord.s3 = S_S_BOX[PwWord.s3 & 0x000000ff]; \
}
#define SBoxLookup(offset) { \
    sBoxValue.s0 = S_S_BOX[((offset - (4 * i)) + X.s0) & 0xff]; \
    sBoxValue.s1 = S_S_BOX[((offset - (4 * i)) + X.s1) & 0xff]; \
    sBoxValue.s2 = S_S_BOX[((offset - (4 * i)) + X.s2) & 0xff]; \
    sBoxValue.s3 = S_S_BOX[((offset - (4 * i)) + X.s3) & 0xff]; \
}
#elif grt_vector_8
#define KeyWordSbox() { \
    KeyWord.s0 = (KeyWord.s0 << 8) | S_S_BOX[sboxIndex.s0]; \
    KeyWord.s1 = (KeyWord.s1 << 8) | S_S_BOX[sboxIndex.s1]; \
    KeyWord.s2 = (KeyWord.s2 << 8) | S_S_BOX[sboxIndex.s2]; \
    KeyWord.s3 = (KeyWord.s3 << 8) | S_S_BOX[sboxIndex.s3]; \
    KeyWord.s4 = (KeyWord.s4 << 8) | S_S_BOX[sboxIndex.s4]; \
    KeyWord.s5 = (KeyWord.s5 << 8) | S_S_BOX[sboxIndex.s5]; \
    KeyWord.s6 = (KeyWord.s6 << 8) | S_S_BOX[sboxIndex.s6]; \
    KeyWord.s7 = (KeyWord.s7 << 8) | S_S_BOX[sboxIndex.s7]; \
}
#define KeyWordLoad() { \
    KeyWord.s0 = S_S_BOX[PwWord.s0 & 0x000000ff]; \
    KeyWord.s1 = S_S_BOX[PwWord.s1 & 0x000000ff]; \
    KeyWord.s2 = S_S_BOX[PwWord.s2 & 0x000000ff]; \
    KeyWord.s3 = S_S_BOX[PwWord.s3 & 0x000000ff]; \
    KeyWord.s4 = S_S_BOX[PwWord.s4 & 0x000000ff]; \
    KeyWord.s5 = S_S_BOX[PwWord.s5 & 0x000000ff]; \
    KeyWord.s6 = S_S_BOX[PwWord.s6 & 0x000000ff]; \
    KeyWord.s7 = S_S_BOX[PwWord.s7 & 0x000000ff]; \
}
#define SBoxLookup(offset) { \
    sBoxValue.s0 = S_S_BOX[((offset - (4 * i)) + X.s0) & 0xff]; \
    sBoxValue.s1 = S_S_BOX[((offset - (4 * i)) + X.s1) & 0xff]; \
    sBoxValue.s2 = S_S_BOX[((offset - (4 * i)) + X.s2) & 0xff]; \
    sBoxValue.s3 = S_S_BOX[((offset - (4 * i)) + X.s3) & 0xff]; \
    sBoxValue.s4 = S_S_BOX[((offset - (4 * i)) + X.s4) & 0xff]; \
    sBoxValue.s5 = S_S_BOX[((offset - (4 * i)) + X.s5) & 0xff]; \
    sBoxValue.s6 = S_S_BOX[((offset - (4 * i)) + X.s6) & 0xff]; \
    sBoxValue.s7 = S_S_BOX[((offset - (4 * i)) + X.s7) & 0xff]; \
}
#else
#pragma error "Vector width not supported."
#endif


__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_LOTUS(
    __constant unsigned char const * restrict deviceCharsetPlainLOTUS, /* 0 */
    __constant unsigned char const * restrict deviceReverseCharsetPlainLOTUS, /* 1 */
    __constant unsigned char const * restrict charsetLengthsPlainLOTUS, /* 2 */
    __constant unsigned char const * restrict constantBitmapAPlainLOTUS, /* 3 */
        
    __private unsigned long const numberOfHashesPlainLOTUS, /* 4 */
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainLOTUS, /* 5 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlainLOTUS, /* 6 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainLOTUS, /* 7 */
        
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainLOTUS, /* 8 */
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainLOTUS, /* 9 */
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainLOTUS, /* 10 */
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainLOTUS, /* 11 */
        
    __global   unsigned char *deviceGlobalStartPointsPlainLOTUS, /* 12 */
    __private unsigned long const deviceNumberThreadsPlainLOTUS, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainLOTUS, /* 14 */
    __global   unsigned int * restrict deviceGlobalStartPasswordsPlainLOTUS, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainLOTUS /* 16 */
) {
    // Start the kernel.
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    __local unsigned char lotusWorkingSpace[THREADSPERBLOCK*VECTOR_WIDTH*64];
    __local unsigned int *lotusWorkingSpace32 = (__local unsigned int *)lotusWorkingSpace;
    __local unsigned char S_S_BOX[256];
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, a, b, c, d, bitmap_index;

    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;
    vector_type tmp;
    vector_type blockWord, newBlockWord, X;
    int i;


    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainLOTUS[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < 256; counter++) {
            S_S_BOX[counter] = S_BOX[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = (vector_type)0;
    a = b = c = d = (vector_type) 0;

    b0 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainLOTUS[0]);
    if (PASSWORD_LENGTH > 4) {b1 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainLOTUS[1 * deviceNumberThreadsPlainLOTUS]);}
    if (PASSWORD_LENGTH > 8) {b2 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainLOTUS[2 * deviceNumberThreadsPlainLOTUS]);}
    if (PASSWORD_LENGTH > 12) {b3 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainLOTUS[3 * deviceNumberThreadsPlainLOTUS]);}
        
    //printf("Lotus working space array size: %d\n", THREADSPERBLOCK*VECTOR_WIDTH*64);
    while (password_count < deviceNumberStepsToRunPlainLOTUS) {
        // Initial init of array values.
        vstore_type(0xd3503c3e, lotusCLSharedIndex32(0), lotusWorkingSpace32);
        vstore_type(0x5cc587ab, lotusCLSharedIndex32(1), lotusWorkingSpace32);
        vstore_type(0x9db9d4bc, lotusCLSharedIndex32(2), lotusWorkingSpace32);
        vstore_type(0x29d76e38, lotusCLSharedIndex32(3), lotusWorkingSpace32);
        
        {
            vector_type lengthPadding = (vector_type)
                    ((16 - PASSWORD_LENGTH) | ((16 - PASSWORD_LENGTH) << 8) |
                    ((16 - PASSWORD_LENGTH) << 16) | ((16 - PASSWORD_LENGTH) << 24));

            for (i = 4; i < 12; i++) {
                vstore_type(lengthPadding, lotusCLSharedIndex32(i), lotusWorkingSpace32);
            }
        }

        unsigned int passMask = 0;
        switch (PASSWORD_LENGTH % 4) {
            case 0:
                passMask = 0xffffffff;
                break;
            case 1:
                passMask = 0x000000ff;
                break;
            case 2:
                passMask = 0x0000ffff;
                break;
            case 3:
                passMask = 0x00ffffff;
                break;
        }
        
        if (PASSWORD_LENGTH <= 4) {
            tmp = vload_type(lotusCLSharedIndex32(4), lotusWorkingSpace32);
            tmp &= ~passMask;
            tmp |= (b0 & passMask);
            vstore_type(tmp, lotusCLSharedIndex32(4), lotusWorkingSpace32);
            tmp = vload_type(lotusCLSharedIndex32(8), lotusWorkingSpace32);
            tmp &= ~passMask;
            tmp |= (b0 & passMask);
            vstore_type(tmp, lotusCLSharedIndex32(8), lotusWorkingSpace32);
        } else if (PASSWORD_LENGTH <= 8) {
            vstore_type(b0,  lotusCLSharedIndex32(4), lotusWorkingSpace32);
            tmp = vload_type(lotusCLSharedIndex32(5), lotusWorkingSpace32);
            tmp &= ~passMask;
            tmp |= (b1 & passMask);
            vstore_type(tmp, lotusCLSharedIndex32(5), lotusWorkingSpace32);
            vstore_type(b0,  lotusCLSharedIndex32(8), lotusWorkingSpace32);
            tmp = vload_type(lotusCLSharedIndex32(9), lotusWorkingSpace32);
            tmp &= ~passMask;
            tmp |= (b1 & passMask);
            vstore_type(tmp, lotusCLSharedIndex32(9), lotusWorkingSpace32);
        } else if (PASSWORD_LENGTH <= 12) {
            vstore_type(b0,  lotusCLSharedIndex32(4), lotusWorkingSpace32);
            vstore_type(b1,  lotusCLSharedIndex32(5), lotusWorkingSpace32);
            tmp = vload_type(lotusCLSharedIndex32(6), lotusWorkingSpace32);
            tmp &= ~passMask;
            tmp |= (b2 & passMask);
            vstore_type(tmp, lotusCLSharedIndex32(6), lotusWorkingSpace32);
            vstore_type(b0,  lotusCLSharedIndex32(8), lotusWorkingSpace32);
            vstore_type(b1,  lotusCLSharedIndex32(9), lotusWorkingSpace32);
            tmp = vload_type(lotusCLSharedIndex32(10), lotusWorkingSpace32);
            tmp &= ~passMask;
            tmp |= (b2 & passMask);
            vstore_type(tmp, lotusCLSharedIndex32(10), lotusWorkingSpace32);
        } else if (PASSWORD_LENGTH <= 16) {
            vstore_type(b0,  lotusCLSharedIndex32(4), lotusWorkingSpace32);
            vstore_type(b1,  lotusCLSharedIndex32(5), lotusWorkingSpace32);
            vstore_type(b2,  lotusCLSharedIndex32(6), lotusWorkingSpace32);
            tmp = vload_type(lotusCLSharedIndex32(7), lotusWorkingSpace32);
            tmp &= ~passMask;
            tmp |= (b3 & passMask);
            vstore_type(tmp, lotusCLSharedIndex32(6), lotusWorkingSpace32);
            vstore_type(b0,  lotusCLSharedIndex32(8), lotusWorkingSpace32);
            vstore_type(b1,  lotusCLSharedIndex32(9), lotusWorkingSpace32);
            vstore_type(b2,  lotusCLSharedIndex32(10), lotusWorkingSpace32);
            tmp = vload_type(lotusCLSharedIndex32(11), lotusWorkingSpace32);
            tmp &= ~passMask;
            tmp |= (b3 & passMask);
            vstore_type(tmp, lotusCLSharedIndex32(11), lotusWorkingSpace32);
        }
        /*
        if (get_local_id(0) == 0) {
            printf("Password space size\n");
            printf("Working space size %d: \n", THREADSPERBLOCK*VECTOR_WIDTH*64);
            for (int i = 0; i < (THREADSPERBLOCK*VECTOR_WIDTH*64); i++) {
                if (i && ((i % 4 == 0))) {
                    printf(" ");
                }
                if (i && ((i % 16 == 0))) {
                    printf("\n");
                }
                printf("%02x ", lotusWorkingSpace[i]);
            }
            printf("\n\n");
        }
         */ 
        {
            vector_type KeyWord;
            vector_type PwWord;
            vector_type sboxIndex;

            PwWord = vload_type(lotusCLSharedIndex32(4), lotusWorkingSpace32);

            // 0
            KeyWordLoad();
            // 1
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();
            // 2
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();
            // 3
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();

            // Endian swap for storage.
            vstore_type((KeyWord << 24) | (KeyWord << 8 & 0xff0000) | (KeyWord >> 8 & 0xff00) | (KeyWord >> 24), 
                lotusCLSharedIndex32(12), lotusWorkingSpace32);
            
            PwWord = vload_type(lotusCLSharedIndex32(5), lotusWorkingSpace32);
            // 0
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();
            // 1
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();
            // 2
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();
            // 3
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();

            vstore_type((KeyWord << 24) | (KeyWord << 8 & 0xff0000) | (KeyWord >> 8 & 0xff00) | (KeyWord >> 24), 
                lotusCLSharedIndex32(13), lotusWorkingSpace32);

            PwWord = vload_type(lotusCLSharedIndex32(6), lotusWorkingSpace32);
            // 0
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();
            // 1
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();
            // 2
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();
            // 3
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();

            vstore_type((KeyWord << 24) | (KeyWord << 8 & 0xff0000) | (KeyWord >> 8 & 0xff00) | (KeyWord >> 24), 
                lotusCLSharedIndex32(14), lotusWorkingSpace32);

            PwWord = vload_type(lotusCLSharedIndex32(7), lotusWorkingSpace32);
            // 0
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();
            // 1
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();
            // 2
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();
            // 3
            PwWord = PwWord >> 8;
            sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
            KeyWordSbox();

            vstore_type((KeyWord << 24) | (KeyWord << 8 & 0xff0000) | (KeyWord >> 8 & 0xff00) | (KeyWord >> 24), 
                lotusCLSharedIndex32(15), lotusWorkingSpace32);
        }

        blockWord = vload_type(lotusCLSharedIndex32(3), lotusWorkingSpace32);
        X = (blockWord >> 24 & 0xff);
        //printf("X: %02x\n", X);
        vector_type sBoxValue;
        //#pragma unroll 8
        for (i = 4; i < 12; i++) {
            blockWord = vload_type(lotusCLSharedIndex32(i), lotusWorkingSpace32);
            newBlockWord = 0;
            SBoxLookup(48);
            X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            SBoxLookup(47);
            X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            SBoxLookup(46);
            X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            SBoxLookup(45);
            X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
            newBlockWord |= X;
            vstore_type((newBlockWord << 24) | (newBlockWord << 8 & 0xff0000) | (newBlockWord >> 8 & 0xff00) | (newBlockWord >> 24), 
                lotusCLSharedIndex32(i), lotusWorkingSpace32);
        }
        
        for (uint j = 17; j > 0; j--) {
            #pragma unroll 12
            for (i = 0; i < 12; i++) {
                blockWord = vload_type(lotusCLSharedIndex32(i), lotusWorkingSpace32);
                newBlockWord = 0;
                SBoxLookup(48);
                X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
                newBlockWord |= X;
                newBlockWord = newBlockWord << 8;
                blockWord = blockWord >> 8;
                SBoxLookup(47);
                X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
                newBlockWord |= X;
                newBlockWord = newBlockWord << 8;
                blockWord = blockWord >> 8;
                SBoxLookup(46);
                X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
                newBlockWord |= X;
                newBlockWord = newBlockWord << 8;
                blockWord = blockWord >> 8;
                SBoxLookup(45);
                X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
                newBlockWord |= X;
                vstore_type((newBlockWord << 24) | (newBlockWord << 8 & 0xff0000) | (newBlockWord >> 8 & 0xff00) | (newBlockWord >> 24), 
                    lotusCLSharedIndex32(i), lotusWorkingSpace32);
            }
        }

        // Do the swapping with 32-bit accesses instead of 64 bit
        {
            vstore_type(vload_type(lotusCLSharedIndex32(12), lotusWorkingSpace32), 
                lotusCLSharedIndex32(4), lotusWorkingSpace32);
            vstore_type(vload_type(lotusCLSharedIndex32(13), lotusWorkingSpace32), 
                lotusCLSharedIndex32(5), lotusWorkingSpace32);
            vstore_type(vload_type(lotusCLSharedIndex32(14), lotusWorkingSpace32), 
                lotusCLSharedIndex32(6), lotusWorkingSpace32);
            vstore_type(vload_type(lotusCLSharedIndex32(15), lotusWorkingSpace32), 
                lotusCLSharedIndex32(7), lotusWorkingSpace32);

            vstore_type(vload_type(lotusCLSharedIndex32(0), lotusWorkingSpace32) ^ 
                    vload_type(lotusCLSharedIndex32(12), lotusWorkingSpace32), 
                    lotusCLSharedIndex32(8), lotusWorkingSpace32);
            vstore_type(vload_type(lotusCLSharedIndex32(1), lotusWorkingSpace32) ^ 
                    vload_type(lotusCLSharedIndex32(13), lotusWorkingSpace32), 
                    lotusCLSharedIndex32(9), lotusWorkingSpace32);
            vstore_type(vload_type(lotusCLSharedIndex32(2), lotusWorkingSpace32) ^ 
                    vload_type(lotusCLSharedIndex32(14), lotusWorkingSpace32), 
                    lotusCLSharedIndex32(10), lotusWorkingSpace32);
            vstore_type(vload_type(lotusCLSharedIndex32(3), lotusWorkingSpace32) ^ 
                    vload_type(lotusCLSharedIndex32(15), lotusWorkingSpace32), 
                    lotusCLSharedIndex32(11), lotusWorkingSpace32);
        }

        X = 0;

        //#pragma unroll 16
        for (uint j = 17; j > 0; j--) {
            #pragma unroll 12
            for (i = 0; i < 12; i++) {
                vector_type sBoxValue;
                blockWord = vload_type(lotusCLSharedIndex32(i), lotusWorkingSpace32);
                newBlockWord = 0;
                SBoxLookup(48);
                X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
                newBlockWord |= X;
                newBlockWord = newBlockWord << 8;
                blockWord = blockWord >> 8;
                SBoxLookup(47);
                X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
                newBlockWord |= X;
                newBlockWord = newBlockWord << 8;
                blockWord = blockWord >> 8;
                SBoxLookup(46);
                X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
                newBlockWord |= X;
                newBlockWord = newBlockWord << 8;
                blockWord = blockWord >> 8;
                SBoxLookup(45);
                X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
                newBlockWord |= X;
                vstore_type((newBlockWord << 24) | (newBlockWord << 8 & 0xff0000) | (newBlockWord >> 8 & 0xff00) | (newBlockWord >> 24), 
                    lotusCLSharedIndex32(i), lotusWorkingSpace32);
            }
        }
        //#pragma unroll 4
        for (int i = 0; i < 4; i++) {
            vector_type sBoxValue;
            blockWord = vload_type(lotusCLSharedIndex32(i), lotusWorkingSpace32);
            newBlockWord = 0;
            SBoxLookup(48);
            X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            SBoxLookup(47);
            X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            SBoxLookup(46);
            X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            SBoxLookup(45);
            X = ((blockWord & 0xff) ^ sBoxValue) & 0xff;
            newBlockWord |= X;
            vstore_type((newBlockWord << 24) | (newBlockWord << 8 & 0xff0000) | (newBlockWord >> 8 & 0xff00) | (newBlockWord >> 24), 
                lotusCLSharedIndex32(i), lotusWorkingSpace32);
        }
        
        /*
        if (get_local_id(0) == 0) {
            printf("Working space size %d: \n", THREADSPERBLOCK*VECTOR_WIDTH*64);
            for (int i = 0; i < (THREADSPERBLOCK*VECTOR_WIDTH*64); i++) {
                if (i && ((i % 4 == 0))) {
                    printf(" ");
                }
                if (i && ((i % 16 == 0))) {
                    printf("\n");
                }
                printf("%02x ", lotusWorkingSpace[i]);
            }
            printf("\n\n");
        }
        */
        
        a = vload_type(lotusCLSharedIndex32(0), lotusWorkingSpace32);
        b = vload_type(lotusCLSharedIndex32(1), lotusWorkingSpace32);
        c = vload_type(lotusCLSharedIndex32(2), lotusWorkingSpace32);
        d = vload_type(lotusCLSharedIndex32(3), lotusWorkingSpace32);
        
        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainLOTUS, 
            deviceGlobalBitmapBPlainLOTUS, deviceGlobalBitmapCPlainLOTUS, 
            deviceGlobalBitmapDPlainLOTUS, deviceGlobalHashlistAddressPlainLOTUS, 
            deviceGlobalFoundPasswordsPlainLOTUS, deviceGlobalFoundPasswordFlagsPlainLOTUS,
            numberOfHashesPlainLOTUS, deviceGlobal256kbBitmapAPlainLOTUS);
        
        OpenCLNoMemPasswordIncrementorLE();
        
        password_count++; 
    }
    vstore_type(b0, get_global_id(0), &deviceGlobalStartPasswordsPlainLOTUS[0]);
    if (PASSWORD_LENGTH > 4) {vstore_type(b1, get_global_id(0), &deviceGlobalStartPasswordsPlainLOTUS[1 * deviceNumberThreadsPlainLOTUS]);}
    if (PASSWORD_LENGTH > 8) {vstore_type(b2, get_global_id(0), &deviceGlobalStartPasswordsPlainLOTUS[2 * deviceNumberThreadsPlainLOTUS]);}
    if (PASSWORD_LENGTH > 12) {vstore_type(b3, get_global_id(0), &deviceGlobalStartPasswordsPlainLOTUS[3 * deviceNumberThreadsPlainLOTUS]);}
}
