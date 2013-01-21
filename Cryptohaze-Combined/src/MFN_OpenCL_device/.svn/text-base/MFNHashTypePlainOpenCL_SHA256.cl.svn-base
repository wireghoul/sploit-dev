
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
    #define restrict
    #include <vector_types.h>
    #define NUMTHREADS 1
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


#define print_hash(num, vector) printf("%d: %08x %08x %08x %08x %08x\n", num, a.vector, b.vector, c.vector, d.vector, e.vector);

#define print_all_hash(num) { \
print_hash(num, s0); \
}

#define reverse(x)(x>>24)|((x<<8) & 0x00FF0000)|((x>>8) & 0x0000FF00)|(x<<24);

#define SHA256_ROTL(val, bits) rotate(val, bits)
#define SHA256_ROTR(val, bits) rotate(val, (32 - bits))

#define SHR(x,n) ((x & 0xFFFFFFFF) >> n)

#define S0(x) (SHA256_ROTR(x, 7) ^ SHA256_ROTR(x,18) ^  SHR(x, 3))
#define S1(x) (SHA256_ROTR(x,17) ^ SHA256_ROTR(x,19) ^  SHR(x,10))

#define S2(x) (SHA256_ROTR(x, 2) ^ SHA256_ROTR(x,13) ^ SHA256_ROTR(x,22))
#define S3(x) (SHA256_ROTR(x, 6) ^ SHA256_ROTR(x,11) ^ SHA256_ROTR(x,25))

//#define F0(x,y,z) ((x & y) | (z & (x | y)))
//#define F1(x,y,z) (z ^ (x & (y ^ z)))
#ifdef BITALIGN
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#define F0(x,y,z) amd_bytealign((z^x), (y), (x))
#define F1(x,y,z) amd_bytealign(x, y, z)
#else
#define F0(x,y,z) bitselect(x,y,(z^x))
#define F1(x,y,z) bitselect(z,y,x)
#endif

#define P(a,b,c,d,e,f,g,h,x,K)                  \
{                                               \
    temp1 = h + S3(e) + F1(e,f,g) + K + x;      \
    temp2 = S2(a) + F0(a,b,c);                  \
    d += temp1; h = temp1 + temp2;              \
}

__constant uint k[64] = {
0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define OPENCL_SHA256_FULL() { \
    vector_type temp1, temp2; \
    a = 0x6A09E667; \
    b = 0xBB67AE85; \
    c = 0x3C6EF372; \
    d = 0xA54FF53A; \
    e = 0x510E527F; \
    f = 0x9B05688C; \
    g = 0x1F83D9AB; \
    h = 0x5BE0CD19; \
    P( a, b, c, d, e, f, g, h,  b0, 0x428A2F98 ); \
    P( h, a, b, c, d, e, f, g,  b1, 0x71374491 ); \
    P( g, h, a, b, c, d, e, f,  b2, 0xB5C0FBCF ); \
    P( f, g, h, a, b, c, d, e,  b3, 0xE9B5DBA5 ); \
    P( e, f, g, h, a, b, c, d,  b4, 0x3956C25B ); \
    P( d, e, f, g, h, a, b, c,  b5, 0x59F111F1 ); \
    P( c, d, e, f, g, h, a, b,  b6, 0x923F82A4 ); \
    P( b, c, d, e, f, g, h, a,  b7, 0xAB1C5ED5 ); \
    P( a, b, c, d, e, f, g, h,  b8, 0xD807AA98 ); \
    P( h, a, b, c, d, e, f, g,  b9, 0x12835B01 ); \
    P( g, h, a, b, c, d, e, f, b10, 0x243185BE ); \
    P( f, g, h, a, b, c, d, e, b11, 0x550C7DC3 ); \
    P( e, f, g, h, a, b, c, d, b12, 0x72BE5D74 ); \
    P( d, e, f, g, h, a, b, c, b13, 0x80DEB1FE ); \
    P( c, d, e, f, g, h, a, b, b14, 0x9BDC06A7 ); \
    P( b, c, d, e, f, g, h, a, b15, 0xC19BF174 ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, 0xE49B69C1 ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, 0xEFBE4786 ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, 0x0FC19DC6 ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, 0x240CA1CC ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, 0x2DE92C6F ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, 0x4A7484AA ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, 0x5CB0A9DC ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, 0x76F988DA ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, 0x983E5152 ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, 0xA831C66D ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, 0xB00327C8 ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, 0xBF597FC7 ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, 0xC6E00BF3 ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, 0xD5A79147 ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, 0x06CA6351 ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, 0x14292967 ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, 0x27B70A85 ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, 0x2E1B2138 ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, 0x4D2C6DFC ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, 0x53380D13 ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, 0x650A7354 ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, 0x766A0ABB ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, 0x81C2C92E ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, 0x92722C85 ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, 0xA2BFE8A1 ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, 0xA81A664B ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, 0xC24B8B70 ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, 0xC76C51A3 ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, 0xD192E819 ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, 0xD6990624 ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, 0xF40E3585 ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, 0x106AA070 ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, 0x19A4C116 ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, 0x1E376C08 ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, 0x2748774C ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, 0x34B0BCB5 ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, 0x391C0CB3 ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, 0x4ED8AA4A ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, 0x5B9CCA4F ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, 0x682E6FF3 ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, 0x748F82EE ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, 0x78A5636F ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, 0x84C87814 ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, 0x8CC70208 ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, 0x90BEFFFA ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, 0xA4506CEB ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, 0xBEF9A3F7 ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, 0xC67178F2 ); \
    a += 0x6A09E667; \
    b += 0xBB67AE85; \
    c += 0x3C6EF372; \
    d += 0xA54FF53A; \
    e += 0x510E527F; \
    f += 0x9B05688C; \
    g += 0x1F83D9AB; \
    h += 0x5BE0CD19; \
}

#define OPENCL_SHA256_FULL_CONSTANTS() { \
    vector_type temp1, temp2; \
    a = 0x6A09E667; \
    b = 0xBB67AE85; \
    c = 0x3C6EF372; \
    d = 0xA54FF53A; \
    e = 0x510E527F; \
    f = 0x9B05688C; \
    g = 0x1F83D9AB; \
    h = 0x5BE0CD19; \
    P( a, b, c, d, e, f, g, h,  b0, (vector_type)k[0] ); \
    P( h, a, b, c, d, e, f, g,  b1, (vector_type)k[1] ); \
    P( g, h, a, b, c, d, e, f,  b2, (vector_type)k[2] ); \
    P( f, g, h, a, b, c, d, e,  b3, (vector_type)k[3] ); \
    P( e, f, g, h, a, b, c, d,  b4, (vector_type)k[4] ); \
    P( d, e, f, g, h, a, b, c,  b5, (vector_type)k[5] ); \
    P( c, d, e, f, g, h, a, b,  b6, (vector_type)k[6] ); \
    P( b, c, d, e, f, g, h, a,  b7, (vector_type)k[7] ); \
    P( a, b, c, d, e, f, g, h,  b8, (vector_type)k[8] ); \
    P( h, a, b, c, d, e, f, g,  b9, (vector_type)k[9] ); \
    P( g, h, a, b, c, d, e, f, b10, (vector_type)k[10] ); \
    P( f, g, h, a, b, c, d, e, b11, (vector_type)k[11] ); \
    P( e, f, g, h, a, b, c, d, b12, (vector_type)k[12] ); \
    P( d, e, f, g, h, a, b, c, b13, (vector_type)k[13] ); \
    P( c, d, e, f, g, h, a, b, b14, (vector_type)k[14] ); \
    P( b, c, d, e, f, g, h, a, b15, (vector_type)k[15] ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, (vector_type)k[16] ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, (vector_type)k[17] ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, (vector_type)k[18] ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, (vector_type)k[19] ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, (vector_type)k[20] ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, (vector_type)k[21] ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, (vector_type)k[22] ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, (vector_type)k[23] ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, (vector_type)k[24] ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, (vector_type)k[25] ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, (vector_type)k[26] ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, (vector_type)k[27] ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, (vector_type)k[28] ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, (vector_type)k[29] ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, (vector_type)k[30] ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, (vector_type)k[31] ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, (vector_type)k[32] ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, (vector_type)k[33] ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, (vector_type)k[34] ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, (vector_type)k[35] ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, (vector_type)k[36] ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, (vector_type)k[37] ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, (vector_type)k[38] ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, (vector_type)k[39] ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, (vector_type)k[40] ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, (vector_type)k[41] ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, (vector_type)k[42] ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, (vector_type)k[43] ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, (vector_type)k[44] ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, (vector_type)k[45] ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, (vector_type)k[46] ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, (vector_type)k[47] ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, (vector_type)k[48] ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, (vector_type)k[49] ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, (vector_type)k[50] ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, (vector_type)k[51] ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, (vector_type)k[52] ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, (vector_type)k[53] ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, (vector_type)k[54] ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, (vector_type)k[55] ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, (vector_type)k[56] ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, (vector_type)k[57] ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, (vector_type)k[58] ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, (vector_type)k[59] ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, (vector_type)k[60] ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, (vector_type)k[61] ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, (vector_type)k[62] ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, (vector_type)k[63] ); \
    a += 0x6A09E667; \
    b += 0xBB67AE85; \
    c += 0x3C6EF372; \
    d += 0xA54FF53A; \
    e += 0x510E527F; \
    f += 0x9B05688C; \
    g += 0x1F83D9AB; \
    h += 0x5BE0CD19; \
}
/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_CHARSET_LENGTH 128

// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
#define CopyFoundPasswordToMemory256(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 16: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b3.s##suffix >> 0) & 0xff; \
        case 15: \
            dfp[search_index * PASSWORD_LENGTH + 14] = (b3.s##suffix >> 8) & 0xff; \
        case 14: \
            dfp[search_index * PASSWORD_LENGTH + 13] = (b3.s##suffix >> 16) & 0xff; \
        case 13: \
            dfp[search_index * PASSWORD_LENGTH + 12] = (b3.s##suffix >> 24) & 0xff; \
        case 12: \
            dfp[search_index * PASSWORD_LENGTH + 11] = (b2.s##suffix >> 0) & 0xff; \
        case 11: \
            dfp[search_index * PASSWORD_LENGTH + 10] = (b2.s##suffix >> 8) & 0xff; \
        case 10: \
            dfp[search_index * PASSWORD_LENGTH + 9] = (b2.s##suffix >> 16) & 0xff; \
        case 9: \
            dfp[search_index * PASSWORD_LENGTH + 8] = (b2.s##suffix >> 24) & 0xff; \
        case 8: \
            dfp[search_index * PASSWORD_LENGTH + 7] = (b1.s##suffix >> 0) & 0xff; \
        case 7: \
            dfp[search_index * PASSWORD_LENGTH + 6] = (b1.s##suffix >> 8) & 0xff; \
        case 6: \
            dfp[search_index * PASSWORD_LENGTH + 5] = (b1.s##suffix >> 16) & 0xff; \
        case 5: \
            dfp[search_index * PASSWORD_LENGTH + 4] = (b1.s##suffix >> 24) & 0xff; \
        case 4: \
            dfp[search_index * PASSWORD_LENGTH + 3] = (b0.s##suffix >> 0) & 0xff; \
        case 3: \
            dfp[search_index * PASSWORD_LENGTH + 2] = (b0.s##suffix >> 8) & 0xff; \
        case 2: \
            dfp[search_index * PASSWORD_LENGTH + 1] = (b0.s##suffix >> 16) & 0xff; \
        case 1: \
            dfp[search_index * PASSWORD_LENGTH + 0] = (b0.s##suffix >> 24) & 0xff; \
    } \
    dfpf[search_index] = (unsigned char) 1; \
}


#define CheckPassword256(dgh, dfp, dfpf, dnh, suffix) { \
    search_high = dnh; \
    search_low = 0; \
    while (search_low < search_high) { \
        search_index = search_low + (search_high - search_low) / 2; \
        current_hash_value = dgh[8 * search_index]; \
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
        while (search_index && (a.s##suffix == dgh[(search_index - 1) * 8])) { \
            search_index--; \
        } \
        while ((a.s##suffix == dgh[search_index * 8])) { \
            if (b.s##suffix == dgh[search_index * 8 + 1]) { \
                if (c.s##suffix == dgh[search_index * 8 + 2]) { \
                    if (d.s##suffix == dgh[search_index * 8 + 3]) { \
                    CopyFoundPasswordToMemory256(dfp, dfpf, suffix); \
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
void MFNHashTypePlainOpenCL_SHA256(
    __constant unsigned char const * restrict deviceCharsetPlainSHA256, /* 0 */
    __constant unsigned char const * restrict deviceReverseCharsetPlainSHA256, /* 1 */
    __constant unsigned char const * restrict charsetLengthsPlainSHA256, /* 2 */
    __constant unsigned char const * restrict constantBitmapAPlainSHA256, /* 3 */
        
    __private unsigned long const numberOfHashesPlainSHA256, /* 4 */
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainSHA256, /* 5 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlainSHA256, /* 6 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainSHA256, /* 7 */
        
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainSHA256, /* 8 */
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainSHA256, /* 9 */
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainSHA256, /* 10 */
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainSHA256, /* 11 */
        
    __global   unsigned char *deviceGlobalStartPointsPlainSHA256, /* 12 */
    __private unsigned long const deviceNumberThreadsPlainSHA256, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainSHA256, /* 14 */
    __global   unsigned int * deviceGlobalStartPasswordsPlainSHA256, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainSHA256 /* 16 */
) {
    // Start the kernel.
    __local unsigned char sharedCharsetPlainSHA256[MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    __local unsigned char sharedReverseCharsetPlainSHA256[MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    __local unsigned char sharedCharsetLengthsPlainSHA256[PASSWORD_LENGTH];
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    //__local unsigned int  plainStore[(((PASSWORD_LENGTH + 1)/4) + 1) * THREADSPERBLOCK * VECTOR_WIDTH];
    
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, e, f, g, h, bitmap_index;
    vector_type b0_t, b1_t, b2_t, b3_t;
    
    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainSHA256[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < (MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_CHARSET_LENGTH * PASSWORD_LENGTH); counter++) {
            sharedCharsetPlainSHA256[counter] = deviceCharsetPlainSHA256[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < (MFN_HASH_TYPE_PLAIN_CUDA_SHA256_MAX_CHARSET_LENGTH * PASSWORD_LENGTH); counter++) {
            sharedReverseCharsetPlainSHA256[counter] = deviceReverseCharsetPlainSHA256[counter];
        }
        for (counter = 0; counter < PASSWORD_LENGTH; counter++) {
            sharedCharsetLengthsPlainSHA256[counter] = charsetLengthsPlainSHA256[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    b15 = (vector_type) (PASSWORD_LENGTH * 8);
    a = b = c = d = e = f = g = h = (vector_type) 0;

    b0 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSHA256[0]);
    if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSHA256[1 * deviceNumberThreadsPlainSHA256]);}
    if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSHA256[2 * deviceNumberThreadsPlainSHA256]);}
    if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSHA256[3 * deviceNumberThreadsPlainSHA256]);}
    
    while (password_count < deviceNumberStepsToRunPlainSHA256) {
        // Store the plains - they get mangled by SHA256 state.
//        vstore_type(b0, get_local_id(0), &plainStore[0]);
//        if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_local_id(0), &plainStore[VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_local_id(0), &plainStore[2 * VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_local_id(0),&plainStore[3 * VECTOR_WIDTH * THREADSPERBLOCK]);}

            
        b0_t = b0;
        if (PASSWORD_LENGTH > 3) {b1_t = b1;} 
        if (PASSWORD_LENGTH > 7) {b2_t = b2;}
        if (PASSWORD_LENGTH > 11) {b3_t = b3;}        
        
        b15 = (vector_type) (PASSWORD_LENGTH * 8);
        OPENCL_SHA256_FULL_CONSTANTS();
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;

        b0 = b0_t;
        if (PASSWORD_LENGTH > 3) {b1 = b1_t;} 
        if (PASSWORD_LENGTH > 7) {b2 = b2_t;}
        if (PASSWORD_LENGTH > 11) {b3 = b3_t;}        

//        b0 = vload_type(get_local_id(0), &plainStore[0]);
//        if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_local_id(0), &plainStore[VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_local_id(0), &plainStore[2 * VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_local_id(0),&plainStore[3 * VECTOR_WIDTH * THREADSPERBLOCK]);}
//        
//        printf(".s0 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s0 >> 24) & 0xff, (b0.s0 >> 16) & 0xff,
//                (b0.s0 >> 8) & 0xff, (b0.s0 >> 0) & 0xff,
//                (b1.s0 >> 24) & 0xff,
//                a.s0, b.s0, c.s0, d.s0, e.s0);
//        printf(".s1 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s1 >> 24) & 0xff, (b0.s1 >> 16) & 0xff,
//                (b0.s1 >> 8) & 0xff, (b0.s1 >> 0) & 0xff,
//                (b1.s1 >> 24) & 0xff,
//                a.s1, b.s1, c.s1, d.s1, e.s1);
//        printf(".s2 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s2 >> 24) & 0xff, (b0.s2 >> 16) & 0xff,
//                (b0.s2 >> 8) & 0xff, (b0.s2 >> 0) & 0xff,
//                (b1.s2 >> 24) & 0xff,
//                a.s2, b.s2, c.s2, d.s2, e.s2);
//        printf(".s3 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s3 >> 24) & 0xff, (b0.s3 >> 16) & 0xff,
//                (b0.s3 >> 8) & 0xff, (b0.s3 >> 0) & 0xff,
//                (b1.s3 >> 24) & 0xff,
//                a.s3, b.s3, c.s3, d.s3, e.s3);
//        

        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainSHA256, 
            deviceGlobalBitmapBPlainSHA256, deviceGlobalBitmapCPlainSHA256, 
            deviceGlobalBitmapDPlainSHA256, deviceGlobalHashlistAddressPlainSHA256, 
            deviceGlobalFoundPasswordsPlainSHA256, deviceGlobalFoundPasswordFlagsPlainSHA256,
            numberOfHashesPlainSHA256, deviceGlobal256kbBitmapAPlainSHA256);

        OpenCLPasswordIncrementorBE(sharedCharsetPlainSHA256, sharedReverseCharsetPlainSHA256, sharedCharsetLengthsPlainSHA256);

        password_count++; 
    }
    vstore_type(b0, get_global_id(0), &deviceGlobalStartPasswordsPlainSHA256[0]);
    if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_global_id(0), &deviceGlobalStartPasswordsPlainSHA256[1 * deviceNumberThreadsPlainSHA256]);}
    if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_global_id(0), &deviceGlobalStartPasswordsPlainSHA256[2 * deviceNumberThreadsPlainSHA256]);}
    if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_global_id(0), &deviceGlobalStartPasswordsPlainSHA256[3 * deviceNumberThreadsPlainSHA256]);}
  
}
