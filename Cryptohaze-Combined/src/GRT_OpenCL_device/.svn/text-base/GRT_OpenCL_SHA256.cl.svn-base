
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

void padSHA256Hash(int length,
                vector_type *b0, vector_type *b1, vector_type *b2, vector_type *b3, vector_type *b4, vector_type *b5, vector_type *b6, vector_type *b7,
		vector_type *b8, vector_type *b9, vector_type *b10, vector_type *b11, vector_type *b12, vector_type *b13, vector_type *b14, vector_type *b15);



inline void padSHA256Hash(int length,
                vector_type *b0, vector_type *b1, vector_type *b2, vector_type *b3, vector_type *b4, vector_type *b5, vector_type *b6, vector_type *b7,
		vector_type *b8, vector_type *b9, vector_type *b10, vector_type *b11, vector_type *b12, vector_type *b13, vector_type *b14, vector_type *b15) {

  // Set length properly (length in bits)
  *b15 = (vector_type)(length * 8);

  if (length == 0) {
    *b0 |= (vector_type)0x00000080;
    return;
  }
  if (length == 1) {
    *b0 |= (vector_type)0x00008000;
    return;
  }
  if (length == 2) {
    *b0 |= (vector_type)0x00800000;
    return;
  }
  if (length == 3) {
    *b0 |= (vector_type)0x80000000;
    return;
  }
  if (length == 4) {
    *b1 |= (vector_type)0x00000080;
    return;
  }
  if (length == 5) {
    *b1 |= (vector_type)0x00008000;
    return;
  }
  if (length == 6) {
    *b1 |= (vector_type)0x00800000;
    return;
  }
  if (length == 7) {
    *b1 |= (vector_type)0x80000000;
    return;
  }
  if (length == 8) {
    *b2 |= (vector_type)0x00000080;
    return;
  }
  if (length == 9) {
    *b2 |= (vector_type)0x00008000;
    return;
  }
  if (length == 10) {
    *b2 |= (vector_type)0x00800000;
    return;
  }
  if (length == 11) {
   *b2 |= (vector_type)0x80000000;
    return;
  }
  if (length == 12) {
    *b3 |= (vector_type)0x00000080;
    return;
  }
  if (length == 13) {
    *b3 |= (vector_type)0x00008000;
    return;
  }
  if (length == 14) {
    *b3 |= (vector_type)0x00800000;
    return;
  }
  if (length == 15) {
    *b3 |= (vector_type)0x80000000;
    return;
  }
  if (length == 16) {
    *b4 |= (vector_type)0x00000080;
    return;
  }
  if (length == 17) {
    *b4 |= (vector_type)0x00008000;
    return;
  }
  if (length == 18) {
    *b4 |= (vector_type)0x00800000;
    return;
  }
  if (length == 19) {
    *b4 |= (vector_type)0x80000000;
    return;
  }
  if (length == 20) {
    *b5 |= (vector_type)0x00000080;
    return;
  }
  if (length == 21) {
    *b5 |= (vector_type)0x00008000;
    return;
  }
  if (length == 22) {
    *b5 |= (vector_type)0x00800000;
    return;
  }
  if (length == 23) {
    *b5 |= (vector_type)0x80000000;
    return;
  }
  if (length == 24) {
    *b6 |= (vector_type)0x00000080;
    return;
  }
  if (length == 25) {
    *b6 |= (vector_type)0x00008000;
    return;
  }
  if (length == 26) {
    *b6 |= (vector_type)0x00800000;
    return;
  }
  if (length == 27) {
    *b6 |= (vector_type)0x80000000;
    return;
  }
  if (length == 28) {
    *b7 |= (vector_type)0x00000080;
    return;
  }
  if (length == 29) {
    *b7 |= (vector_type)0x00008000;
    return;
  }
  if (length == 30) {
    *b7 |= (vector_type)0x00800000;
    return;
  }
  if (length == 31) {
    *b7 |= (vector_type)0x80000000;
    return;
  }

}

