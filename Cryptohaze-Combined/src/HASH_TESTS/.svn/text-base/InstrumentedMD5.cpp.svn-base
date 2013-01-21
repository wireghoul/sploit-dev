/**
 * This is an instrumented version of MD5 for better understanding the hash
 * output and what can & cannot be altered to improve performance.  This is
 * public code, fully available to anyone who wants to use it for whatever
 * purposes.  This is not GPLv2 as the rest of the project is, it is fully
 * open.  Please stop using MD5 for password storage.
 *
 * To use, simply run with the only command line argument being the string
 * to hash.  It will only run a single block at a time, as the reversing
 * is only relevant on the last block.
 *
 * -Bitweasil
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

//====== MD5 Defines =====

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

// OPTIMIZED MD5 FUNCTIONS HERE
//#define MD5F(x,y,z) (((y ^ z) & x) ^ z)
//#define MD5G(x,y,z) (((x & y) & z) ^ y)

/* F, G, H and I are basic MD5 functions.
 */
#define MD5F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD5G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define MD5H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5I(x, y, z) ((y) ^ ((x) | (~z)))

/* ROTATE_LEFT rotates x left n bits.
 */
#define MD5ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4.
Rotation is separate from addition to prevent recomputation.
 */
#define MD5FF(a, b, c, d, x, s, ac) { \
 (a) += MD5F ((b), (c), (d)) + (x) + (uint32_t)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5GG(a, b, c, d, x, s, ac) { \
 (a) += MD5G ((b), (c), (d)) + (x) + (uint32_t)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5HH(a, b, c, d, x, s, ac) { \
 (a) += MD5H ((b), (c), (d)) + (x) + (uint32_t)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5II(a, b, c, d, x, s, ac) { \
 (a) += MD5I ((b), (c), (d)) + (x) + (uint32_t)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }

#define MD5ROTATE_RIGHT(x, n) (((x) >> (n)) | ((x) << (32-(n))))

#define REV_II(a,b,c,d,data,shift,constant) \
    a = MD5ROTATE_RIGHT((a - b), shift) - data - constant - (c ^ (b | (~d)));

#define reverse(x) (uint32_t)((x>>24)|((x<<8) & 0x00FF0000)|((x>>8) & 0x0000FF00)|(x<<24))

#define printValues() { \
        printf(" b0- b3: %08x %08x %08x %08x\n", b0, b1, b2, b3); \
        printf(" b4- b7: %08x %08x %08x %08x\n", b4, b5, b6, b7); \
        printf(" b8-b11: %08x %08x %08x %08x\n", b8, b9, b10, b11); \
        printf("b12-b15: %08x %08x %08x %08x\n", b12, b13, b14, b15); \
}

#define printad() { \
        printf("  %08x %08x %08x %08x\n", reverse(a), reverse(b), reverse(c), reverse(d)); \
}

int main(int argc, char *argv[]) {
    std::vector <uint8_t> md5data;
    int i;

    uint32_t *md5data32;

    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
    uint32_t a, b, c, d;

    printf("Instrumented MD5\n");

    if (argc != 2) {
        printf("Usage: %s [md5 string to evaluate]\n", argv[0]);
    }

    // Push the data from the command line into the vector
    for (i = 0; i < strlen(argv[1]); i++) {
        md5data.push_back(argv[1][i]);
    }
    // Add the padding bit
    md5data.push_back(0x80);

    if (md5data.size() > 55) {
        printf("Error: Too much data.  Can only handle 55 chars.\n");
        exit(1);
    }

    // Resize the buffer to the full length, pad with 0s.
    md5data.resize(64, 0);

    // Load the values!
    md5data32 = (uint32_t *) & md5data[0];

    b0 = md5data32[0];
    b1 = md5data32[1];
    b2 = md5data32[2];
    b3 = md5data32[3];
    b4 = md5data32[4];
    b5 = md5data32[5];
    b6 = md5data32[6];
    b7 = md5data32[7];
    b8 = md5data32[8];
    b9 = md5data32[9];
    b10 = md5data32[10];
    b11 = md5data32[11];
    b12 = md5data32[12];
    b13 = md5data32[13];

    // Length values.
    b14 = strlen(argv[1]) * 8;
    b15 = 0x00;

    printValues();

    a = 0x67452301;
    b = 0xefcdab89;
    c = 0x98badcfe;
    d = 0x10325476;

    MD5FF(a, b, c, d, b0, MD5S11, 0xd76aa478); /* 1 */
    MD5FF(d, a, b, c, b1, MD5S12, 0xe8c7b756); /* 2 */
    MD5FF(c, d, a, b, b2, MD5S13, 0x242070db); /* 3 */
    MD5FF(b, c, d, a, b3, MD5S14, 0xc1bdceee); /* 4 */
    MD5FF(a, b, c, d, b4, MD5S11, 0xf57c0faf); /* 5 */
    MD5FF(d, a, b, c, b5, MD5S12, 0x4787c62a); /* 6 */
    MD5FF(c, d, a, b, b6, MD5S13, 0xa8304613); /* 7 */
    MD5FF(b, c, d, a, b7, MD5S14, 0xfd469501); /* 8 */
    MD5FF(a, b, c, d, b8, MD5S11, 0x698098d8); /* 9 */
    MD5FF(d, a, b, c, b9, MD5S12, 0x8b44f7af); /* 10 */
    MD5FF(c, d, a, b, b10, MD5S13, 0xffff5bb1); /* 11 */
    MD5FF(b, c, d, a, b11, MD5S14, 0x895cd7be); /* 12 */
    MD5FF(a, b, c, d, b12, MD5S11, 0x6b901122); /* 13 */
    MD5FF(d, a, b, c, b13, MD5S12, 0xfd987193); /* 14 */
    MD5FF(c, d, a, b, b14, MD5S13, 0xa679438e); /* 15 */
    MD5FF(b, c, d, a, b15, MD5S14, 0x49b40821); /* 16 */

    /* Round 2 */
    MD5GG(a, b, c, d, b1, MD5S21, 0xf61e2562); /* 17 */
    MD5GG(d, a, b, c, b6, MD5S22, 0xc040b340); /* 18 */
    MD5GG(c, d, a, b, b11, MD5S23, 0x265e5a51); /* 19 */
    MD5GG(b, c, d, a, b0, MD5S24, 0xe9b6c7aa); /* 20 */
    MD5GG(a, b, c, d, b5, MD5S21, 0xd62f105d); /* 21 */
    MD5GG(d, a, b, c, b10, MD5S22, 0x2441453); /* 22 */
    MD5GG(c, d, a, b, b15, MD5S23, 0xd8a1e681); /* 23 */
    MD5GG(b, c, d, a, b4, MD5S24, 0xe7d3fbc8); /* 24 */
    MD5GG(a, b, c, d, b9, MD5S21, 0x21e1cde6); /* 25 */
    MD5GG(d, a, b, c, b14, MD5S22, 0xc33707d6); /* 26 */
    MD5GG(c, d, a, b, b3, MD5S23, 0xf4d50d87); /* 27 */
    MD5GG(b, c, d, a, b8, MD5S24, 0x455a14ed); /* 28 */
    MD5GG(a, b, c, d, b13, MD5S21, 0xa9e3e905); /* 29 */
    MD5GG(d, a, b, c, b2, MD5S22, 0xfcefa3f8); /* 30 */
    MD5GG(c, d, a, b, b7, MD5S23, 0x676f02d9); /* 31 */
    MD5GG(b, c, d, a, b12, MD5S24, 0x8d2a4c8a); /* 32 */

    /* Round 3 */
    MD5HH(a, b, c, d, b5, MD5S31, 0xfffa3942); /* 33 */
    MD5HH(d, a, b, c, b8, MD5S32, 0x8771f681); /* 34 */
    MD5HH(c, d, a, b, b11, MD5S33, 0x6d9d6122); /* 35 */
    MD5HH(b, c, d, a, b14, MD5S34, 0xfde5380c); /* 36 */
    MD5HH(a, b, c, d, b1, MD5S31, 0xa4beea44); /* 37 */
    MD5HH(d, a, b, c, b4, MD5S32, 0x4bdecfa9); /* 38 */
    MD5HH(c, d, a, b, b7, MD5S33, 0xf6bb4b60); /* 39 */
    MD5HH(b, c, d, a, b10, MD5S34, 0xbebfbc70); /* 40 */
    MD5HH(a, b, c, d, b13, MD5S31, 0x289b7ec6); /* 41 */
    MD5HH(d, a, b, c, b0, MD5S32, 0xeaa127fa); /* 42 */
    MD5HH(c, d, a, b, b3, MD5S33, 0xd4ef3085); /* 43 */
    MD5HH(b, c, d, a, b6, MD5S34, 0x4881d05); /* 44 */
    MD5HH(a, b, c, d, b9, MD5S31, 0xd9d4d039); /* 45 */
    MD5HH(d, a, b, c, b12, MD5S32, 0xe6db99e5); /* 46 */
    MD5HH(c, d, a, b, b15, MD5S33, 0x1fa27cf8); /* 47 */
    MD5HH(b, c, d, a, b2, MD5S34, 0xc4ac5665); /* 48 */

    printf("0  : "); printad();

    /* Round 4 */
    MD5II(a, b, c, d, b0, MD5S41, 0xf4292244); /* 49 */
    printf("1  : "); printad();
    MD5II(d, a, b, c, b7, MD5S42, 0x432aff97); /* 50 */
    printf("2  : "); printad();
    MD5II(c, d, a, b, b14, MD5S43, 0xab9423a7); /* 51 */
    printf("3  : "); printad();
    MD5II(b, c, d, a, b5, MD5S44, 0xfc93a039); /* 52 */
    printf("4  : "); printad();
    MD5II(a, b, c, d, b12, MD5S41, 0x655b59c3); /* 53 */
    printf("5  : "); printad();
    MD5II(d, a, b, c, b3, MD5S42, 0x8f0ccc92); /* 54 */
    printf("6  : "); printad();
    MD5II(c, d, a, b, b10, MD5S43, 0xffeff47d); /* 55 */
    printf("7  : "); printad();
    MD5II(b, c, d, a, b1, MD5S44, 0x85845dd1); /* 56 */
    printf("8  : "); printad();
    MD5II(a, b, c, d, b8, MD5S41, 0x6fa87e4f); /* 57 */
    printf("9  : "); printad();
    MD5II(d, a, b, c, b15, MD5S42, 0xfe2ce6e0); /* 58 */
    printf("10 : "); printad();
    MD5II(c, d, a, b, b6, MD5S43, 0xa3014314); /* 59 */
    printf("11 : "); printad();
    MD5II(b, c, d, a, b13, MD5S44, 0x4e0811a1); /* 60 */
    printf("12 : "); printad();
    MD5II(a, b, c, d, b4, MD5S41, 0xf7537e82); /* 61 */
    printf("13 : "); printad();
    MD5II(d, a, b, c, b11, MD5S42, 0xbd3af235); /* 62 */
    printf("14 : "); printad();
    MD5II(c, d, a, b, b2, MD5S43, 0x2ad7d2bb); /* 63 */
    printf("15 : "); printad();
    MD5II(b, c, d, a, b9, MD5S44, 0xeb86d391); /* 64 */
    printf("16 : "); printad();


    // Finally, add initial values, as this is the only pass we make.
    a += 0x67452301;
    b += 0xefcdab89;
    c += 0x98badcfe;
    d += 0x10325476;
    printf("Final hash: "); printad();
    
    printf("Reversing...\n");
    
    a -= 0x67452301;
    b -= 0xefcdab89;
    c -= 0x98badcfe;
    d -= 0x10325476;
    
    printf("20 : "); printad();

    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = 0x00;
    
    // Reverse the last few steps
    REV_II (b, c, d, a, b9, MD5S44, 0xeb86d391); //64
    printf("16 : "); printad();
    REV_II (c, d, a, b, b2, MD5S43, 0x2ad7d2bb); //63
    printf("15 : "); printad();
    REV_II (d, a, b, c, b11, MD5S42, 0xbd3af235); //62
    printf("14 : "); printad();
    REV_II (a, b, c, d, b4, MD5S41, 0xf7537e82); //61
    printf("13 : "); printad();
    REV_II (b, c, d, a, b13, MD5S44, 0x4e0811a1); //60
    printf("12 : "); printad();
    REV_II (c, d, a, b, b6, MD5S43, 0xa3014314); //59
    printf("11 : "); printad();
    REV_II (d, a, b, c, b15, MD5S42, 0xfe2ce6e0); //58
    printf("10 : "); printad();
    REV_II (a, b, c, d, b8, MD5S41, 0x6fa87e4f); //57
    printf("9  : "); printad();
    REV_II (b, c, d, a, b1, MD5S44, 0x85845dd1); //56
    printf("8  : "); printad();
    REV_II (c, d, a, b, b10, MD5S43, 0xffeff47d); //55
    printf("7  : "); printad();
    REV_II (d, a, b, c, b3, MD5S42, 0x8f0ccc92); //54
    printf("6  : "); printad();
    REV_II (a, b, c, d, b12, MD5S41, 0x655b59c3); //53
    printf("5  : "); printad();
    REV_II (b, c, d, a, b5, MD5S44, 0xfc93a039); //52
    printf("4  : "); printad();
    REV_II (c, d, a, b, b14, MD5S43, 0xab9423a7); //51
    printf("3  : "); printad();
    REV_II (d, a, b, c, b7, MD5S42, 0x432aff97); //50
    printf("2  : "); printad();



}