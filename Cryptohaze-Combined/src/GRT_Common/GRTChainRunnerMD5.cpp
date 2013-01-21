


#include "GRT_Common/GRTChainRunnerMD5.h"
#include "GRT_Common/GRTTableHeader.h"
#include <stdlib.h>

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

#define MD5F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD5G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define MD5H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5I(x, y, z) ((y) ^ ((x) | (~z)))
#define MD5ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))
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



// Hash output: 16 bytes.
// Hash input block: 64 bytes

GRTChainRunnerMD5::GRTChainRunnerMD5() : GRTChainRunner(16, 64) {

}

void GRTChainRunnerMD5::hashFunction(unsigned char *hashInput, unsigned char *hashOutput) {
    // 32-bit unsigned values for the hash
    uint32_t a, b, c, d;
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;

    int length = this->TableHeader->getPasswordLength();

    // 32-bit accesses to the hash arrays
    uint32_t *InitialArray32;
    uint32_t *OutputArray32;
    InitialArray32 = (uint32_t *) hashInput;
    OutputArray32 = (uint32_t *) hashOutput;



    b0 = (uint32_t) InitialArray32[0];
    b1 = (uint32_t) InitialArray32[1];
    b2 = (uint32_t) InitialArray32[2];
    b3 = (uint32_t) InitialArray32[3];
    b4 = (uint32_t) InitialArray32[4];
    b5 = (uint32_t) InitialArray32[5];
    b6 = (uint32_t) InitialArray32[6];
    b7 = (uint32_t) InitialArray32[7];
    b8 = (uint32_t) InitialArray32[8];
    b9 = (uint32_t) InitialArray32[9];
    b10 = (uint32_t) InitialArray32[10];
    b11 = (uint32_t) InitialArray32[11];
    b12 = (uint32_t) InitialArray32[12];
    b13 = (uint32_t) InitialArray32[13];
    b14 = (uint32_t) InitialArray32[14];

    switch (length) {
        case 0:
            b0 |= 0x00000080;
            break;
        case 1:
            b0 |= 0x00008000;
            break;
        case 2:
            b0 |= 0x00800000;
            break;
        case 3:
            b0 |= 0x80000000;
            break;
        case 4:
            b1 |= 0x00000080;
            break;
        case 5:
            b1 |= 0x00008000;
            break;
        case 6:
            b1 |= 0x00800000;
            break;
        case 7:
            b1 |= 0x80000000;
            break;
        case 8:
            b2 |= 0x00000080;
            break;
        case 9:
            b2 |= 0x00008000;
            break;
        case 10:
            b2 |= 0x00800000;
            break;
        case 11:
            b2 |= 0x80000000;
            break;
        case 12:
            b3 |= 0x00000080;
            break;
        case 13:
            b3 |= 0x00008000;
            break;
        case 14:
            b3 |= 0x00800000;
            break;
        case 15:
            b3 |= 0x80000000;
            break;
        case 16:
            b4 |= 0x00000080;
            break;
        case 17:
            b4 |= 0x00008000;
            break;
        case 18:
            b4 |= 0x00800000;
            break;
        case 19:
            b4 |= 0x80000000;
            break;
        default:
            printf("Length %d not supported!\n", length);
            exit(1);
    }

    b14 = length * 8;
    b15 = 0x00000000;

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

    /* Round 4 */
    MD5II(a, b, c, d, b0, MD5S41, 0xf4292244); /* 49 */
    MD5II(d, a, b, c, b7, MD5S42, 0x432aff97); /* 50 */
    MD5II(c, d, a, b, b14, MD5S43, 0xab9423a7); /* 51 */
    MD5II(b, c, d, a, b5, MD5S44, 0xfc93a039); /* 52 */
    MD5II(a, b, c, d, b12, MD5S41, 0x655b59c3); /* 53 */
    MD5II(d, a, b, c, b3, MD5S42, 0x8f0ccc92); /* 54 */
    MD5II(c, d, a, b, b10, MD5S43, 0xffeff47d); /* 55 */
    MD5II(b, c, d, a, b1, MD5S44, 0x85845dd1); /* 56 */
    MD5II(a, b, c, d, b8, MD5S41, 0x6fa87e4f); /* 57 */
    MD5II(d, a, b, c, b15, MD5S42, 0xfe2ce6e0); /* 58 */
    MD5II(c, d, a, b, b6, MD5S43, 0xa3014314); /* 59 */
    MD5II(b, c, d, a, b13, MD5S44, 0x4e0811a1); /* 60 */
    MD5II(a, b, c, d, b4, MD5S41, 0xf7537e82); /* 61 */
    MD5II(d, a, b, c, b11, MD5S42, 0xbd3af235); /* 62 */
    MD5II(c, d, a, b, b2, MD5S43, 0x2ad7d2bb); /* 63 */
    MD5II(b, c, d, a, b9, MD5S44, 0xeb86d391); /* 64 */

    // Finally, add initial values, as this is the only pass we make.
    a += 0x67452301;
    b += 0xefcdab89;
    c += 0x98badcfe;
    d += 0x10325476;

    OutputArray32[0] = a;
    OutputArray32[1] = b;
    OutputArray32[2] = c;
    OutputArray32[3] = d;
}

void GRTChainRunnerMD5::reduceFunction(unsigned char *password, unsigned char *hash, uint32_t CurrentStep) {
    UINT4 a, b, c, d;

    uint32_t charset_offset = CurrentStep % this->charsetLength;
    uint32_t PasswordLength = this->TableHeader->getPasswordLength();
    uint32_t Device_Table_Index = this->TableHeader->getTableIndex();

    a = (hash[3]*(256*256*256) + hash[2]*(256*256) + hash[1]*256 + hash[0]);
    b = (hash[7]*(256*256*256) + hash[6]*(256*256) + hash[5]*256 + hash[4]);
    c = (hash[11]*(256*256*256) + hash[10]*(256*256) + hash[9]*256 + hash[8]);
    d = (hash[15]*(256*256*256) + hash[14]*(256*256) + hash[13]*256 + hash[12]);

    UINT4 z;
    // Reduce it
    // First 3
    z = (UINT4)(a+CurrentStep+Device_Table_Index) % (256*256*256);
    password[0] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 1) {return;}
    z /= 256;
    password[1] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 2) {return;}
    z /= 256;
    password[2] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 3) {return;}

    // Second 3
    z = (UINT4)(b+CurrentStep+Device_Table_Index) % (256*256*256);
    password[3] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 4) {return;}
    z /= 256;
    password[4] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 5) {return;}
    z /= 256;
    password[5] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 6) {return;}

    z = (UINT4)(c+CurrentStep+Device_Table_Index) % (256*256*256);
    password[6] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 7) {return;}
    z /= 256;
    password[7] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 8) {return;}
    z /= 256;
    password[8] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 9) {return;}

    z = (UINT4)(d+CurrentStep+Device_Table_Index) % (256*256*256);
    password[9] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 10) {return;}
    z /= 256;
    password[10] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 11) {return;}
    z /= 256;
    password[11] = (UINT4)this->charset[(z % 256) + charset_offset];
    if (PasswordLength == 12) {return;}

}
