


#include "GRT_Common/GRTChainRunnerNTLM.h"
#include "GRT_Common/GRTTableHeader.h"
#include <stdlib.h>

/* MD4 Defines as per RFC reference implementation */
#define MD4F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD4G(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define MD4H(x, y, z) ((x) ^ (y) ^ (z))
#define MD4ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))
#define MD4FF(a, b, c, d, x, s) { \
    (a) += MD4F ((b), (c), (d)) + (x); \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#define MD4GG(a, b, c, d, x, s) { \
    (a) += MD4G ((b), (c), (d)) + (x) + (UINT4)0x5a827999; \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#define MD4HH(a, b, c, d, x, s) { \
    (a) += MD4H ((b), (c), (d)) + (x) + (UINT4)0x6ed9eba1; \
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


// Hash output: 16 bytes.
// Hash input block: 64 bytes

GRTChainRunnerNTLM::GRTChainRunnerNTLM() : GRTChainRunner(16, 64) {

}

void GRTChainRunnerNTLM::hashFunction(unsigned char *hashInput, unsigned char *hashOutput) {
    // 32-bit unsigned values for the hash
    uint32_t a, b, c, d;
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;

    int length = this->TableHeader->getPasswordLength() * 2;

    // 32-bit accesses to the hash arrays
    uint32_t *InitialArray32;
    uint32_t *OutputArray32;
    InitialArray32 = (uint32_t *) hashInput;
    OutputArray32 = (uint32_t *) hashOutput;


    b0 = (uint32_t) 0x00;
    b1 = (uint32_t) 0x00;
    b2 = (uint32_t) 0x00;
    b3 = (uint32_t) 0x00;
    b4 = (uint32_t) 0x00;
    b5 = (uint32_t) 0x00;
    b6 = (uint32_t) 0x00;
    b7 = (uint32_t) 0x00;
    b8 = (uint32_t) 0x00;
    b9 = (uint32_t) 0x00;
    b10 = (uint32_t) 0x00;
    b11 = (uint32_t) 0x00;
    b12 = (uint32_t) 0x00;
    b13 = (uint32_t) 0x00;
    b14 = (uint32_t) 0x00;

    b15 = (UINT4)InitialArray32[0];
    b0 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
    b1 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
    b15 = (UINT4)InitialArray32[1];
    b2 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
    b3 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
    b15 = (UINT4)InitialArray32[2];
    b4 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
    b5 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);

    b15 = (uint32_t) 0x00;

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
        case 20:
            b5 |= 0x00000080;
            break;
        case 21:
            b5 |= 0x00008000;
            break;
        case 22:
            b5 |= 0x00800000;
            break;
        case 23:
            b5 |= 0x80000000;
            break;
        case 24:
            b6 |= 0x00000080;
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

    MD4FF(a, b, c, d, b0, MD4S11); /* 1 */
    MD4FF(d, a, b, c, b1, MD4S12); /* 2 */
    MD4FF(c, d, a, b, b2, MD4S13); /* 3 */
    MD4FF(b, c, d, a, b3, MD4S14); /* 4 */
    MD4FF(a, b, c, d, b4, MD4S11); /* 5 */
    MD4FF(d, a, b, c, b5, MD4S12); /* 6 */
    MD4FF(c, d, a, b, b6, MD4S13); /* 7 */
    MD4FF(b, c, d, a, b7, MD4S14); /* 8 */
    MD4FF(a, b, c, d, b8, MD4S11); /* 9 */
    MD4FF(d, a, b, c, b9, MD4S12); /* 10 */
    MD4FF(c, d, a, b, b10, MD4S13); /* 11 */
    MD4FF(b, c, d, a, b11, MD4S14); /* 12 */
    MD4FF(a, b, c, d, b12, MD4S11); /* 13 */
    MD4FF(d, a, b, c, b13, MD4S12); /* 14 */
    MD4FF(c, d, a, b, b14, MD4S13); /* 15 */
    MD4FF(b, c, d, a, b15, MD4S14); /* 16 */

    /* Round 2 */
    MD4GG(a, b, c, d, b0, MD4S21); /* 17 */
    MD4GG(d, a, b, c, b4, MD4S22); /* 18 */
    MD4GG(c, d, a, b, b8, MD4S23); /* 19 */
    MD4GG(b, c, d, a, b12, MD4S24); /* 20 */
    MD4GG(a, b, c, d, b1, MD4S21); /* 21 */
    MD4GG(d, a, b, c, b5, MD4S22); /* 22 */
    MD4GG(c, d, a, b, b9, MD4S23); /* 23 */
    MD4GG(b, c, d, a, b13, MD4S24); /* 24 */
    MD4GG(a, b, c, d, b2, MD4S21); /* 25 */
    MD4GG(d, a, b, c, b6, MD4S22); /* 26 */
    MD4GG(c, d, a, b, b10, MD4S23); /* 27 */
    MD4GG(b, c, d, a, b14, MD4S24); /* 28 */
    MD4GG(a, b, c, d, b3, MD4S21); /* 29 */
    MD4GG(d, a, b, c, b7, MD4S22); /* 30 */
    MD4GG(c, d, a, b, b11, MD4S23); /* 31 */
    MD4GG(b, c, d, a, b15, MD4S24); /* 32 */


    /* Round 3 */
    MD4HH(a, b, c, d, b0, MD4S31); /* 33 */
    MD4HH(d, a, b, c, b8, MD4S32); /* 34 */
    MD4HH(c, d, a, b, b4, MD4S33); /* 35 */
    MD4HH(b, c, d, a, b12, MD4S34); /* 36 */
    MD4HH(a, b, c, d, b2, MD4S31); /* 37 */
    MD4HH(d, a, b, c, b10, MD4S32); /* 38 */
    MD4HH(c, d, a, b, b6, MD4S33); /* 39 */
    MD4HH(b, c, d, a, b14, MD4S34); /* 40 */
    MD4HH(a, b, c, d, b1, MD4S31); /* 41 */
    MD4HH(d, a, b, c, b9, MD4S32); /* 42 */
    MD4HH(c, d, a, b, b5, MD4S33); /* 43 */
    MD4HH(b, c, d, a, b13, MD4S34); /* 44 */
    MD4HH(a, b, c, d, b3, MD4S31); /* 45 */
    MD4HH(d, a, b, c, b11, MD4S32); /* 46 */
    MD4HH(c, d, a, b, b7, MD4S33); /* 47 */
    MD4HH(b, c, d, a, b15, MD4S34); /* 48 */

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

/*
void GRTChainRunnerNTLM::reduceFunction(unsigned char *password, unsigned char *hash, uint32_t CurrentStep) {
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
    password[0] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 1) {return;}
    z /= 256;
    password[2] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 2) {return;}
    z /= 256;
    password[4] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 3) {return;}

    // Second 3
    z = (UINT4)(b+CurrentStep+Device_Table_Index) % (256*256*256);
    password[6] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 4) {return;}
    z /= 256;
    password[8] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 5) {return;}
    z /= 256;
    password[10] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 6) {return;}

    // Last 2
    z = (UINT4)(c+CurrentStep+Device_Table_Index) % (256*256*256);
    password[12] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 7) {return;}
    z /= 256;
    password[14] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 8) {return;}
    z /= 256;
    password[16] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 9) {return;}

    z = (UINT4)(d+CurrentStep+Device_Table_Index) % (256*256*256);
    password[18] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 10) {return;}
    z /= 256;
    password[20] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 11) {return;}
    z /= 256;
    password[22] = (UINT4)charset[(z % 256) + charset_offset];
    if (PasswordLength == 12) {return;}
    
}
*/

void GRTChainRunnerNTLM::reduceFunction(unsigned char *password, unsigned char *hash, uint32_t CurrentStep) {
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
