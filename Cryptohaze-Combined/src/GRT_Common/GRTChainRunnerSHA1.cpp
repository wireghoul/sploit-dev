


#include "GRT_Common/GRTChainRunnerSHA1.h"
#include "GRT_Common/GRTTableHeader.h"
#include <stdlib.h>

#include "CUDA_Common/CUDA_SHA1.h"

// Hash output: 16 bytes.
// Hash input block: 64 bytes

GRTChainRunnerSHA1::GRTChainRunnerSHA1() : GRTChainRunner(20, 64) {

}

void GRTChainRunnerSHA1::hashFunction(unsigned char *hashInput, unsigned char *hashOutput) {
    // 32-bit unsigned values for the hash
    uint32_t a, b, c, d, e;
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

    b15 = ((length * 8) & 0xff) << 24 | (((length * 8) >> 8) & 0xff) << 16;
/*
    printf("b0 : %08x\n", b0);
    printf("b1 : %08x\n", b1);
    printf("b2 : %08x\n", b2);
    printf("b3 : %08x\n", b3);
    printf("b4 : %08x\n", b4);
    printf("b5 : %08x\n", b5);
    printf("b6 : %08x\n", b6);
    printf("b7 : %08x\n", b7);
    printf("b8 : %08x\n", b8);
    printf("b9 : %08x\n", b9);
    printf("b10: %08x\n", b10);
    printf("b11: %08x\n", b11);
    printf("b12: %08x\n", b12);
    printf("b13: %08x\n", b13);
    printf("b14: %08x\n", b14);
    printf("b15: %08x\n", b15);
*/

    SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \

    a = reverse(a);
    b = reverse(b);
    c = reverse(c);
    d = reverse(d);
    e = reverse(e);

    OutputArray32[0] = a;
    OutputArray32[1] = b;
    OutputArray32[2] = c;
    OutputArray32[3] = d;
    OutputArray32[4] = e;
}

void GRTChainRunnerSHA1::reduceFunction(unsigned char *password, unsigned char *hash, uint32_t CurrentStep) {
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
