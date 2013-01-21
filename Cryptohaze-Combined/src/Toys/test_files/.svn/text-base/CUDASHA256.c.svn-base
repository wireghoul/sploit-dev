#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CUDASHA256.h"

int main() {
    printf("SHA256 test!\n");

    //char msg[] = "abc";
    char msg[] = "00";

    uint32_t W[64];
    int i;
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;

    memset(W, 0, sizeof(uint32_t) * 64);

    b0 = 0x00000000;
    b1 = 0x00000000;
    b2 = 0x00000000;
    b3 = 0x00000000;
    b4 = 0x00000000;
    b5 = 0x00000000;
    b6 = 0x00000000;
    b7 = 0x00000000;
    b8 = 0x00000000;
    b9 = 0x00000000;
    b10 = 0x00000000;
    b11 = 0x00000000;
    b12 = 0x00000000;
    b13 = 0x00000000;
    b14 = 0x00000000;
    b15 = 0x00000000;


    b0 = *(uint32_t *)msg;
    b0 |= 0x00800000;


    b0 = reverse(b0);
    
    b15 = ((2 * 8) & 0xff) << 24 | (((2 * 8) >> 8) & 0xff) << 16;
    b15 = reverse(b15);

    printf("b00: 0x%08x\n", b0);
    printf("b01: 0x%08x\n", b1);
    printf("b02: 0x%08x\n", b2);
    printf("b03: 0x%08x\n", b3);
    printf("b04: 0x%08x\n", b4);
    printf("b05: 0x%08x\n", b5);
    printf("b06: 0x%08x\n", b6);
    printf("b07: 0x%08x\n", b7);
    printf("b08: 0x%08x\n", b8);
    printf("b09: 0x%08x\n", b9);
    printf("b10: 0x%08x\n", b10);
    printf("b11: 0x%08x\n", b11);
    printf("b12: 0x%08x\n", b12);
    printf("b13: 0x%08x\n", b13);
    printf("b14: 0x%08x\n", b14);
    printf("b15: 0x%08x\n", b15);
/*
    printf("Testing expansions...\n");

    W[0] = b0;
    W[1] = b1;
    W[2] = b2;
    W[3] = b3;
    W[4] = b4;
    W[5] = b5;
    W[6] = b6;
    W[7] = b7;
    W[8] = b8;
    W[9] = b9;
    W[10] = b10;
    W[11] = b11;
    W[12] = b12;
    W[13] = b13;
    W[14] = b14;
    W[15] = b15;

    for (i = 16; i < 64; i++) {
        printf("W[i-2]: 0x%08x\n", W[i-2]);
        printf("W[i-7]: 0x%08x\n", W[i-7]);
        printf("W[i-15]: 0x%08x\n", W[i-15]);
        printf("W[i-16]: 0x%08x\n", W[i-16]);
        W[i] = S1(W[i -  2]) + W[i -  7] +
           S0(W[i - 15]) + W[i - 16];
        printf("W[%d] = 0x%08x\n", i, W[i]);
    }

b0 = S1(b14) + b9 + S0(b1) + b0;
printf("Step %d: 0x%08x\n", 16, b0);
printf("W[%d]:   0x%08x\n", 16, W[16]);
b1 = S1(b15) + b10 + S0(b2) + b1;
printf("Step %d: 0x%08x\n", 17, b1);
printf("W[%d]:   0x%08x\n", 17, W[17]);
b2 = S1(b0) + b11 + S0(b3) + b2;
printf("Step %d: 0x%08x\n", 18, b2);
printf("W[%d]:   0x%08x\n", 18, W[18]);
b3 = S1(b1) + b12 + S0(b4) + b3;
printf("Step %d: 0x%08x\n", 19, b3);
printf("W[%d]:   0x%08x\n", 19, W[19]);
b4 = S1(b2) + b13 + S0(b5) + b4;
printf("Step %d: 0x%08x\n", 20, b4);
printf("W[%d]:   0x%08x\n", 20, W[20]);
b5 = S1(b3) + b14 + S0(b6) + b5;
printf("Step %d: 0x%08x\n", 21, b5);
printf("W[%d]:   0x%08x\n", 21, W[21]);
b6 = S1(b4) + b15 + S0(b7) + b6;
printf("Step %d: 0x%08x\n", 22, b6);
printf("W[%d]:   0x%08x\n", 22, W[22]);
b7 = S1(b5) + b0 + S0(b8) + b7;
printf("Step %d: 0x%08x\n", 23, b7);
printf("W[%d]:   0x%08x\n", 23, W[23]);
b8 = S1(b6) + b1 + S0(b9) + b8;
printf("Step %d: 0x%08x\n", 24, b8);
printf("W[%d]:   0x%08x\n", 24, W[24]);
b9 = S1(b7) + b2 + S0(b10) + b9;
printf("Step %d: 0x%08x\n", 25, b9);
printf("W[%d]:   0x%08x\n", 25, W[25]);
b10 = S1(b8) + b3 + S0(b11) + b10;
printf("Step %d: 0x%08x\n", 26, b10);
printf("W[%d]:   0x%08x\n", 26, W[26]);
b11 = S1(b9) + b4 + S0(b12) + b11;
printf("Step %d: 0x%08x\n", 27, b11);
printf("W[%d]:   0x%08x\n", 27, W[27]);
b12 = S1(b10) + b5 + S0(b13) + b12;
printf("Step %d: 0x%08x\n", 28, b12);
printf("W[%d]:   0x%08x\n", 28, W[28]);
b13 = S1(b11) + b6 + S0(b14) + b13;
printf("Step %d: 0x%08x\n", 29, b13);
printf("W[%d]:   0x%08x\n", 29, W[29]);
b14 = S1(b12) + b7 + S0(b15) + b14;
printf("Step %d: 0x%08x\n", 30, b14);
printf("W[%d]:   0x%08x\n", 30, W[30]);
b15 = S1(b13) + b8 + S0(b0) + b15;
printf("Step %d: 0x%08x\n", 31, b15);
printf("W[%d]:   0x%08x\n", 31, W[31]);
b0 = S1(b14) + b9 + S0(b1) + b0;
printf("Step %d: 0x%08x\n", 32, b0);
printf("W[%d]:   0x%08x\n", 32, W[32]);
b1 = S1(b15) + b10 + S0(b2) + b1;
printf("Step %d: 0x%08x\n", 33, b1);
printf("W[%d]:   0x%08x\n", 33, W[33]);
b2 = S1(b0) + b11 + S0(b3) + b2;
printf("Step %d: 0x%08x\n", 34, b2);
printf("W[%d]:   0x%08x\n", 34, W[34]);
b3 = S1(b1) + b12 + S0(b4) + b3;
printf("Step %d: 0x%08x\n", 35, b3);
printf("W[%d]:   0x%08x\n", 35, W[35]);
b4 = S1(b2) + b13 + S0(b5) + b4;
printf("Step %d: 0x%08x\n", 36, b4);
printf("W[%d]:   0x%08x\n", 36, W[36]);
b5 = S1(b3) + b14 + S0(b6) + b5;
printf("Step %d: 0x%08x\n", 37, b5);
printf("W[%d]:   0x%08x\n", 37, W[37]);
b6 = S1(b4) + b15 + S0(b7) + b6;
printf("Step %d: 0x%08x\n", 38, b6);
printf("W[%d]:   0x%08x\n", 38, W[38]);
b7 = S1(b5) + b0 + S0(b8) + b7;
printf("Step %d: 0x%08x\n", 39, b7);
printf("W[%d]:   0x%08x\n", 39, W[39]);
b8 = S1(b6) + b1 + S0(b9) + b8;
printf("Step %d: 0x%08x\n", 40, b8);
printf("W[%d]:   0x%08x\n", 40, W[40]);
b9 = S1(b7) + b2 + S0(b10) + b9;
printf("Step %d: 0x%08x\n", 41, b9);
printf("W[%d]:   0x%08x\n", 41, W[41]);
b10 = S1(b8) + b3 + S0(b11) + b10;
printf("Step %d: 0x%08x\n", 42, b10);
printf("W[%d]:   0x%08x\n", 42, W[42]);
b11 = S1(b9) + b4 + S0(b12) + b11;
printf("Step %d: 0x%08x\n", 43, b11);
printf("W[%d]:   0x%08x\n", 43, W[43]);
b12 = S1(b10) + b5 + S0(b13) + b12;
printf("Step %d: 0x%08x\n", 44, b12);
printf("W[%d]:   0x%08x\n", 44, W[44]);
b13 = S1(b11) + b6 + S0(b14) + b13;
printf("Step %d: 0x%08x\n", 45, b13);
printf("W[%d]:   0x%08x\n", 45, W[45]);
b14 = S1(b12) + b7 + S0(b15) + b14;
printf("Step %d: 0x%08x\n", 46, b14);
printf("W[%d]:   0x%08x\n", 46, W[46]);
b15 = S1(b13) + b8 + S0(b0) + b15;
printf("Step %d: 0x%08x\n", 47, b15);
printf("W[%d]:   0x%08x\n", 47, W[47]);
b0 = S1(b14) + b9 + S0(b1) + b0;
printf("Step %d: 0x%08x\n", 48, b0);
printf("W[%d]:   0x%08x\n", 48, W[48]);
b1 = S1(b15) + b10 + S0(b2) + b1;
printf("Step %d: 0x%08x\n", 49, b1);
printf("W[%d]:   0x%08x\n", 49, W[49]);
b2 = S1(b0) + b11 + S0(b3) + b2;
printf("Step %d: 0x%08x\n", 50, b2);
printf("W[%d]:   0x%08x\n", 50, W[50]);
b3 = S1(b1) + b12 + S0(b4) + b3;
printf("Step %d: 0x%08x\n", 51, b3);
printf("W[%d]:   0x%08x\n", 51, W[51]);
b4 = S1(b2) + b13 + S0(b5) + b4;
printf("Step %d: 0x%08x\n", 52, b4);
printf("W[%d]:   0x%08x\n", 52, W[52]);
b5 = S1(b3) + b14 + S0(b6) + b5;
printf("Step %d: 0x%08x\n", 53, b5);
printf("W[%d]:   0x%08x\n", 53, W[53]);
b6 = S1(b4) + b15 + S0(b7) + b6;
printf("Step %d: 0x%08x\n", 54, b6);
printf("W[%d]:   0x%08x\n", 54, W[54]);
b7 = S1(b5) + b0 + S0(b8) + b7;
printf("Step %d: 0x%08x\n", 55, b7);
printf("W[%d]:   0x%08x\n", 55, W[55]);
b8 = S1(b6) + b1 + S0(b9) + b8;
printf("Step %d: 0x%08x\n", 56, b8);
printf("W[%d]:   0x%08x\n", 56, W[56]);
b9 = S1(b7) + b2 + S0(b10) + b9;
printf("Step %d: 0x%08x\n", 57, b9);
printf("W[%d]:   0x%08x\n", 57, W[57]);
b10 = S1(b8) + b3 + S0(b11) + b10;
printf("Step %d: 0x%08x\n", 58, b10);
printf("W[%d]:   0x%08x\n", 58, W[58]);
b11 = S1(b9) + b4 + S0(b12) + b11;
printf("Step %d: 0x%08x\n", 59, b11);
printf("W[%d]:   0x%08x\n", 59, W[59]);
b12 = S1(b10) + b5 + S0(b13) + b12;
printf("Step %d: 0x%08x\n", 60, b12);
printf("W[%d]:   0x%08x\n", 60, W[60]);
b13 = S1(b11) + b6 + S0(b14) + b13;
printf("Step %d: 0x%08x\n", 61, b13);
printf("W[%d]:   0x%08x\n", 61, W[61]);
b14 = S1(b12) + b7 + S0(b15) + b14;
printf("Step %d: 0x%08x\n", 62, b14);
printf("W[%d]:   0x%08x\n", 62, W[62]);
b15 = S1(b13) + b8 + S0(b0) + b15;
printf("Step %d: 0x%08x\n", 63, b15);
printf("W[%d]:   0x%08x\n", 63, W[63]);
*/

    CUDA_SHA256( b0,  b1,  b2,  b3,  b4,  b5,  b6,  b7,
                  b8,  b9,  b10,  b11,  b12,  b13,  b14,  b15,
                  a,  b,  c,  d,  e,  f,  g,  h);

printf("Final: %08x %08x %08x %08x...\n", a, b, c, d);
}
