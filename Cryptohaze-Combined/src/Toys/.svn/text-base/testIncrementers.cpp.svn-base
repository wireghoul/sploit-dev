// Test the incrementors.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <string.h>

#include "../../inc/MFN_CUDA_device/inc.h"


int main(int argc, char *argv[]) {
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11;
    uint32_t passOffset;

    int passLength;
    
    uint32_t numberCombinations = 1;

    std::vector<uint8_t> charset;

    std::vector<uint8_t> forwardCharsetLookup;
    std::vector<uint8_t> reverseCharsetLookup;
    std::vector<uint8_t> charsetLengths;

    int i;

    if (argc != 3) {
        printf("Useage: %s [pass len] [charset]\n", argv[0]);
        exit(1);
    }

    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = 0x00;
    b0 = b1 = 0x2d2d2d2d;

    printf("Got charset length %d\n", strlen(argv[2]));
    charset.resize(strlen(argv[2]));

    for (i = 0; i < strlen(argv[2]); i++) {
        charset[i] = argv[2][i];
    }
    passLength = atoi(argv[1]);

    // Generate forward charset lookup array
    forwardCharsetLookup.resize(charset.size());
    for (i = 0; i < charset.size(); i++) {
        forwardCharsetLookup[i] = charset[i];
    }

    // Generate reverse charset array
    reverseCharsetLookup.resize(128, 0);
    for (i = 0; i < charset.size(); i++) {
        reverseCharsetLookup[charset[i]] = i;
    }

    // Generate length array
    charsetLengths.resize(1);
    charsetLengths[0] = charset.size();

    if (1) {
        printf("Forward charset: \n");
        for (i = 0; i < forwardCharsetLookup.size(); i++) {
            printf("%03d: %c\n", i, forwardCharsetLookup[i]);
        }

        printf("\nReverse charset: \n");
        for (i = 0; i < reverseCharsetLookup.size(); i++) {
            printf("%03d (%02x): %d (%02x)\n", i, i, reverseCharsetLookup[i], reverseCharsetLookup[i]);
        }
    }

    for (i = 0; i < passLength; i++) {
        numberCombinations *= charset.size();
    }

    switch(passLength) {
        case 8:
            b1 &= 0x00ffffff;
            b1 |= ((uint32_t)charset[0] << 24);
        case 7:
            b1 &= 0xff00ffff;
            b1 |= ((uint32_t)charset[0] << 16);
        case 6:
            b1 &= 0xffff00ff;
            b1 |= ((uint32_t)charset[0] << 8);
        case 5:
            b1 &= 0xffffff00;
            b1 |= ((uint32_t)charset[0] << 0);
        case 4:
            b0 &= 0x00ffffff;
            b0 |= ((uint32_t)charset[0] << 24);
        case 3:
            b0 &= 0xff00ffff;
            b0 |= ((uint32_t)charset[0] << 16);
        case 2:
            b0 &= 0xffff00ff;
            b0 |= ((uint32_t)charset[0] << 8);
        case 1:
            b0 &= 0xffffff00;
            b0 |= ((uint32_t)charset[0] << 0);
    }

    printf("b0: %08x %c%c%c%c\n", b0, (b0 & 0xff), ((b0 >> 8) & 0xff), ((b0 >> 16) & 0xff), ((b0 >> 24) & 0xff));
    printf("b1: %08x %c%c%c%c\n", b1, (b1 & 0xff), ((b1 >> 8) & 0xff), ((b1 >> 16) & 0xff), ((b1 >> 24) & 0xff));

    for (i = 0; i < numberCombinations; i++) {
        printf("%d: ", i);
        printf("b0: %08x %c%c%c%c  ", b0, (b0 & 0xff), ((b0 >> 8) & 0xff), ((b0 >> 16) & 0xff), ((b0 >> 24) & 0xff));
        printf("b1: %08x %c%c%c%c\n", b1, (b1 & 0xff), ((b1 >> 8) & 0xff), ((b1 >> 16) & 0xff), ((b1 >> 24) & 0xff));
        switch (passLength) {
            case 1:
                makeMFNSingleIncrementors1(forwardCharsetLookup, reverseCharsetLookup, charsetLengths);
                break;
            case 2:
                makeMFNSingleIncrementors2(forwardCharsetLookup, reverseCharsetLookup, charsetLengths);
                break;
            case 3:
                makeMFNSingleIncrementors3(forwardCharsetLookup, reverseCharsetLookup, charsetLengths);
                break;
            case 4:
                makeMFNSingleIncrementors4(forwardCharsetLookup, reverseCharsetLookup, charsetLengths);
                break;
            case 5:
                makeMFNSingleIncrementors5(forwardCharsetLookup, reverseCharsetLookup, charsetLengths);
                break;
            case 6:
                makeMFNSingleIncrementors6(forwardCharsetLookup, reverseCharsetLookup, charsetLengths);
                break;
            case 7:
                makeMFNSingleIncrementors7(forwardCharsetLookup, reverseCharsetLookup, charsetLengths);
                break;
            case 8:
                makeMFNSingleIncrementors8(forwardCharsetLookup, reverseCharsetLookup, charsetLengths);
                break;
        }
    }
}