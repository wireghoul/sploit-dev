// Test out vector style incrementing.



#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


inline void printPassword(uint32_t passChunk) {
    printf("%c%c%c%c\n",
    (passChunk >> 0) & 0xff, (passChunk >> 8) & 0xff,
    (passChunk >> 16) & 0xff, (passChunk >> 24) & 0xff);
}

int main() {
    uint32_t b0[4];
    uint32_t b1[4];
    
    // Set the charset
    uint32_t forwardCharset[128];
    uint32_t reverseCharset[128];
    uint8_t charsetLength[16];
    
    int i, password;
    
    memset(b0, 0, sizeof(b0));
    memset(b1, 0, sizeof(b1));
    memset(forwardCharset, 0, sizeof(forwardCharset));
    memset(reverseCharset, 0, sizeof(reverseCharset));
    memset(charsetLength, 0, sizeof(charsetLength));
    
    forwardCharset[0] = '0';
    forwardCharset[1] = '1';
    forwardCharset[2] = '2';
    forwardCharset[3] = '3';
    forwardCharset[4] = '4';
    forwardCharset[5] = '5';
    forwardCharset[6] = '6';
    forwardCharset[7] = '7';
    forwardCharset[8] = '8';
    forwardCharset[9] = '9';
    
    charsetLength[0] = 10;
    
    for (i = 0; i < charsetLength[0]; i++) {
        reverseCharset[forwardCharset[i]] = i;
    }
    
    for (i = 0; i < charsetLength[0]; i++) {
        printf("Forward charset [%d]: %c\n", i, forwardCharset[i]);
    }
    for (i = 0; i < 128; i++) {
        if (reverseCharset[i] > 0) {
            printf("Reverse charset [%d] (%c): %d\n", i, (char)i, reverseCharset[i]);
        }
    }
    
    b0[0] = ((uint32_t)'0' << 0) | ((uint32_t)'0' << 8) | ((uint32_t)'0' << 16) | ((uint32_t)'0' << 24);
    b0[1] = ((uint32_t)'1' << 0) | ((uint32_t)'2' << 8) | ((uint32_t)'3' << 16) | ((uint32_t)'4' << 24);
    b0[2] = ((uint32_t)'4' << 0) | ((uint32_t)'2' << 8) | ((uint32_t)'7' << 16) | ((uint32_t)'8' << 24);
    b0[3] = ((uint32_t)'9' << 0) | ((uint32_t)'9' << 8) | ((uint32_t)'1' << 16) | ((uint32_t)'1' << 24);
    
    
    printf("Initial .s0: "); printPassword(b0[0]);
    printf("Initial .s1: "); printPassword(b0[1]);
    printf("Initial .s2: "); printPassword(b0[2]);
    printf("Initial .s3: "); printPassword(b0[3]);
    
    uint32_t lookupIndex;
    uint32_t passwordOffsetVector;
    uint32_t newPasswordCharacters;
    
    for (password = 0; password < 1000; password++) {
    
    lookupIndex = (b0[0] >> 0) & 0xff;

    passwordOffsetVector = reverseCharset[lookupIndex];
                    
    b0[0] &= 0xffffff00;
                    
    passwordOffsetVector += 1;

    passwordOffsetVector = (passwordOffsetVector >= charsetLength[0]) ? 0 : passwordOffsetVector;
                    
    newPasswordCharacters = (unsigned int)forwardCharset[passwordOffsetVector];
                    
    b0[0] |= newPasswordCharacters;
                    
    if (!passwordOffsetVector) {
        uint32_t maskVector;
        uint32_t enableMask;

        maskVector = (!passwordOffsetVector) ? 0xffff00ff : 0xffffffff; 

        enableMask = (!passwordOffsetVector) ? 0x01 : 0x00;

        lookupIndex = (b0[0] >> 8) & 0xff;

        passwordOffsetVector = reverseCharset[lookupIndex];

        passwordOffsetVector++;

        passwordOffsetVector = (passwordOffsetVector >= charsetLength[0]) ? 0 : passwordOffsetVector;

        newPasswordCharacters = (unsigned int)forwardCharset[passwordOffsetVector];
        newPasswordCharacters = newPasswordCharacters << 8;

        b0[0] &= maskVector;

        b0[0] |= (~maskVector & newPasswordCharacters);

        enableMask = (enableMask & !passwordOffsetVector) ? 1 : 0;
                        
        if (enableMask) {
                            
            maskVector = (!passwordOffsetVector) ? 0xff00ffff : 0xffffffff; 

            enableMask = (!passwordOffsetVector) ? 0x01 : 0x00;

            lookupIndex = (b0[0] >> 16) & 0xff;

            passwordOffsetVector = reverseCharset[lookupIndex];

            passwordOffsetVector++;

            passwordOffsetVector = (passwordOffsetVector >= charsetLength[0]) ? 0 : passwordOffsetVector;

            newPasswordCharacters = (unsigned int)forwardCharset[passwordOffsetVector];
            newPasswordCharacters = newPasswordCharacters << 16;

            b0[0] &= maskVector;

            b0[0] |= (~maskVector & newPasswordCharacters);

            enableMask = (enableMask & !passwordOffsetVector) ? 1 : 0;

            if (enableMask) {
                maskVector = (!passwordOffsetVector) ? 0x00ffffff : 0xffffffff; 

                enableMask = (!passwordOffsetVector) ? 0x01 : 0x00;

                lookupIndex = (b0[0] >> 24) & 0xff;

                passwordOffsetVector = reverseCharset[lookupIndex];

                passwordOffsetVector++;

                passwordOffsetVector = (passwordOffsetVector >= charsetLength[0]) ? 0 : passwordOffsetVector;

                newPasswordCharacters = (unsigned int)forwardCharset[passwordOffsetVector];
                newPasswordCharacters = newPasswordCharacters << 24;

                b0[0] &= maskVector;

                b0[0] |= (~maskVector & newPasswordCharacters);

                enableMask = (enableMask & !passwordOffsetVector) ? 1 : 0;

                if (enableMask) {

                    maskVector = (!passwordOffsetVector) ? 0xffffff00 : 0xffffffff; 

                    enableMask = (!passwordOffsetVector) ? 0x01 : 0x00;

                    lookupIndex = (b0[0] >> 0) & 0xff;

                    passwordOffsetVector = reverseCharset[lookupIndex];

                    passwordOffsetVector++;

                    passwordOffsetVector = (passwordOffsetVector >= charsetLength[0]) ? 0 : passwordOffsetVector;

                    newPasswordCharacters = (unsigned int)forwardCharset[passwordOffsetVector];

                    b0[0] &= maskVector;

                    b0[0] |= (~maskVector & newPasswordCharacters);
                }
            }
        }
    }
    printPassword(b0[0]);
    }
    
    /*
    
    uint32_t lookupIndex[4];
    uint32_t passwordOffsetVector[4];
    uint32_t newPasswordCharacters[4];
    uint32_t maskVector[4];
    uint32_t enableMask[4];
    
    for (password = 0; password < 10; password++) {
        printf("======== PASSWORD PASS %d=======\n", password);

    // Vector operation on GPU
    lookupIndex[0] = (b0[0] >> 0) & 0xff;
    lookupIndex[1] = (b0[1] >> 0) & 0xff;
    lookupIndex[2] = (b0[2] >> 0) & 0xff;
    lookupIndex[3] = (b0[3] >> 0) & 0xff;
    
    // Element-wise lookups
    passwordOffsetVector[0] = reverseCharset[lookupIndex[0]];
    passwordOffsetVector[1] = reverseCharset[lookupIndex[1]];
    passwordOffsetVector[2] = reverseCharset[lookupIndex[2]];
    passwordOffsetVector[3] = reverseCharset[lookupIndex[3]];
    
    // Vector operation
    b0[0] &= 0xffffff00;
    b0[1] &= 0xffffff00;
    b0[2] &= 0xffffff00;
    b0[3] &= 0xffffff00;
    
    // Vector operation
    passwordOffsetVector[0]++;
    passwordOffsetVector[1]++;
    passwordOffsetVector[2]++;
    passwordOffsetVector[3]++;
    printf("passwordOffsetVector %d %d %d %d\n", passwordOffsetVector[0], 
            passwordOffsetVector[1], passwordOffsetVector[2], passwordOffsetVector[3]);


    // Element-wise operations, I think - could be vector?
    passwordOffsetVector[0] = (passwordOffsetVector[0] >= charsetLength[0]) ? 0 : passwordOffsetVector[0];
    passwordOffsetVector[1] = (passwordOffsetVector[1] >= charsetLength[0]) ? 0 : passwordOffsetVector[1];
    passwordOffsetVector[2] = (passwordOffsetVector[2] >= charsetLength[0]) ? 0 : passwordOffsetVector[2];
    passwordOffsetVector[3] = (passwordOffsetVector[3] >= charsetLength[0]) ? 0 : passwordOffsetVector[3];
    // Element-wise
    newPasswordCharacters[0] = (unsigned int)forwardCharset[passwordOffsetVector[0]];
    newPasswordCharacters[1] = (unsigned int)forwardCharset[passwordOffsetVector[1]];
    newPasswordCharacters[2] = (unsigned int)forwardCharset[passwordOffsetVector[2]];
    newPasswordCharacters[3] = (unsigned int)forwardCharset[passwordOffsetVector[3]];
    
    b0[0] |= newPasswordCharacters[0];
    b0[1] |= newPasswordCharacters[1];
    b0[2] |= newPasswordCharacters[2];
    b0[3] |= newPasswordCharacters[3];
    
    printf("1 Initial .s0: "); printPassword(b0[0]);
    printf("1 Initial .s1: "); printPassword(b0[1]);
    printf("1 Initial .s2: "); printPassword(b0[2]);
    printf("1 Initial .s3: "); printPassword(b0[3]);
    

    //====================
    
    maskVector[0] = (!passwordOffsetVector[0]) ? 0xffff00ff : 0xffffffff; 
    maskVector[1] = (!passwordOffsetVector[1]) ? 0xffff00ff : 0xffffffff; 
    maskVector[2] = (!passwordOffsetVector[2]) ? 0xffff00ff : 0xffffffff; 
    maskVector[3] = (!passwordOffsetVector[3]) ? 0xffff00ff : 0xffffffff; 
    
    enableMask[0] = (!passwordOffsetVector[0]) ? 1 : 0;
    enableMask[1] = (!passwordOffsetVector[1]) ? 1 : 0;
    enableMask[2] = (!passwordOffsetVector[2]) ? 1 : 0;
    enableMask[3] = (!passwordOffsetVector[3]) ? 1 : 0;
    
    printf("maskVector Values: %08x %08x %08x %08x\n", maskVector[0], maskVector[1], maskVector[2], maskVector[3]);
    printf("enableMask Values: %08x %08x %08x %08x\n", enableMask[0], enableMask[1], enableMask[2], enableMask[3]);
    
    
    lookupIndex[0] = (b0[0] >> 8) & 0xff;
    lookupIndex[1] = (b0[1] >> 8) & 0xff;
    lookupIndex[2] = (b0[2] >> 8) & 0xff;
    lookupIndex[3] = (b0[3] >> 8) & 0xff;

    passwordOffsetVector[0] = (enableMask[0]) ? reverseCharset[lookupIndex[0]] : 0;
    passwordOffsetVector[1] = (enableMask[1]) ? reverseCharset[lookupIndex[1]] : 0;
    passwordOffsetVector[2] = (enableMask[2]) ? reverseCharset[lookupIndex[2]] : 0;
    passwordOffsetVector[3] = (enableMask[3]) ? reverseCharset[lookupIndex[3]] : 0;

    passwordOffsetVector[0]++;
    passwordOffsetVector[1]++;
    passwordOffsetVector[2]++;
    passwordOffsetVector[3]++;
    printf("passwordOffsetVector %d %d %d %d\n", passwordOffsetVector[0], 
            passwordOffsetVector[1], passwordOffsetVector[2], passwordOffsetVector[3]);
            
    passwordOffsetVector[0] = (passwordOffsetVector[0] >= charsetLength[0]) ? 0 : passwordOffsetVector[0];
    passwordOffsetVector[1] = (passwordOffsetVector[1] >= charsetLength[0]) ? 0 : passwordOffsetVector[1];
    passwordOffsetVector[2] = (passwordOffsetVector[2] >= charsetLength[0]) ? 0 : passwordOffsetVector[2];
    passwordOffsetVector[3] = (passwordOffsetVector[3] >= charsetLength[0]) ? 0 : passwordOffsetVector[3];

    newPasswordCharacters[0] = (unsigned int)forwardCharset[passwordOffsetVector[0]] << 8;
    newPasswordCharacters[1] = (unsigned int)forwardCharset[passwordOffsetVector[1]] << 8;
    newPasswordCharacters[2] = (unsigned int)forwardCharset[passwordOffsetVector[2]] << 8;
    newPasswordCharacters[3] = (unsigned int)forwardCharset[passwordOffsetVector[3]] << 8;

    b0[0] &= maskVector[0];
    b0[1] &= maskVector[1];
    b0[2] &= maskVector[2];
    b0[3] &= maskVector[3];
    
    b0[0] |= (~maskVector[0] & newPasswordCharacters[0]);
    b0[1] |= (~maskVector[1] & newPasswordCharacters[1]);
    b0[2] |= (~maskVector[2] & newPasswordCharacters[2]);
    b0[3] |= (~maskVector[3] & newPasswordCharacters[3]);

    printf("2 Initial .s0: "); printPassword(b0[0]);
    printf("2 Initial .s1: "); printPassword(b0[1]);
    printf("2 Initial .s2: "); printPassword(b0[2]);
    printf("2 Initial .s3: "); printPassword(b0[3]);

    enableMask[0] = enableMask[0] & !passwordOffsetVector[0];
    enableMask[1] = enableMask[1] & !passwordOffsetVector[1];
    enableMask[2] = enableMask[2] & !passwordOffsetVector[2];
    enableMask[3] = enableMask[3] & !passwordOffsetVector[3];
    
    enableMask[0] = (enableMask[0]) ? 1 : 0;
    enableMask[1] = (enableMask[1]) ? 1 : 0;
    enableMask[2] = (enableMask[2]) ? 1 : 0;
    enableMask[3] = (enableMask[3]) ? 1 : 0;
    printf("enableMask Values: %08x %08x %08x %08x\n", enableMask[0], enableMask[1], enableMask[2], enableMask[3]);

    //===========================
    
    maskVector[0] = (enableMask[0]) ? 0xff00ffff : 0xffffffff; 
    maskVector[1] = (enableMask[1]) ? 0xff00ffff : 0xffffffff; 
    maskVector[2] = (enableMask[2]) ? 0xff00ffff : 0xffffffff; 
    maskVector[3] = (enableMask[3]) ? 0xff00ffff : 0xffffffff; 
    
    printf("enableMask Values: %08x %08x %08x %08x\n", enableMask[0], enableMask[1], enableMask[2], enableMask[3]);
    
    lookupIndex[0] = (b0[0] >> 16) & 0xff;
    lookupIndex[1] = (b0[1] >> 16) & 0xff;
    lookupIndex[2] = (b0[2] >> 16) & 0xff;
    lookupIndex[3] = (b0[3] >> 16) & 0xff;

    passwordOffsetVector[0] = (enableMask[0]) ? reverseCharset[lookupIndex[0]] : 0;
    passwordOffsetVector[1] = (enableMask[1]) ? reverseCharset[lookupIndex[1]] : 0;
    passwordOffsetVector[2] = (enableMask[2]) ? reverseCharset[lookupIndex[2]] : 0;
    passwordOffsetVector[3] = (enableMask[3]) ? reverseCharset[lookupIndex[3]] : 0;

    passwordOffsetVector[0]++;
    passwordOffsetVector[1]++;
    passwordOffsetVector[2]++;
    passwordOffsetVector[3]++;
    printf("passwordOffsetVector %d %d %d %d\n", passwordOffsetVector[0], 
            passwordOffsetVector[1], passwordOffsetVector[2], passwordOffsetVector[3]);
            
    passwordOffsetVector[0] = (passwordOffsetVector[0] >= charsetLength[0]) ? 0 : passwordOffsetVector[0];
    passwordOffsetVector[1] = (passwordOffsetVector[1] >= charsetLength[0]) ? 0 : passwordOffsetVector[1];
    passwordOffsetVector[2] = (passwordOffsetVector[2] >= charsetLength[0]) ? 0 : passwordOffsetVector[2];
    passwordOffsetVector[3] = (passwordOffsetVector[3] >= charsetLength[0]) ? 0 : passwordOffsetVector[3];

    newPasswordCharacters[0] = (unsigned int)forwardCharset[passwordOffsetVector[0]] << 16;
    newPasswordCharacters[1] = (unsigned int)forwardCharset[passwordOffsetVector[1]] << 16;
    newPasswordCharacters[2] = (unsigned int)forwardCharset[passwordOffsetVector[2]] << 16;
    newPasswordCharacters[3] = (unsigned int)forwardCharset[passwordOffsetVector[3]] << 16;

    b0[0] &= maskVector[0];
    b0[1] &= maskVector[1];
    b0[2] &= maskVector[2];
    b0[3] &= maskVector[3];
    
    b0[0] |= (~maskVector[0] & newPasswordCharacters[0]);
    b0[1] |= (~maskVector[1] & newPasswordCharacters[1]);
    b0[2] |= (~maskVector[2] & newPasswordCharacters[2]);
    b0[3] |= (~maskVector[3] & newPasswordCharacters[3]);

    printf("3 Initial .s0: "); printPassword(b0[0]);
    printf("3 Initial .s1: "); printPassword(b0[1]);
    printf("3 Initial .s2: "); printPassword(b0[2]);
    printf("3 Initial .s3: "); printPassword(b0[3]);

    enableMask[0] = enableMask[0] & !passwordOffsetVector[0];
    enableMask[1] = enableMask[1] & !passwordOffsetVector[1];
    enableMask[2] = enableMask[2] & !passwordOffsetVector[2];
    enableMask[3] = enableMask[3] & !passwordOffsetVector[3];
    
    enableMask[0] = (enableMask[0]) ? 1 : 0;
    enableMask[1] = (enableMask[1]) ? 1 : 0;
    enableMask[2] = (enableMask[2]) ? 1 : 0;
    enableMask[3] = (enableMask[3]) ? 1 : 0;
    printf("enableMask Values: %08x %08x %08x %08x\n", enableMask[0], enableMask[1], enableMask[2], enableMask[3]);
    
    //===========================
    
    maskVector[0] = (enableMask[0]) ? 0x00ffffff : 0xffffffff; 
    maskVector[1] = (enableMask[1]) ? 0x00ffffff : 0xffffffff; 
    maskVector[2] = (enableMask[2]) ? 0x00ffffff : 0xffffffff; 
    maskVector[3] = (enableMask[3]) ? 0x00ffffff : 0xffffffff; 
    
    printf("enableMask Values: %08x %08x %08x %08x\n", enableMask[0], enableMask[1], enableMask[2], enableMask[3]);
    
    lookupIndex[0] = (b0[0] >> 24) & 0xff;
    lookupIndex[1] = (b0[1] >> 24) & 0xff;
    lookupIndex[2] = (b0[2] >> 24) & 0xff;
    lookupIndex[3] = (b0[3] >> 24) & 0xff;

    passwordOffsetVector[0] = (enableMask[0]) ? reverseCharset[lookupIndex[0]] : 0;
    passwordOffsetVector[1] = (enableMask[1]) ? reverseCharset[lookupIndex[1]] : 0;
    passwordOffsetVector[2] = (enableMask[2]) ? reverseCharset[lookupIndex[2]] : 0;
    passwordOffsetVector[3] = (enableMask[3]) ? reverseCharset[lookupIndex[3]] : 0;

    passwordOffsetVector[0]++;
    passwordOffsetVector[1]++;
    passwordOffsetVector[2]++;
    passwordOffsetVector[3]++;
    printf("passwordOffsetVector %d %d %d %d\n", passwordOffsetVector[0], 
            passwordOffsetVector[1], passwordOffsetVector[2], passwordOffsetVector[3]);
            
    passwordOffsetVector[0] = (passwordOffsetVector[0] >= charsetLength[0]) ? 0 : passwordOffsetVector[0];
    passwordOffsetVector[1] = (passwordOffsetVector[1] >= charsetLength[0]) ? 0 : passwordOffsetVector[1];
    passwordOffsetVector[2] = (passwordOffsetVector[2] >= charsetLength[0]) ? 0 : passwordOffsetVector[2];
    passwordOffsetVector[3] = (passwordOffsetVector[3] >= charsetLength[0]) ? 0 : passwordOffsetVector[3];

    newPasswordCharacters[0] = (unsigned int)forwardCharset[passwordOffsetVector[0]] << 24;
    newPasswordCharacters[1] = (unsigned int)forwardCharset[passwordOffsetVector[1]] << 24;
    newPasswordCharacters[2] = (unsigned int)forwardCharset[passwordOffsetVector[2]] << 24;
    newPasswordCharacters[3] = (unsigned int)forwardCharset[passwordOffsetVector[3]] << 24;

    b0[0] &= maskVector[0];
    b0[1] &= maskVector[1];
    b0[2] &= maskVector[2];
    b0[3] &= maskVector[3];
    
    b0[0] |= (~maskVector[0] & newPasswordCharacters[0]);
    b0[1] |= (~maskVector[1] & newPasswordCharacters[1]);
    b0[2] |= (~maskVector[2] & newPasswordCharacters[2]);
    b0[3] |= (~maskVector[3] & newPasswordCharacters[3]);

    printf("3 Initial .s0: "); printPassword(b0[0]);
    printf("3 Initial .s1: "); printPassword(b0[1]);
    printf("3 Initial .s2: "); printPassword(b0[2]);
    printf("3 Initial .s3: "); printPassword(b0[3]);

    enableMask[0] = enableMask[0] & !passwordOffsetVector[0];
    enableMask[1] = enableMask[1] & !passwordOffsetVector[1];
    enableMask[2] = enableMask[2] & !passwordOffsetVector[2];
    enableMask[3] = enableMask[3] & !passwordOffsetVector[3];
    
    enableMask[0] = (enableMask[0]) ? 1 : 0;
    enableMask[1] = (enableMask[1]) ? 1 : 0;
    enableMask[2] = (enableMask[2]) ? 1 : 0;
    enableMask[3] = (enableMask[3]) ? 1 : 0;
    printf("enableMask Values: %08x %08x %08x %08x\n", enableMask[0], enableMask[1], enableMask[2], enableMask[3]);    
    }
    */
    //
//    if ((!passwordOffsetVector[0]) || (!passwordOffsetVector[1]) || 
//            (!passwordOffsetVector[2]) ||(!passwordOffsetVector[3])) {
//        // Next character position, only for vectors with a zero component.
//        printf("In position 2 increment...\n");
//        for (i = 0; i < 4; i++) {
//            if (!passwordOffsetVector[i]) {
//                printf("Position %d working\n", i);
//            }
//        }
//
//        (!passwordOffsetVector[0]) ? enableMask[0] = 1 : enableMask[0] = 0;
//        (!passwordOffsetVector[1]) ? enableMask[1] = 1 : enableMask[1] = 0;
//        (!passwordOffsetVector[2]) ? enableMask[2] = 1 : enableMask[2] = 0;
//        (!passwordOffsetVector[3]) ? enableMask[3] = 1 : enableMask[3] = 0;
//        
//        printf("enableMask Values: %d %d %d %d\n", enableMask[0], enableMask[1], enableMask[2], enableMask[3]);
//
//        // Vector operation
//        lookupIndex[0] = (b0[0] >> 8) & 0xff;
//        lookupIndex[1] = (b0[1] >> 8) & 0xff;
//        lookupIndex[2] = (b0[2] >> 8) & 0xff;
//        lookupIndex[3] = (b0[3] >> 8) & 0xff;
//
//        // Element-wise lookups
//        (enableMask[0]) ? passwordOffsetVector[0] = reverseCharset[lookupIndex[0]] : passwordOffsetVector[0] = 0x00;
//        (enableMask[1]) ? passwordOffsetVector[1] = reverseCharset[lookupIndex[1]] : passwordOffsetVector[1] = 0x00;
//        (enableMask[2]) ? passwordOffsetVector[2] = reverseCharset[lookupIndex[2]] : passwordOffsetVector[2] = 0x00;
//        (enableMask[3]) ? passwordOffsetVector[3] = reverseCharset[lookupIndex[3]] : passwordOffsetVector[3] = 0x00;
//
//        passwordOffsetVector[0]++;
//        passwordOffsetVector[1]++;
//        passwordOffsetVector[2]++;
//        passwordOffsetVector[3]++;
//        printf("passwordOffsetVector %d %d %d %d\n", passwordOffsetVector[0], 
//                passwordOffsetVector[1], passwordOffsetVector[2], passwordOffsetVector[3]);
//
//        
//    }


    
}