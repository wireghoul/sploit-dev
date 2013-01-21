// Testing some HMAC functions to make sure I understand them...

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


#define __device__
#define UINT4 uint32_t

#define UseB0 1
#define UseB1 1
#define UseB2 1
#define UseB3 1
#define UseB4 1
#define UseB5 1
#define UseB6 1
#define UseB7 1
#define UseB8 1
#define UseB9 1
#define UseB10 1
#define UseB11 1
#define UseB12 1
#define UseB13 1

// Sets the character at the given position.  Requires things to be zeroed first!
__device__ inline void SetCharacterAtPosition(unsigned char character, unsigned char position,
        UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
	UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

    int offset = position / 4;

    if (UseB0 & offset == 0) {
        b0 |= character << (8 * (position % 4));
    } else if (UseB1 & offset == 1) {
        b1 |= character << (8 * (position % 4));
    } else if (UseB2 & offset == 2) {
        b2 |= character << (8 * (position % 4));
    } else if (UseB3 & offset == 3) {
        b3 |= character << (8 * (position % 4));
    } else if (UseB4 & offset == 4) {
        b4 |= character << (8 * (position % 4));
    } else if (UseB5 & offset == 5) {
        b5 |= character << (8 * (position % 4));
    } else if (UseB6 & offset == 6) {
        b6 |= character << (8 * (position % 4));
    } else if (UseB7 & offset == 7) {
        b7 |= character << (8 * (position % 4));
    } else if (UseB8 & offset == 8) {
        b8 |= character << (8 * (position % 4));
    } else if (UseB9 & offset == 9) {
        b9 |= character << (8 * (position % 4));
    } else if (UseB10 & offset == 10) {
        b10 |= character << (8 * (position % 4));
    } else if (UseB11 & offset == 11) {
        b11 |= character << (8 * (position % 4));
    } else if (UseB12 & offset == 12) {
        b12 |= character << (8 * (position % 4));
    } else if (UseB13 & offset == 13) {
        b13 |= character << (8 * (position % 4));
    }
}


#include "../../inc/CUDA_Common/CUDAMD5.h"

void doHmac(unsigned char *key, unsigned char *messageBuffer, int messageBlocks) {
    unsigned char ipad[64];
    unsigned char opad[64];
    int i;
    
    uint32_t *blockWords;

    // Set up the pads
    memset(ipad, 0x36, 64);
    memset(opad, 0x5c, 64);

    for (i = 0; i < strlen((const char *)key); i++) {
        ipad[i] ^= key[i];
        opad[i] ^= key[i];
    }

    uint32_t a, b, c, d;

    // Variables to store the inner hash results.
    uint32_t i_a, i_b, i_c, i_d;

    // Init MD5 for inner loop
    a = 0x67452301;
    b = 0xefcdab89;
    c = 0x98badcfe;
    d = 0x10325476;

    // Perform an MD5 on the ipad block - 64 bytes long.  This involves one pass of
    // the MD5 algorithm, with no length/padding added.
    blockWords = (uint32_t *)ipad;
    CUDA_RAW_MD5_STAGE(blockWords[0], blockWords[1], blockWords[2], blockWords[3], 
            blockWords[4], blockWords[5], blockWords[6], blockWords[7],
            blockWords[8], blockWords[9], blockWords[10], blockWords[11],
            blockWords[12], blockWords[13], blockWords[14], blockWords[15],
            a, b, c, d);

    printf("Raw MD5 1st block results:\n");
    printf("a: 0x%08x\n", a);
    printf("b: 0x%08x\n", b);
    printf("c: 0x%08x\n", c);
    printf("d: 0x%08x\n", d);

    // Add the message to the inner pad.  This involves working with the length of the message.
    for (i = 0; i < messageBlocks; i++) {
        blockWords = (uint32_t *)&messageBuffer[i * 64];
        CUDA_RAW_MD5_STAGE(blockWords[0], blockWords[1], blockWords[2], blockWords[3],
                blockWords[4], blockWords[5], blockWords[6], blockWords[7],
                blockWords[8], blockWords[9], blockWords[10], blockWords[11],
                blockWords[12], blockWords[13], blockWords[14], blockWords[15],
                a, b, c, d);
    }


/*
    // Second pass - as the text is null right now, this is all null,
    // except for padding & length (64 bytes * 8 bits).
    CUDA_RAW_MD5_STAGE(0x00000080, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 64 * 8, 0x00000000,
        a, b, c, d);
*/
    printf("Raw MD5 2st block results:\n");
    printf("a: 0x%08x\n", a);
    printf("b: 0x%08x\n", b);
    printf("c: 0x%08x\n", c);
    printf("d: 0x%08x\n", d);

    i_a = a;
    i_b = b;
    i_c = c;
    i_d = d;

    // Now work on the outer pass.  First step is to MD5 hash the opad (64 bytes).
    blockWords = (uint32_t *)opad;
    a = 0x67452301;
    b = 0xefcdab89;
    c = 0x98badcfe;
    d = 0x10325476;
    CUDA_RAW_MD5_STAGE(blockWords[0], blockWords[1], blockWords[2], blockWords[3],
            blockWords[4], blockWords[5], blockWords[6], blockWords[7],
            blockWords[8], blockWords[9], blockWords[10], blockWords[11],
            blockWords[12], blockWords[13], blockWords[14], blockWords[15],
            a, b, c, d);

    // Now add the inner hash results.  Not sure if need to reverse here...
    // Length is 64 + 16 = 80 bytes

    CUDA_RAW_MD5_STAGE(i_a, i_b, i_c, i_d,
        0x00000080, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 80 * 8, 0x00000000,
        a, b, c, d);

    printf("Raw MD5 opad results:\n");
    printf("a: 0x%08x\n", a);
    printf("b: 0x%08x\n", b);
    printf("c: 0x%08x\n", c);
    printf("d: 0x%08x\n", d);



    
}

int main() {
    printf("HMAC testing.\n");

    unsigned char key[] = "key";
    char message[] = "The quick brown fox jumps over the lazy dog";

    unsigned char *messageBuffer;
    uint32_t *messageBufferWords;
    int messageLength, messageLengthBlocks;

    int i, j;

    // Copy the message into the messageBuffer, with appropriate padding and length.
    messageLength = strlen(message);
    printf("Message length in bytes: %d\n", messageLength);

    // Calculate the padded message length in blocks.
    if ((messageLength % 64) == 0) {
        // Exactly block length message.  Need one more block for padding.
        messageLengthBlocks = (messageLength / 64) + 1;
    } else if ((messageLength % 64) >= 56) {
        // Message fills the end of the block where size goes.  Extend by a block.
        messageLengthBlocks = (messageLength / 64) + 2;
    } else {
        // Message does not fill past 55 bytes, just use this block.
        messageLengthBlocks = (messageLength / 64) + 1;
    }

    printf("Message length in blocks: %d\n", messageLengthBlocks);

    messageBuffer = (unsigned char *)malloc(messageLengthBlocks * 64);
    memset(messageBuffer, 0, messageLengthBlocks * 64);

    // Copy the message into the buffer.
    memcpy(messageBuffer, message, messageLength);
    // Set the padding bit beyond the message.
    messageBuffer[messageLength] = 0x80;
    // Set the length
    messageBufferWords = (uint32_t *)messageBuffer;
    messageBufferWords[messageLengthBlocks * 16 - 2] = (messageLength + 64) * 8;
    printf("Set length at word %d\n", messageLengthBlocks * 16 - 2);

    for (i = 0; i < messageLengthBlocks; i++) {
        printf("Block %d: \n", i);
        for (j = 0; j < 64; j++) {
            printf("%02x", messageBuffer[i * 64 + j]);
            if (j == 31) {
                printf("\n");
            }
        }
        printf("\n");
    }

    



    doHmac(key, messageBuffer, messageLengthBlocks);
}