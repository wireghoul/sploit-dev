// Playing with Domino hashes.

#include <stdio.h>
#include <string.h>
#include <stdint.h>

const uint8_t S_BOX[] = {
  0xBD,0x56,0xEA,0xF2,0xA2,0xF1,0xAC,0x2A,
  0xB0,0x93,0xD1,0x9C,0x1B,0x33,0xFD,0xD0,
  0x30,0x04,0xB6,0xDC,0x7D,0xDF,0x32,0x4B,
  0xF7,0xCB,0x45,0x9B,0x31,0xBB,0x21,0x5A,
  0x41,0x9F,0xE1,0xD9,0x4A,0x4D,0x9E,0xDA,
  0xA0,0x68,0x2C,0xC3,0x27,0x5F,0x80,0x36,
  0x3E,0xEE,0xFB,0x95,0x1A,0xFE,0xCE,0xA8,
  0x34,0xA9,0x13,0xF0,0xA6,0x3F,0xD8,0x0C,
  0x78,0x24,0xAF,0x23,0x52,0xC1,0x67,0x17,
  0xF5,0x66,0x90,0xE7,0xE8,0x07,0xB8,0x60,
  0x48,0xE6,0x1E,0x53,0xF3,0x92,0xA4,0x72,
  0x8C,0x08,0x15,0x6E,0x86,0x00,0x84,0xFA,
  0xF4,0x7F,0x8A,0x42,0x19,0xF6,0xDB,0xCD,
  0x14,0x8D,0x50,0x12,0xBA,0x3C,0x06,0x4E,
  0xEC,0xB3,0x35,0x11,0xA1,0x88,0x8E,0x2B,
  0x94,0x99,0xB7,0x71,0x74,0xD3,0xE4,0xBF,
  0x3A,0xDE,0x96,0x0E,0xBC,0x0A,0xED,0x77,
  0xFC,0x37,0x6B,0x03,0x79,0x89,0x62,0xC6,
  0xD7,0xC0,0xD2,0x7C,0x6A,0x8B,0x22,0xA3,
  0x5B,0x05,0x5D,0x02,0x75,0xD5,0x61,0xE3,
  0x18,0x8F,0x55,0x51,0xAD,0x1F,0x0B,0x5E,
  0x85,0xE5,0xC2,0x57,0x63,0xCA,0x3D,0x6C,
  0xB4,0xC5,0xCC,0x70,0xB2,0x91,0x59,0x0D,
  0x47,0x20,0xC8,0x4F,0x58,0xE0,0x01,0xE2,
  0x16,0x38,0xC4,0x6F,0x3B,0x0F,0x65,0x46,
  0xBE,0x7E,0x2D,0x7B,0x82,0xF9,0x40,0xB5,
  0x1D,0x73,0xF8,0xEB,0x26,0xC7,0x87,0x97,
  0x25,0x54,0xB1,0x28,0xAA,0x98,0x9D,0xA5,
  0x64,0x6D,0x7A,0xD4,0x10,0x81,0x44,0xEF,
  0x49,0xD6,0xAE,0x2E,0xDD,0x76,0x5C,0x2F,
  0xA7,0x1C,0xC9,0x09,0x69,0x9A,0x83,0xCF,
  0x29,0x39,0xB9,0xE9,0x4C,0xFF,0x43,0xAB,
  /* repeated to avoid & 0xff */
  0xBD,0x56,0xEA,0xF2,0xA2,0xF1,0xAC,0x2A,
  0xB0,0x93,0xD1,0x9C,0x1B,0x33,0xFD,0xD0,
  0x30,0x04,0xB6,0xDC,0x7D,0xDF,0x32,0x4B,
  0xF7,0xCB,0x45,0x9B,0x31,0xBB,0x21,0x5A,
  0x41,0x9F,0xE1,0xD9,0x4A,0x4D,0x9E,0xDA,
  0xA0,0x68,0x2C,0xC3,0x27,0x5F,0x80,0x36,
  0x3E,0xEE,0xFB,0x95,0x1A,0xFE,0xCE,0xA8,
  0x34,0xA9,0x13,0xF0,0xA6,0x3F,0xD8,0x0C,
  0x78,0x24,0xAF,0x23,0x52,0xC1,0x67,0x17,
  0xF5,0x66,0x90,0xE7,0xE8,0x07,0xB8,0x60,
  0x48,0xE6,0x1E,0x53,0xF3,0x92,0xA4,0x72,
  0x8C,0x08,0x15,0x6E,0x86,0x00,0x84,0xFA,
  0xF4,0x7F,0x8A,0x42,0x19,0xF6,0xDB,0xCD,
  0x14,0x8D,0x50,0x12,0xBA,0x3C,0x06,0x4E,
  0xEC,0xB3,0x35,0x11,0xA1,0x88,0x8E,0x2B,
  0x94,0x99,0xB7,0x71,0x74,0xD3,0xE4,0xBF,
  0x3A,0xDE,0x96,0x0E,0xBC,0x0A,0xED,0x77,
  0xFC,0x37,0x6B,0x03,0x79,0x89,0x62,0xC6,
  0xD7,0xC0,0xD2,0x7C,0x6A,0x8B,0x22,0xA3,
  0x5B,0x05,0x5D,0x02,0x75,0xD5,0x61,0xE3,
  0x18,0x8F,0x55,0x51,0xAD,0x1F,0x0B,0x5E,
  0x85,0xE5,0xC2,0x57,0x63,0xCA,0x3D,0x6C,
  0xB4,0xC5,0xCC,0x70,0xB2,0x91,0x59,0x0D,
  0x47,0x20,0xC8,0x4F,0x58,0xE0,0x01,0xE2,
  0x16,0x38,0xC4,0x6F,0x3B,0x0F,0x65,0x46,
  0xBE,0x7E,0x2D,0x7B,0x82,0xF9,0x40,0xB5,
  0x1D,0x73,0xF8,0xEB,0x26,0xC7,0x87,0x97,
  0x25,0x54,0xB1,0x28,0xAA,0x98,0x9D,0xA5,
  0x64,0x6D,0x7A,0xD4,0x10,0x81,0x44,0xEF,
  0x49,0xD6,0xAE,0x2E,0xDD,0x76,0x5C,0x2F,
  0xA7,0x1C,0xC9,0x09,0x69,0x9A,0x83,0xCF,
  0x29,0x39,0xB9,0xE9,0x4C,0xFF,0x43,0xAB
};

void dumpRC4_Block(uint8_t *RC4_Block) {
    printf("Block: ");
    for (int i = 0; i < 48; i++) {
        printf("%02x", RC4_Block[i]);
        if ((i % 4) == 3) {
            printf(" ");
        }
    }
    printf("\n");
    printf("Block: ");
    for (int i = 0; i < 48; i++) {
        if ((RC4_Block[i] >= 0x20 ) && (RC4_Block[i] <= 0x7e)) {
            printf("%c", RC4_Block[i]);
        } else {
            printf(".");
        }
        if ((i % 4) == 3) {
            printf(" ");
        }
    }
    printf("\n");
}

void dumpRC4_Key(uint8_t *RC4_Key) {
    printf("Key: ");
    for (int i = 0; i < 16; i++) {
        printf("%02x", RC4_Key[i]);
    }
    printf("\n");
}

void xtn_dom_crypt(char *passwd, int len) {
  static uint8_t RC4_Block[48];
  uint8_t RC4_Key[16];
  int i, j;

  for(j = 0; j < len; j ++) 
    RC4_Block[16 + j] = passwd[j]; //CMAP[(int)passwd[j]];

  dumpRC4_Block(RC4_Block);
  
  for(; j < 16; j ++)
    RC4_Block[16 + j] = (16 - len);

  dumpRC4_Block(RC4_Block);

  memcpy(RC4_Block + 32, RC4_Block + 16, 16);

  dumpRC4_Block(RC4_Block);

  
  RC4_Key[0] = S_BOX[RC4_Block[16]];

  for(j = 1; j < 16; j ++)
    RC4_Key[j] = S_BOX[RC4_Key[j - 1] ^ RC4_Block[16 + j]];

  dumpRC4_Key(RC4_Key);

  if(1) { /* RC4_MixTable1 */
    register uint8_t X;
    register const uint8_t *S;
    register unsigned int i;

    S = &S_BOX[48];

    for(X = 0, i = 0; i < 16; i ++, S --) {
      X = (RC4_Block[i] = S[X]);
    }
    printf("\n5\n");
    dumpRC4_Block(RC4_Block);

    for(i = 16; i < 48; i ++, S --) {
      X = (RC4_Block[i] ^= S[X]);
    }
    printf("\n6\n");
    dumpRC4_Block(RC4_Block);

    for(i = 17; i > 0; i --) {
      register int j;

      S = &S_BOX[48];
      
      for(j = 0; j < 48; j ++, S --) {
	X = (RC4_Block[j] ^= S[X]);
      }
    }
  }
    printf("\n7\n");
    dumpRC4_Block(RC4_Block);

  if(1) {
    unsigned long long *K, *B;

    K = (unsigned long long *)RC4_Key;
    B = (unsigned long long *)RC4_Block;

    B[2] = K[0];
    B[3] = K[1];

    B[4] = B[0] ^ K[0];
    B[5] = B[1] ^ K[1];
  }
    printf("\n8\n");
    dumpRC4_Block(RC4_Block);

  if(1) { /* RC4_MixTable2 */
    register uint8_t X = 0;
    register const uint8_t *S;
    register unsigned int i;

    for(i = 17; i > 0; i --) {
      register int j;

      S = &S_BOX[48];
      for(j = 0; j < 48; j ++, S --) X = (RC4_Block[j] ^= S[X]);
    }
    
    S = &S_BOX[48];
    for(i = 0; i < 16; i ++, S --)   X = (RC4_Block[i] ^= S[X]);
  }

  printf("Hash: ");
  for (i = 0; i < 16; i++) {
      printf("%02x", RC4_Block[i]);
  }
  printf("\n");
}

void newRC4Playing(char *passwd, uint8_t len) {
    uint8_t RC4_Block[48];
    uint32_t *RC4_Block32 = (uint32_t *)RC4_Block;
    
    uint8_t RC4_Key[16];
    uint32_t *RC4_Key32 = (uint32_t *)RC4_Key;
    uint32_t passSize;
    uint32_t passBlock;
    
    memset(RC4_Block, 0, 48);
    
    
    
    // Init the block with the static init vector for the function.
    // This set of 16 bytes appears to be always the same.
    RC4_Block32[0] = 0xd3503c3e; //0x3e3c50d3;
    RC4_Block32[1] = 0x5cc587ab; //0xab87c55c;
    RC4_Block32[2] = 0x9db9d4bc; //0xbcd4b99d;
    RC4_Block32[3] = 0x29d76e38; //0x386ed729;
    
    // Set the password length value.  Not sure what happens if password length
    // is over 16 characters.
    uint8_t lenInverse = (16 - len);
    passSize = lenInverse | (lenInverse << 8) |
            (lenInverse << 16) | (lenInverse << 24);
    
    // Fill the remaining space with the password size.
    for (int i = 4; i < 12; i++) {
        RC4_Block32[i] = passSize;
    }
    
    // Copy the password into positions starting at offset 16 and 32
    for (int i = 0; i < len; i++) {
        RC4_Block[16 + i] = passwd[i];
        RC4_Block[32 + i] = passwd[i];
    }
    
    printf("New Password Loading\n");
    dumpRC4_Block(RC4_Block);
    
    RC4_Key[0] = S_BOX[RC4_Block[16]];
    printf("Key[0]: %02x\n", RC4_Key[0]);
    // Generate the key based on the loaded password.
    for(int j = 1; j < 16; j ++) {
        RC4_Key[j] = S_BOX[RC4_Key[j - 1] ^ RC4_Block[16 + j]];
        printf("Key[%d]: %02x\n", j, RC4_Key[j]);
    }
    
    dumpRC4_Key(RC4_Key);

    uint8_t X;
    uint8_t i;
    uint8_t offset = 32;

    X = RC4_Block[15];
    //printf("\n5\n");
    //dumpRC4_Block(RC4_Block);

    for (i = 16; i < 48; i++, offset--) {
        X = (RC4_Block[i] ^= S_BOX[offset + X]);
    }
    printf("\n6\n");
    dumpRC4_Block(RC4_Block);

    for (i = 17; i > 0; i--) {
        int j;
        offset = 48;

        for (j = 0; j < 48; j++, offset--) {
            X = (RC4_Block[j] ^= S_BOX[X + offset]);
        }
    }
    printf("\n7\n");
    dumpRC4_Block(RC4_Block);

//    B[2] = K[0];
//    B[3] = K[1];
//
//    B[4] = B[0] ^ K[0];
//    B[5] = B[1] ^ K[1];
    // Do the swapping with 32-bit accesses instead of 64 bit
    RC4_Block32[4] = RC4_Key32[0];
    RC4_Block32[5] = RC4_Key32[1];
    RC4_Block32[6] = RC4_Key32[2];
    RC4_Block32[7] = RC4_Key32[3];
    RC4_Block32[8] = RC4_Block32[0] ^ RC4_Key32[0];
    RC4_Block32[9] = RC4_Block32[1] ^ RC4_Key32[1];
    RC4_Block32[10] = RC4_Block32[2] ^ RC4_Key32[2];
    RC4_Block32[11] = RC4_Block32[3] ^ RC4_Key32[3];

    printf("\n8\n");
    dumpRC4_Block(RC4_Block);

    X = 0;

    for(i = 17; i > 0; i --) {
      register int j;

      offset = 48;
      for(j = 0; j < 48; j ++, offset --) X = (RC4_Block[j] ^= S_BOX[X + offset]);
    }
    
    offset = 48;
    for(i = 0; i < 16; i ++, offset --)   X = (RC4_Block[i] ^= S_BOX[X + offset]);

  printf("Hash: ");
  for (i = 0; i < 16; i++) {
      printf("%02x", RC4_Block[i]);
  }
  printf("\n");
}

void vectorRC4Playing(char *passwd, uint8_t len) {
    uint8_t RC4_Block[48];
    uint32_t *RC4_Block32 = (uint32_t *) RC4_Block;
    uint32_t *passwd32 = (uint32_t *) passwd;

    uint8_t RC4_Key[16];
    uint32_t *RC4_Key32 = (uint32_t *) RC4_Key;
    uint32_t passSize;
    uint8_t i;

    memset(RC4_Block, 0, 48);



    // Init the block with the static init vector for the function.
    // This set of 16 bytes appears to be always the same.
    RC4_Block32[0] = 0xd3503c3e; //0x3e3c50d3;
    RC4_Block32[1] = 0x5cc587ab; //0xab87c55c;
    RC4_Block32[2] = 0x9db9d4bc; //0xbcd4b99d;
    RC4_Block32[3] = 0x29d76e38; //0x386ed729;

    // Set the password length value.  Not sure what happens if password length
    // is over 16 characters.
    uint8_t lenInverse = (16 - len);
    passSize = lenInverse | (lenInverse << 8) |
            (lenInverse << 16) | (lenInverse << 24);

    // Fill the remaining space with the password size.
    for (int i = 4; i < 12; i++) {
        RC4_Block32[i] = passSize;
    }
    RC4_Block32[4] = passSize;
    RC4_Block32[5] = passSize;
    RC4_Block32[6] = passSize;
    RC4_Block32[7] = passSize;
    RC4_Block32[8] = passSize;
    RC4_Block32[9] = passSize;
    RC4_Block32[10] = passSize;
    RC4_Block32[11] = passSize;

    // Load the passwords into the positions specified.
    // Set up the password mask based on password length
    uint32_t passMask = 0;
    switch (len % 4) {
        case 0:
            passMask = 0xffffffff;
            break;
        case 1:
            passMask = 0x000000ff;
            break;
        case 2:
            passMask = 0x0000ffff;
            break;
        case 3:
            passMask = 0x00ffffff;
            break;
    }

    if (len <= 4) {
        RC4_Block32[4] &= ~passMask;
        RC4_Block32[4] |= (passwd32[0] & passMask);
        RC4_Block32[8] &= ~passMask;
        RC4_Block32[8] |= (passwd32[0] & passMask);
    } else if (len <= 8) {
        RC4_Block32[4] = passwd32[0];
        RC4_Block32[5] &= ~passMask;
        RC4_Block32[5] |= (passwd32[1] & passMask);
        RC4_Block32[8] = passwd32[0];
        RC4_Block32[9] &= ~passMask;
        RC4_Block32[9] |= (passwd32[1] & passMask);
    }

    printf("New Password Loading\n");
    dumpRC4_Block(RC4_Block);

    // Clear RC4 key
    RC4_Key32[0] = 0;
    RC4_Key32[1] = 0;
    RC4_Key32[2] = 0;
    RC4_Key32[3] = 0;


    {
        uint32_t KeyWord;
        uint32_t PwWord;
        uint8_t sboxIndex;

        PwWord = RC4_Block32[4];

        // 0
        KeyWord = S_BOX[PwWord & 0x000000ff];
        // 1
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];
        // 2
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];
        // 3
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];

        // Endian swap for storage.
        RC4_Key32[0] = (KeyWord << 24) | (KeyWord << 8 & 0xff0000) | (KeyWord >> 8 & 0xff00) | (KeyWord >> 24);

        PwWord = RC4_Block32[5];
        // 0
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];
        // 1
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];
        // 2
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];
        // 3
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];

        RC4_Key32[1] = (KeyWord << 24) | (KeyWord << 8 & 0xff0000) | (KeyWord >> 8 & 0xff00) | (KeyWord >> 24);

        PwWord = RC4_Block32[6];
        // 0
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];
        // 1
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];
        // 2
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];
        // 3
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];

        RC4_Key32[2] = (KeyWord << 24) | (KeyWord << 8 & 0xff0000) | (KeyWord >> 8 & 0xff00) | (KeyWord >> 24);

        PwWord = RC4_Block32[7];
        // 0
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];
        // 1
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];
        // 2
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];
        // 3
        PwWord = PwWord >> 8;
        sboxIndex = (KeyWord & 0xff) ^ (PwWord & 0xff);
        KeyWord = (KeyWord << 8) | S_BOX[sboxIndex];

        RC4_Key32[3] = (KeyWord << 24) | (KeyWord << 8 & 0xff0000) | (KeyWord >> 8 & 0xff00) | (KeyWord >> 24);
    }
    dumpRC4_Key(RC4_Key);

    uint8_t X;
    uint32_t blockWord, newBlockWord;

    blockWord = RC4_Block32[3];
    X = blockWord >> 24;
    printf("X: %02x\n", X);

    // Note: Unroll the **** out of this!
    for (i = 4; i < 12; i++) {
        blockWord = RC4_Block32[i];
        newBlockWord = 0;
        X = (blockWord & 0xff) ^ S_BOX[(48 - (4 * i)) + X];
        newBlockWord |= X;
        newBlockWord = newBlockWord << 8;
        blockWord = blockWord >> 8;
        X = (blockWord & 0xff) ^ S_BOX[(47 - (4 * i)) + X];
        newBlockWord |= X;
        newBlockWord = newBlockWord << 8;
        blockWord = blockWord >> 8;
        X = (blockWord & 0xff) ^ S_BOX[(46 - (4 * i)) + X];
        newBlockWord |= X;
        newBlockWord = newBlockWord << 8;
        blockWord = blockWord >> 8;
        X = (blockWord & 0xff) ^ S_BOX[(45 - (4 * i)) + X];
        newBlockWord |= X;
        RC4_Block32[i] = (newBlockWord << 24) | (newBlockWord << 8 & 0xff0000) | (newBlockWord >> 8 & 0xff00) | (newBlockWord >> 24);
    }

    printf("\n6\n");
    dumpRC4_Block(RC4_Block);

    // Value of X continues to matter through here.

    for (i = 17; i > 0; i--) {
        int j;
        for (j = 0; j < 12; j++) {
            blockWord = RC4_Block32[j];
            newBlockWord = 0;
            X = (blockWord & 0xff) ^ S_BOX[(48 - (4 * j)) + X];
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            X = (blockWord & 0xff) ^ S_BOX[(47 - (4 * j)) + X];
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            X = (blockWord & 0xff) ^ S_BOX[(46 - (4 * j)) + X];
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            X = (blockWord & 0xff) ^ S_BOX[(45 - (4 * j)) + X];
            newBlockWord |= X;
            RC4_Block32[j] = (newBlockWord << 24) | (newBlockWord << 8 & 0xff0000) | (newBlockWord >> 8 & 0xff00) | (newBlockWord >> 24);
        }
    }

    printf("\n7\n");
    dumpRC4_Block(RC4_Block);

    // Do the swapping with 32-bit accesses instead of 64 bit
    {
        RC4_Block32[4] = RC4_Key32[0];
        RC4_Block32[5] = RC4_Key32[1];
        RC4_Block32[6] = RC4_Key32[2];
        RC4_Block32[7] = RC4_Key32[3];
        RC4_Block32[8] = RC4_Block32[0] ^ RC4_Key32[0];
        RC4_Block32[9] = RC4_Block32[1] ^ RC4_Key32[1];
        RC4_Block32[10] = RC4_Block32[2] ^ RC4_Key32[2];
        RC4_Block32[11] = RC4_Block32[3] ^ RC4_Key32[3];
    }
    printf("\n8\n");
    dumpRC4_Block(RC4_Block);

    X = 0;

    for (i = 17; i > 0; i--) {
        register int j;

        for (j = 0; j < 12; j++) {
            blockWord = RC4_Block32[j];
            newBlockWord = 0;
            X = (blockWord & 0xff) ^ S_BOX[(48 - (4 * j)) + X];
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            X = (blockWord & 0xff) ^ S_BOX[(47 - (4 * j)) + X];
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            X = (blockWord & 0xff) ^ S_BOX[(46 - (4 * j)) + X];
            newBlockWord |= X;
            newBlockWord = newBlockWord << 8;
            blockWord = blockWord >> 8;
            X = (blockWord & 0xff) ^ S_BOX[(45 - (4 * j)) + X];
            newBlockWord |= X;
            RC4_Block32[j] = (newBlockWord << 24) | (newBlockWord << 8 & 0xff0000) | (newBlockWord >> 8 & 0xff00) | (newBlockWord >> 24);
        }
    }

    for (int j = 0; j < 4; j++) {
        blockWord = RC4_Block32[j];
        newBlockWord = 0;
        X = (blockWord & 0xff) ^ S_BOX[(48 - (4 * j)) + X];
        newBlockWord |= X;
        newBlockWord = newBlockWord << 8;
        blockWord = blockWord >> 8;
        X = (blockWord & 0xff) ^ S_BOX[(47 - (4 * j)) + X];
        newBlockWord |= X;
        newBlockWord = newBlockWord << 8;
        blockWord = blockWord >> 8;
        X = (blockWord & 0xff) ^ S_BOX[(46 - (4 * j)) + X];
        newBlockWord |= X;
        newBlockWord = newBlockWord << 8;
        blockWord = blockWord >> 8;
        X = (blockWord & 0xff) ^ S_BOX[(45 - (4 * j)) + X];
        newBlockWord |= X;
        RC4_Block32[j] = (newBlockWord << 24) | (newBlockWord << 8 & 0xff0000) | (newBlockWord >> 8 & 0xff00) | (newBlockWord >> 24);
    }

    printf("Hash: ");
    for (i = 0; i < 16; i++) {
        printf("%02x", RC4_Block[i]);
    }
    printf("\n");


}

int main(int argc, char *argv[]) {
    printf("Original\n");
    xtn_dom_crypt(argv[1], strlen(argv[1]));
    
    printf("\n\n\n");
    printf("new\n");
    newRC4Playing(argv[1], strlen(argv[1]));
    printf("\n\n\n");
    printf("vector\n");
    vectorRC4Playing(argv[1], strlen(argv[1]));
}