// Playing with salted Domino hashes.


#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#define PLAINTEXT_LENGTH	64
#define CIPHERTEXT_LENGTH	22
#define BINARY_SIZE		9 /* oh, well :P */
#define SALT_SIZE		5

#define DIGEST_SIZE		16
#define BINARY_BUFFER_SIZE	(DIGEST_SIZE-SALT_SIZE)
#define ASCII_DIGEST_LENGTH	(DIGEST_SIZE*2)
#define MIN_KEYS_PER_CRYPT	1
#define MAX_KEYS_PER_CRYPT	1

static unsigned char key_digest[DIGEST_SIZE];
static char saved_key[PLAINTEXT_LENGTH+1];
static unsigned char crypted_key[DIGEST_SIZE];
static unsigned char salt_and_digest[SALT_SIZE+1+ASCII_DIGEST_LENGTH+1+1] =
	"saalt(................................)";
static unsigned int saved_key_len;

static const char *hex_table[] = {
	"00", "01", "02", "03", "04", "05", "06", "07",
	"08", "09", "0A", "0B",	"0C", "0D", "0E", "0F",
	"10", "11", "12", "13", "14", "15", "16", "17",
	"18", "19", "1A", "1B", "1C", "1D", "1E", "1F",
	"20", "21", "22", "23",	"24", "25", "26", "27",
	"28", "29", "2A", "2B", "2C", "2D", "2E", "2F",
	"30", "31", "32", "33", "34", "35", "36", "37",
	"38", "39", "3A", "3B",	"3C", "3D", "3E", "3F",
	"40", "41", "42", "43", "44", "45", "46", "47",
	"48", "49", "4A", "4B", "4C", "4D", "4E", "4F",
	"50", "51", "52", "53",	"54", "55", "56", "57",
	"58", "59", "5A", "5B", "5C", "5D", "5E", "5F",
	"60", "61", "62", "63", "64", "65", "66", "67",
	"68", "69", "6A", "6B",	"6C", "6D", "6E", "6F",
	"70", "71", "72", "73", "74", "75", "76", "77",
	"78", "79", "7A", "7B", "7C", "7D", "7E", "7F",
	"80", "81", "82", "83",	"84", "85", "86", "87",
	"88", "89", "8A", "8B", "8C", "8D", "8E", "8F",
	"90", "91", "92", "93", "94", "95", "96", "97",
	"98", "99", "9A", "9B",	"9C", "9D", "9E", "9F",
	"A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7",
	"A8", "A9", "AA", "AB", "AC", "AD", "AE", "AF",
	"B0", "B1", "B2", "B3",	"B4", "B5", "B6", "B7",
	"B8", "B9", "BA", "BB", "BC", "BD", "BE", "BF",
	"C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
	"C8", "C9", "CA", "CB",	"CC", "CD", "CE", "CF",
	"D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
	"D8", "D9", "DA", "DB", "DC", "DD", "DE", "DF",
	"E0", "E1", "E2", "E3",	"E4", "E5", "E6", "E7",
	"E8", "E9", "EA", "EB", "EC", "ED", "EE", "EF",
	"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7",
	"F8", "F9", "FA", "FB",	"FC", "FD", "FE", "FF"
};

static const unsigned char lotus_magic_table[] = {
	0xbd, 0x56, 0xea, 0xf2, 0xa2, 0xf1, 0xac, 0x2a,
	0xb0, 0x93, 0xd1, 0x9c, 0x1b, 0x33, 0xfd, 0xd0,
	0x30, 0x04, 0xb6, 0xdc, 0x7d, 0xdf, 0x32, 0x4b,
	0xf7, 0xcb, 0x45, 0x9b, 0x31, 0xbb, 0x21, 0x5a,
	0x41, 0x9f, 0xe1, 0xd9, 0x4a, 0x4d, 0x9e, 0xda,
	0xa0, 0x68, 0x2c, 0xc3, 0x27, 0x5f, 0x80, 0x36,
	0x3e, 0xee, 0xfb, 0x95, 0x1a, 0xfe, 0xce, 0xa8,
	0x34, 0xa9, 0x13, 0xf0, 0xa6, 0x3f, 0xd8, 0x0c,
	0x78, 0x24, 0xaf, 0x23, 0x52, 0xc1, 0x67, 0x17,
	0xf5, 0x66, 0x90, 0xe7, 0xe8, 0x07, 0xb8, 0x60,
	0x48, 0xe6, 0x1e, 0x53, 0xf3, 0x92, 0xa4, 0x72,
	0x8c, 0x08, 0x15, 0x6e, 0x86, 0x00, 0x84, 0xfa,
	0xf4, 0x7f, 0x8a, 0x42, 0x19, 0xf6, 0xdb, 0xcd,
	0x14, 0x8d, 0x50, 0x12, 0xba, 0x3c, 0x06, 0x4e,
	0xec, 0xb3, 0x35, 0x11, 0xa1, 0x88, 0x8e, 0x2b,
	0x94, 0x99, 0xb7, 0x71, 0x74, 0xd3, 0xe4, 0xbf,
	0x3a, 0xde, 0x96, 0x0e, 0xbc, 0x0a, 0xed, 0x77,
	0xfc, 0x37, 0x6b, 0x03, 0x79, 0x89, 0x62, 0xc6,
	0xd7, 0xc0, 0xd2, 0x7c, 0x6a, 0x8b, 0x22, 0xa3,
	0x5b, 0x05, 0x5d, 0x02, 0x75, 0xd5, 0x61, 0xe3,
	0x18, 0x8f, 0x55, 0x51, 0xad, 0x1f, 0x0b, 0x5e,
	0x85, 0xe5, 0xc2, 0x57, 0x63, 0xca, 0x3d, 0x6c,
	0xb4, 0xc5, 0xcc, 0x70, 0xb2, 0x91, 0x59, 0x0d,
	0x47, 0x20, 0xc8, 0x4f, 0x58, 0xe0, 0x01, 0xe2,
	0x16, 0x38, 0xc4, 0x6f, 0x3b, 0x0f, 0x65, 0x46,
	0xbe, 0x7e, 0x2d, 0x7b, 0x82, 0xf9, 0x40, 0xb5,
	0x1d, 0x73, 0xf8, 0xeb, 0x26, 0xc7, 0x87, 0x97,
	0x25, 0x54, 0xb1, 0x28, 0xaa, 0x98, 0x9d, 0xa5,
	0x64, 0x6d, 0x7a, 0xd4, 0x10, 0x81, 0x44, 0xef,
	0x49, 0xd6, 0xae, 0x2e, 0xdd, 0x76, 0x5c, 0x2f,
	0xa7, 0x1c, 0xc9, 0x09, 0x69, 0x9a, 0x83, 0xcf,
	0x29, 0x39, 0xb9, 0xe9, 0x4c, 0xff, 0x43, 0xab,
	/* double power! */
	0xbd, 0x56, 0xea, 0xf2, 0xa2, 0xf1, 0xac, 0x2a,
	0xb0, 0x93, 0xd1, 0x9c, 0x1b, 0x33, 0xfd, 0xd0,
	0x30, 0x04, 0xb6, 0xdc, 0x7d, 0xdf, 0x32, 0x4b,
	0xf7, 0xcb, 0x45, 0x9b, 0x31, 0xbb, 0x21, 0x5a,
	0x41, 0x9f, 0xe1, 0xd9, 0x4a, 0x4d, 0x9e, 0xda,
	0xa0, 0x68, 0x2c, 0xc3, 0x27, 0x5f, 0x80, 0x36,
	0x3e, 0xee, 0xfb, 0x95, 0x1a, 0xfe, 0xce, 0xa8,
	0x34, 0xa9, 0x13, 0xf0, 0xa6, 0x3f, 0xd8, 0x0c,
	0x78, 0x24, 0xaf, 0x23, 0x52, 0xc1, 0x67, 0x17,
	0xf5, 0x66, 0x90, 0xe7, 0xe8, 0x07, 0xb8, 0x60,
	0x48, 0xe6, 0x1e, 0x53, 0xf3, 0x92, 0xa4, 0x72,
	0x8c, 0x08, 0x15, 0x6e, 0x86, 0x00, 0x84, 0xfa,
	0xf4, 0x7f, 0x8a, 0x42, 0x19, 0xf6, 0xdb, 0xcd,
	0x14, 0x8d, 0x50, 0x12, 0xba, 0x3c, 0x06, 0x4e,
	0xec, 0xb3, 0x35, 0x11, 0xa1, 0x88, 0x8e, 0x2b,
	0x94, 0x99, 0xb7, 0x71, 0x74, 0xd3, 0xe4, 0xbf,
	0x3a, 0xde, 0x96, 0x0e, 0xbc, 0x0a, 0xed, 0x77,
	0xfc, 0x37, 0x6b, 0x03, 0x79, 0x89, 0x62, 0xc6,
	0xd7, 0xc0, 0xd2, 0x7c, 0x6a, 0x8b, 0x22, 0xa3,
	0x5b, 0x05, 0x5d, 0x02, 0x75, 0xd5, 0x61, 0xe3,
	0x18, 0x8f, 0x55, 0x51, 0xad, 0x1f, 0x0b, 0x5e,
	0x85, 0xe5, 0xc2, 0x57, 0x63, 0xca, 0x3d, 0x6c,
	0xb4, 0xc5, 0xcc, 0x70, 0xb2, 0x91, 0x59, 0x0d,
	0x47, 0x20, 0xc8, 0x4f, 0x58, 0xe0, 0x01, 0xe2,
	0x16, 0x38, 0xc4, 0x6f, 0x3b, 0x0f, 0x65, 0x46,
	0xbe, 0x7e, 0x2d, 0x7b, 0x82, 0xf9, 0x40, 0xb5,
	0x1d, 0x73, 0xf8, 0xeb, 0x26, 0xc7, 0x87, 0x97,
	0x25, 0x54, 0xb1, 0x28, 0xaa, 0x98, 0x9d, 0xa5,
	0x64, 0x6d, 0x7a, 0xd4, 0x10, 0x81, 0x44, 0xef,
	0x49, 0xd6, 0xae, 0x2e, 0xdd, 0x76, 0x5c, 0x2f,
	0xa7, 0x1c, 0xc9, 0x09, 0x69, 0x9a, 0x83, 0xcf,
	0x29, 0x39, 0xb9, 0xe9, 0x4c, 0xff, 0x43, 0xab,
};


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



void newRC4Playing(char *passwd, uint8_t len, unsigned char *crypted_key) {
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
    
    //printf("New Password Loading\n");
    //dumpRC4_Block(RC4_Block);
    
    RC4_Key[0] = S_BOX[RC4_Block[16]];
    //printf("Key[0]: %02x\n", RC4_Key[0]);
    // Generate the key based on the loaded password.
    for(int j = 1; j < 16; j ++) {
        RC4_Key[j] = S_BOX[RC4_Key[j - 1] ^ RC4_Block[16 + j]];
        //printf("Key[%d]: %02x\n", j, RC4_Key[j]);
    }
    
    //dumpRC4_Key(RC4_Key);

    uint8_t X;
    uint8_t i;
    uint8_t offset = 32;

    X = RC4_Block[15];
    //printf("\n5\n");
    //dumpRC4_Block(RC4_Block);

    for (i = 16; i < 48; i++, offset--) {
        X = (RC4_Block[i] ^= S_BOX[offset + X]);
    }
    //printf("\n6\n");
    //dumpRC4_Block(RC4_Block);

    for (i = 17; i > 0; i--) {
        int j;
        offset = 48;

        for (j = 0; j < 48; j++, offset--) {
            X = (RC4_Block[j] ^= S_BOX[X + offset]);
        }
    }
    //printf("\n7\n");
    //dumpRC4_Block(RC4_Block);

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

    //printf("\n8\n");
    //dumpRC4_Block(RC4_Block);

    X = 0;

    for(i = 17; i > 0; i --) {
      register int j;

      offset = 48;
      for(j = 0; j < 48; j ++, offset --) X = (RC4_Block[j] ^= S_BOX[X + offset]);
    }
    
    offset = 48;
    for(i = 0; i < 16; i ++, offset --)   X = (RC4_Block[i] ^= S_BOX[X + offset]);

  printf("Crypted Hash: ");
  for (i = 0; i < 16; i++) {
      printf("%02x", RC4_Block[i]);
      crypted_key[i] = RC4_Block[i];
  }
  printf("\n");
}

static void dominosec_decode(char *ascii_cipher, unsigned char *binary)
{
	unsigned int out = 0, apsik = 0, loop;
	unsigned int i;
	unsigned char ch;

	ascii_cipher += 2;
	i = 0;
	do {
		if (apsik < 8) {
			/* should be using proper_mul, but what the heck...
			it's nearly the same :] */
			loop = 2; /* ~ loop = proper_mul(13 - apsik); */
			apsik += loop*6;

			do {
				out <<= 6;
				ch = *ascii_cipher;

				if (ch < '0' || ch > '9')
					if (ch < 'A' || ch > 'Z')
						if (ch < 'a' || ch > 'z')
							if (ch != '+')
								if (ch == '/')
									out += '?';
								else
									; /* shit happens */
							else
								out += '>';
						else
							out += ch-'=';
					else
						out += ch-'7';
				else
					out += ch-'0';
				++ascii_cipher;
			} while (--loop);
		}

		loop = apsik-8;
		ch = out >> loop;
		*(binary+i) = ch;
		ch <<= loop;
		apsik = loop;
		out -= ch;
	} while (++i < 15);

	binary[3] += -4;
}



struct cipher_binary_struct {
	unsigned char salt[SALT_SIZE];
	unsigned char hash[BINARY_BUFFER_SIZE];
} cipher_binary;

static void mdtransform(unsigned char state[16], unsigned char checksum[16], unsigned char block[16])
{
	unsigned char x[48];
	unsigned int t = 0;
	unsigned int i,j;
	unsigned char * pt;
	unsigned char c;

	memcpy(x, state, 16);
	memcpy(x+16, block, 16);
        
        printf("x: ");
        for (int z = 0; z < 48; z++) {printf("%02x", x[z]);}
        printf("\n");

	for(i=0;i<16;i++)
		x[i+32] = state[i] ^ block[i];

	for (i = 0; i < 18; ++i)
	{
		pt = (unsigned char*)&x;
		for (j = 48; j > 0; j--)
		{
			*pt ^= lotus_magic_table[j+t];
			t = *pt;
			pt++;
		}
	}

	memcpy(state, x, 16);

	t = checksum[15];
	for (i = 0; i < 16; i++)
	{
		c = lotus_magic_table[block[i]^t];
		checksum[i] ^= c;
		t = checksum[i];
	}
}

static void mdtransform_norecalc(unsigned char state[16], unsigned char block[16])
{
	unsigned char x[48], *pt;
	unsigned int t = 0;
	unsigned int i,j;

	memcpy(x, state, 16);
	memcpy(x+16, block, 16);

	for(i=0;i<16;i++)
		x[i+32] = state[i] ^ block[i];

	for(i = 0; i < 18; ++i)
	{
		pt = (unsigned char*)&x;
		for (j = 48; j > 0; j--)
		{
			*pt ^= lotus_magic_table[j+t];
			t = *pt;
			pt++;
		}
  	}

	memcpy(state, x, 16);
}

static void domino_big_md(unsigned char * saved_key, int size, unsigned char * crypt_key)
{
	unsigned char state[16] = {0};
	unsigned char checksum[16] = {0};
	unsigned char block[16];
	unsigned int x;
	unsigned int curpos = 0;

	while(curpos + 15 < size)
	{
            printf("Running block mode 1\n");
		memcpy(block, saved_key + curpos, 16);
                printf("Pre state:\n");
                printf("state:    ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", state[i]);
                }
                printf("\n");
                printf("checksum: ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", checksum[i]);
                }
                printf("\n");
                printf("block:    ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", block[i]);
                }
                printf("\n");
                
		mdtransform(state, checksum, block);
                printf("Post state:\n");
                printf("state:    ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", state[i]);
                }
                printf("\n");
                printf("checksum: ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", checksum[i]);
                }
                printf("\n");
                printf("block:    ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", block[i]);
                }
                printf(" ");
                for (int i = 0; i < 16; i++) {
                    printf("%c", block[i]);
                }
                printf("\n");
		
                curpos += 16;
	}

	if(curpos != size)
	{
            printf("Block mode 2\n");
		x = size - curpos;
		memcpy(block, saved_key + curpos, x);
		memset(block + x, 16 - x, 16 - x);
                printf("Pre state:\n");
                printf("state:    ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", state[i]);
                }
                printf("\n");
                printf("checksum: ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", checksum[i]);
                }
                printf("\n");
                printf("block:    ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", block[i]);
                }
                printf("\n");

                mdtransform(state, checksum, block);

                printf("Post state:\n");
                printf("state:    ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", state[i]);
                }
                printf("\n");
                printf("checksum: ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", checksum[i]);
                }
                printf("\n");
                printf("block:    ");
                for (int i = 0; i < 16; i++) {
                    printf("%02x ", block[i]);
                }
                printf("\n");
	}
	else
	{
            printf("Block mode 3\n");
		memset(block, 16, 16);
		mdtransform(state, checksum, block);
	}

        printf("Running final recalc\n");

        printf("Pre state:\n");
        printf("state:    ");
        for (int i = 0; i < 16; i++) {
            printf("%02x ", state[i]);
        }
        printf("\n");
        printf("checksum: ");
        for (int i = 0; i < 16; i++) {
            printf("%02x ", checksum[i]);
        }
        printf("\n");

        mdtransform_norecalc(state, checksum);

        printf("Post state:\n");
        printf("state:    ");
        for (int i = 0; i < 16; i++) {
            printf("%02x ", state[i]);
        }
        printf("\n");
        printf("checksum: ");
        for (int i = 0; i < 16; i++) {
            printf("%02x ", checksum[i]);
        }
        printf("\n");

	memcpy(crypt_key, state, 16);
}

void runFullHash(unsigned char *buffer, int hashBlocks, unsigned char *hashoutput) {
    uint8_t MD2_Block[48];
    uint32_t *MD2_Block32 = (uint32_t *) MD2_Block;

    uint8_t MD2_Key[16];
    uint32_t *MD2_Key32 = (uint32_t *) MD2_Key;
    uint32_t passSize;
    uint32_t passBlock;
	unsigned int i,j;
	unsigned char c;

    unsigned int t = 0;

    uint8_t *pt;

    memset(MD2_Block, 0, 48);
    memset(MD2_Key, 0, 48);

    // Loop over all the input blocks
    for (passBlock = 0; passBlock < hashBlocks; passBlock++) {
        for (i = 0; i < 16; i++)
            MD2_Block[i + 32] = MD2_Block[i] ^ buffer[(16 * passBlock) + i];

        for (i = 0; i < 18; ++i) {
            pt = (unsigned char*) &MD2_Block;
            t = 0;
            for (j = 48; j > 0; j--) {
                *pt ^= lotus_magic_table[j + t];
                t = *pt;
                pt++;
            }
        }

        t = MD2_Key[15];
        for (i = 0; i < 16; i++) {
            c = lotus_magic_table[buffer[(16 * passBlock) + i]^t];
            MD2_Key[i] ^= c;
            t = MD2_Key[i];
        }
    }
    
    // End of main block
}


static void dominosec_set_salt(void *salt)
{
	memcpy(salt_and_digest, salt, SALT_SIZE);
}

static void dominosec_set_key(char *key, int index)
{
	unsigned char *offset = salt_and_digest+6;
	unsigned int i;

	saved_key_len = strlen(key);
	strncpy(saved_key, key, PLAINTEXT_LENGTH);
        saved_key[PLAINTEXT_LENGTH] = 0;

	domino_big_md((unsigned char*)key, saved_key_len, key_digest);

	i = 0;
	do {
		memcpy(offset, *(hex_table+*(key_digest+i)), 2);
		offset += 2;
	} while (++i < 14);

	/*
	 * Not (++i < 16) !
	 * Domino will do hash of first 34 bytes ignoring The Fact that now
	 * there is a salt at a beginning of buffer. This means that last 5
	 * bytes "EEFF)" of password digest are meaningless.
	 */
}

static char *dominosec_get_key(int index)
{
	return saved_key;
}

static void dominosec_crypt_all(int count)
{
	domino_big_md(salt_and_digest, 34, crypted_key);
}

int main (int argc, char *argv[]) {
    
    if (argc != 3) {
        printf("Use: binaryname (salted_hash_stuff) (plaintext)\n");
        exit(1);
    }
    
    printf("Salted Hash: %s\n", argv[1]);
    printf("Plaintext to try: %s\n", argv[2]);
    
    unsigned char binarydata[128];
    memset(binarydata, 0, sizeof(binarydata));
    
    unsigned char crypted_hash[16];
    
    dominosec_decode(argv[1], (unsigned char*)&cipher_binary);

    printf("Salt: ");
    for (int i = 0; i < SALT_SIZE; i++) {
        printf("%02x", cipher_binary.salt[i]);
    }
    printf("\n");

    printf("Hash: ");
    for (int i = 0; i < DIGEST_SIZE; i++) {
        printf("%02x", cipher_binary.hash[i]);
    }
    printf("\n");
    
    dominosec_set_key(argv[2], 0);
    memcpy(salt_and_digest, cipher_binary.salt, SALT_SIZE);
    
    printf("digest: ");
    for (int i = 0; i < strlen((char *)salt_and_digest); i++) {
        printf("%c", salt_and_digest[i]);
    }
    printf("\n");
    
    domino_big_md(salt_and_digest, 34, crypted_key);
    
    printf("Final: ");
    for (int i = 0; i < DIGEST_SIZE; i++) {
        printf("%02x", crypted_key[i]);
    }
    printf("\n\n");
    
    
    newRC4Playing(argv[2], strlen(argv[2]), crypted_hash);
    
    // Set up the new shiny 34 bytes of data to hash.
    unsigned char dataToHash[64];
    
    char hashString[33];
    memset(hashString, 0, 33);
    for (int i = 0; i < 16; i++) {
        sprintf(hashString, "%s%02X", hashString, crypted_hash[i]);
    }
    
    printf("Hash string, in ascii: %s\n", hashString);
    
    memset(dataToHash, 0, 64);
    
    // Copy 5 bytes of salt into the data
    for (int i = 0; i < 5; i++) {
        dataToHash[i] = cipher_binary.salt[i];
    }
    dataToHash[5] = '(';
    for (int i = 0; i < 28; i++) {
        dataToHash[6 + i] = hashString[i];
    }
    
    printf("Data to hash: ");
    for (int i = 0; i < 34; i++) {
        printf("%02x", dataToHash[i]);
    }
    printf("\n");
    
    unsigned char output[64];
    memset(output, 0, 64);
    
    domino_big_md(dataToHash, 34, output);
    
    printf("Output: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", output[i]);
    }
    /*
    
    // Test padding stuff
    
    unsigned char output[128];
    
    int paddedLength = strlen(argv[2]);
    
    printf("Password length: %d\n", paddedLength);
    
    // Add at least one padding byte.
    paddedLength++;
    
    while ((paddedLength % 16)) {
        paddedLength++;
    }
    
    printf("Revised padded length: %d\n", paddedLength);
    
    
    
    unsigned char *paddedHash = (unsigned char *)malloc(paddedLength);
    memcpy(paddedHash, argv[2], strlen(argv[2]));
    
    char paddingValue = (paddedLength - strlen(argv[2]));
    printf("Padding value: %d\n", paddingValue);
    
    for (int i = strlen(argv[2]); i < paddedLength; i++) {
        paddedHash[i] = paddingValue;
    }
    
    for (int i = 0; i < paddedLength; i++) {
        printf("%02x", paddedHash[i]);
    }
    
    runFullHash(paddedHash, (paddedLength / 16), output);
    */
    printf("\n\n");
}