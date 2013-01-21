// Test duplication of passwords.


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


char charset[] = "abcdefghijklmnopqrstuvwxyz";

#define DuplicatePassword(pass_length) { \
if (pass_length == 1) { \
    b0 = (b0 & 0x000000ff) | ((b0 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 2) { \
    b0 = (b0 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b1 = 0x00000080; \
} else if (pass_length == 3) {\
    b0 = (b0 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b1 = ((b0 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 4) {\
    b1 = b0; \
    b2 = 0x00000080; \
} else if (pass_length == 5) {\
    b1 = (b1 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b2 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 6) {\
    b1 = (b1 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b2 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b3 = 0x00000080; \
} else if (pass_length == 7) {\
    b1 = (b1 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b2 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b3 = ((b1 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 8) {\
    b2 = b0; \
    b3 = b1; \
    b4 = 0x00000080; \
} else if (pass_length == 9) {\
    b2 = (b2 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b3 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x00ffffff) << 8); \
    b4 = ((b1 & 0xff000000) >> 24) | ((b2 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 10) {\
    b2 = (b2 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b3 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b4 = ((b1 & 0xffff0000) >> 16) | ((b2 & 0x0000ffff) << 16); \
    b5 = 0x00000080; \
} else if (pass_length == 11) {\
    b2 = (b2 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b3 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b4 = ((b1 & 0xffffff00) >> 8) | ((b2 & 0x000000ff) << 24); \
    b5 = ((b2 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 12) {\
    b3 = b0; \
    b4 = b1; \
    b5 = b2; \
    b6 = 0x00000080; \
} else if (pass_length == 13) {\
    b3 = (b3 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b4 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x00ffffff) << 8); \
    b5 = ((b1 & 0xff000000) >> 24) | ((b2 & 0x00ffffff) << 8); \
    b6 = ((b2 & 0xff000000) >> 24) | ((b3 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 14) {\
    b3 = (b3 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b4 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b5 = ((b1 & 0xffff0000) >> 16) | ((b2 & 0x0000ffff) << 16); \
    b6 = ((b2 & 0xffff0000) >> 16) | ((b3 & 0x0000ffff) << 16); \
    b7 = 0x00000080; \
} else if (pass_length == 15) {\
    b3 = (b3 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b4 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b5 = ((b1 & 0xffffff00) >> 8) | ((b2 & 0x000000ff) << 24); \
    b6 = ((b2 & 0xffffff00) >> 8) | ((b3 & 0x000000ff) << 24); \
    b7 = ((b3 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 16) {\
    b4 = b0; \
    b5 = b1; \
    b6 = b2; \
    b7 = b3; \
    b8 = 0x00000080; \
} else if (pass_length == 17) {\
    b4 = (b4 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b5 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x00ffffff) << 8); \
    b6 = ((b1 & 0xff000000) >> 24) | ((b2 & 0x00ffffff) << 8); \
    b7 = ((b2 & 0xff000000) >> 24) | ((b3 & 0x00ffffff) << 8); \
    b8 = ((b3 & 0xff000000) >> 24) | ((b4 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 18) {\
    b4 = (b4 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b5 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b6 = ((b1 & 0xffff0000) >> 16) | ((b2 & 0x0000ffff) << 16); \
    b7 = ((b2 & 0xffff0000) >> 16) | ((b3 & 0x0000ffff) << 16); \
    b8 = ((b3 & 0xffff0000) >> 16) | ((b4 & 0x0000ffff) << 16); \
    b9 = 0x00000080; \
} else if (pass_length == 19) {\
    b4 = (b4 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b5 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b6 = ((b1 & 0xffffff00) >> 8) | ((b2 & 0x000000ff) << 24); \
    b7 = ((b2 & 0xffffff00) >> 8) | ((b3 & 0x000000ff) << 24); \
    b8 = ((b3 & 0xffffff00) >> 8) | ((b4 & 0x000000ff) << 24); \
    b9 = ((b4 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 20) {\
    b5 = b0; \
    b6 = b1; \
    b7 = b2; \
    b8 = b3; \
    b9 = b4; \
    b10 = 0x00000080; \
} else if (pass_length == 21) {\
    b5 = (b5 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b6 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x00ffffff) << 8); \
    b7 = ((b1 & 0xff000000) >> 24) | ((b2 & 0x00ffffff) << 8); \
    b8 = ((b2 & 0xff000000) >> 24) | ((b3 & 0x00ffffff) << 8); \
    b9 = ((b3 & 0xff000000) >> 24) | ((b4 & 0x00ffffff) << 8); \
    b10 = ((b4 & 0xff000000) >> 24) | ((b5 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 22) {\
    b5 = (b5 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b6 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b7 = ((b1 & 0xffff0000) >> 16) | ((b2 & 0x0000ffff) << 16); \
    b8 = ((b2 & 0xffff0000) >> 16) | ((b3 & 0x0000ffff) << 16); \
    b9 = ((b3 & 0xffff0000) >> 16) | ((b4 & 0x0000ffff) << 16); \
    b10 = ((b4 & 0xffff0000) >> 16) | ((b5 & 0x0000ffff) << 16); \
    b11 = 0x00000080; \
} else if (pass_length == 23) {\
    b5 = (b5 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b6 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b7 = ((b1 & 0xffffff00) >> 8) | ((b2 & 0x000000ff) << 24); \
    b8 = ((b2 & 0xffffff00) >> 8) | ((b3 & 0x000000ff) << 24); \
    b9 = ((b3 & 0xffffff00) >> 8) | ((b4 & 0x000000ff) << 24); \
    b10 = ((b4 & 0xffffff00) >> 8) | ((b5 & 0x000000ff) << 24); \
    b11 = ((b5 & 0x00ffff00) >> 8) | 0x00800000; \
} else if (pass_length == 24) {\
    b6 = b0; \
    b7 = b1; \
    b8 = b2; \
    b9 = b3; \
    b10 = b4; \
    b11 = b5; \
    b12 = 0x00000080; \
} else if (pass_length == 25) {\
    b6 = (b6 & 0x000000ff) | ((b0 & 0x00ffffff) << 8); \
    b7 = ((b0 & 0xff000000) >> 24) | ((b1 & 0x00ffffff) << 8); \
    b8 = ((b1 & 0xff000000) >> 24) | ((b2 & 0x00ffffff) << 8); \
    b9 = ((b2 & 0xff000000) >> 24) | ((b3 & 0x00ffffff) << 8); \
    b10 = ((b3 & 0xff000000) >> 24) | ((b4 & 0x00ffffff) << 8); \
    b11 = ((b4 & 0xff000000) >> 24) | ((b5 & 0x00ffffff) << 8); \
    b12 = ((b5 & 0xff000000) >> 24) | ((b6 & 0x000000ff) << 8) | 0x00800000; \
} else if (pass_length == 26) {\
    b6 = (b6 & 0x0000ffff) | ((b0 & 0x0000ffff) << 16); \
    b7 = ((b0 & 0xffff0000) >> 16) | ((b1 & 0x0000ffff) << 16); \
    b8 = ((b1 & 0xffff0000) >> 16) | ((b2 & 0x0000ffff) << 16); \
    b9 = ((b2 & 0xffff0000) >> 16) | ((b3 & 0x0000ffff) << 16); \
    b10 = ((b3 & 0xffff0000) >> 16) | ((b4 & 0x0000ffff) << 16); \
    b11 = ((b4 & 0xffff0000) >> 16) | ((b5 & 0x0000ffff) << 16); \
    b12 = ((b5 & 0xffff0000) >> 16) | ((b6 & 0x0000ffff) << 16); \
    b13 = 0x00000080; \
} else if (pass_length == 27) {\
    b6 = (b6 & 0x00ffffff) | ((b0 & 0x000000ff) << 24); \
    b7 = ((b0 & 0xffffff00) >> 8) | ((b1 & 0x000000ff) << 24); \
    b8 = ((b1 & 0xffffff00) >> 8) | ((b2 & 0x000000ff) << 24); \
    b9 = ((b2 & 0xffffff00) >> 8) | ((b3 & 0x000000ff) << 24); \
    b10 = ((b3 & 0xffffff00) >> 8) | ((b4 & 0x000000ff) << 24); \
    b11 = ((b4 & 0xffffff00) >> 8) | ((b5 & 0x000000ff) << 24); \
    b12 = ((b5 & 0xffffff00) >> 8) | ((b6 & 0x000000ff) << 24); \
    b13 = ((b6 & 0x00ffff00) >> 8) | 0x00800000; \
}\
}

int main(int argc, char *argv[]) {
    // Don't need all of them - b14 & b15 are reserved for length.
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13;
    
    // Byte array in memory for the hash to be read from for LE hashes like MD5.
    uint8_t hashInputArray[4 * 16];
    // 32-bit little endian access
    uint32_t *hashArray32 = (uint32_t *)hashInputArray;
    
    if (argc != 2) {
        printf("Usage: %s [passlen]\n", argv[0]);
        exit(1);
    }

    int passLength = atoi(argv[1]);

    memset(hashInputArray, 0, sizeof(hashInputArray));
    
    printf("Setting passlength %d\n", passLength);
    
    for (int i = 0; i < passLength; i++) {
        hashInputArray[i] = charset[i % strlen(charset)];
    }
    
    printf("Password: ");
    for (int i = 0; i < passLength; i++) {
        printf("%c", hashInputArray[i]);
        if ((i % 4) == 3) {
            printf(" ");
        }
    }
    printf("\n");
    
    b0 = hashArray32[0];
    b1 = hashArray32[1];
    b2 = hashArray32[2];
    b3 = hashArray32[3];
    b4 = hashArray32[4];
    b5 = hashArray32[5];
    b6 = hashArray32[6];
    b7 = hashArray32[7];
    b8 = hashArray32[8];
    b9 = hashArray32[9];
    b10 = hashArray32[10];
    b11 = hashArray32[11];
    b12 = hashArray32[12];
    b13 = hashArray32[13];

    DuplicatePassword(passLength);
    
    hashArray32[0] = b0;
    hashArray32[1] = b1;
    hashArray32[2] = b2;
    hashArray32[3] = b3;
    hashArray32[4] = b4;
    hashArray32[5] = b5;
    hashArray32[6] = b6;
    hashArray32[7] = b7;
    hashArray32[8] = b8;
    hashArray32[9] = b9;
    hashArray32[10] = b10;
    hashArray32[11] = b11;
    hashArray32[12] = b12;
    hashArray32[13] = b13;

    printf("Dup Password: ");
    for (int i = 0; i < (passLength * 2); i++) {
        printf("%c", hashInputArray[i]);
        if ((i % 4) == 3) {
            printf(" ");
        }
    }
    printf("\n");

    printf("Dup Hex: ");
    for (int i = 0; i < (passLength * 2) + 1; i++) {
        printf("%02x", hashInputArray[i]);
        if ((i % 4) == 3) {
            printf(" ");
        }
    }
    printf("\n");
    
    // Perform sanity check
    for (int i = 0; i < passLength; i++) {
        if (hashInputArray[i] != hashInputArray[i + passLength]) {
            printf("====Mismatch in position %d!====\n", i);
        }
    }
    if (hashInputArray[passLength * 2] != 0x80) {
        printf("====Missing padding bit!====\n");
    }
    
}
