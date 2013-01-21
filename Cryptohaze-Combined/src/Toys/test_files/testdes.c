#include <stdio.h>
#include <string.h>
#include "des.h"


int main() {

bool key[56];

bool outBlk[64];

bool inBlk[64];


// Key to encrypt stuff with.
char magicString[] = "KGS!@#$%";

int i;

memset(key, 0, sizeof(bool) * 56);
memset(inBlk, 0, sizeof(bool) * 64);

printf("Block to encrypt: 0x");
for (i = 0; i < 8; i++) {
    printf("%02x", magicString[i]);
}
printf("\n");

for (i = 0; i < 64; i++) {
    inBlk[i] = (magicString[i / 8] >> (i % 8)) & 0x1;
}


EncryptDES(key, outBlk, inBlk, 1);



}
