#include <stdio.h>
#include <stdlib.h>

static unsigned const char cov_2char[65]="./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
static int hash_ret_len1=16;

static int b64_pton(char const *src, char *target)
{
    int y,j;
    unsigned char c1,c2,c3,c4;

    c1=c2=c3=c4=0;
    y=0;
    for (j=0; j<65; j++) if (cov_2char[j]==src[3]) c1=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[2]) c2=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[1]) c3=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[0]) c4=(j&255);
    y=(c1<<26)|(c2<<20)|(c3<<14)|(c4<<8);
    target[2]=(y>>24)&255;
    target[1]=(y>>16)&255;
    target[0]=(y>>8)&255;

    y=0;
    for (j=0; j<65; j++) if (cov_2char[j]==src[7]) c1=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[6]) c2=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[5]) c3=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[4]) c4=(j&255);
    y=(c1<<26)|(c2<<20)|(c3<<14)|(c4<<8);
    target[5]=(y>>24)&255;
    target[4]=(y>>16)&255;
    target[3]=(y>>8)&255;

    y=0;
    for (j=0; j<65; j++) if (cov_2char[j]==src[11]) c1=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[10]) c2=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[9]) c3=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[8]) c4=(j&255);
    y=(c1<<26)|(c2<<20)|(c3<<14)|(c4<<8);
    target[8]=(y>>24)&255;
    target[7]=(y>>16)&255;
    target[6]=(y>>8)&255;

    y=0;
    for (j=0; j<65; j++) if (cov_2char[j]==src[15]) c1=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[14]) c2=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[13]) c3=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[12]) c4=(j&255);
    y=(c1<<26)|(c2<<20)|(c3<<14)|(c4<<8);
    target[11]=(y>>24)&255;
    target[10]=(y>>16)&255;
    target[9]=(y>>8)&255;

    y=0;
    for (j=0; j<65; j++) if (cov_2char[j]==src[19]) c1=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[18]) c2=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[17]) c3=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[16]) c4=(j&255);
    y=(c1<<26)|(c2<<20)|(c3<<14)|(c4<<8);
    target[14]=(y>>24)&255;
    target[13]=(y>>16)&255;
    target[12]=(y>>8)&255;

    y=0;
    for (j=0; j<65; j++) if (cov_2char[j]==src[21]) c1=(j&255);
    for (j=0; j<65; j++) if (cov_2char[j]==src[20]) c2=(j&255);
printf("c1: %02x\n", c1);
printf("C2: %02x\n", c2);
    y=(c1<<26)|(c2<<20)|0;//(c3<<14)|(c4<<8);
printf("y: %08x\n", y);
    target[15]=(y>>20)&255;
    target[16]=0;
    return 0;
}

int main() {

//char src[] = "wsCNMw4OGMmFMFFBScx22/";
char src[] = "4o/I5JYcFMXtuo52vFVit/";
char target[16];

b64_pton(src, target);
int i;
for (i = 0; i < 16; i++) {
printf("%02x", (unsigned char)target[i]);
}
printf("\n");
}

