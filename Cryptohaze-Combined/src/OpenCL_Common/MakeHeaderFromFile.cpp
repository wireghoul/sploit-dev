/**
 * A tiny little utility that will turn a file into a header that contains
 * a binary representation of that file as a character array.
 * 
 * Public domain.  Use where you will.  This applies to this file only, not the
 * rest of the source.
 * 
 * Written by Bitweasil <bitweasil@gmail.com> in 2012.
 * 
 * Hopefully it's useful.
 * 
 * Usage: ./MakeHeaderFromFile [filename] [data structure name] [output file]
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    
    if (argc != 4) {
        printf("Usage: %s [filename] [data structure name] [output filename]\n",
                argv[0]);
        exit(1);
    }
    
    FILE *inputFile, *outputFile;
    char byteRead, totalBytesCopied = 0;
    
    // Open as binary for Windows platform...
    inputFile = fopen(argv[1], "rb");
    if (!inputFile) {
        printf("Cannot open input file %s!\n", argv[1]);
    }
    // Open as ascii output.
    outputFile = fopen(argv[3], "w");
    if (!outputFile) {
        printf("Cannot open output file %s!\n", argv[3]);
    }
   
    fprintf(outputFile, "char %s[] = {\n    ", argv[2]);
    while (((byteRead = fgetc(inputFile)) != EOF)) {
        fprintf(outputFile, "0x%02x, ", byteRead);
        totalBytesCopied++;
        if ((totalBytesCopied % 8) == 0) {
            fprintf(outputFile, "\n    ");
        }
    }
    // Add null termination byte to string
    fprintf(outputFile, "0x00, ");
    totalBytesCopied++;
    if ((totalBytesCopied % 8) == 0) {
        fprintf(outputFile, "\n    ");
    }
    fprintf(outputFile, "};\n\n");
    fclose(inputFile);
    fclose(outputFile);
}