#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "GRT_Common/GRTChainRunner.h"

GRTChainRunner::GRTChainRunner(int newHashLengthBytes, int newHashBlockSizeBytes) {
    this->TableHeader = NULL;
    this->hashLengthBytes = newHashLengthBytes;
    this->hashBlockSizeBytes = newHashBlockSizeBytes;
    this->showEachChain = 0;
}

GRTChainRunner::~GRTChainRunner() {
    // Nothing to do for now...
}

void GRTChainRunner::setTableHeader(GRTTableHeader *newTableHeader) {
    this->TableHeader = newTableHeader;

    char** hostCharset2D; // The 16x256 array of characters
    char *CharsetLengths;
    int i;

    hostCharset2D = this->TableHeader->getCharset();
    CharsetLengths = this->TableHeader->getCharsetLengths();

    this->charsetLength = CharsetLengths[0];

    printf("Charset length: %d\n", charsetLength);

    for (i = 0; i < 512; i++) {
        this->charset[i] = hostCharset2D[0][i % charsetLength];
    }

    for (i = 0; i < 16; i++) {
        delete[] hostCharset2D[i];
    }
    delete[] hostCharset2D;
    delete[] CharsetLengths;

}

int GRTChainRunner::generateChain(hashPasswordData * chain, uint64_t numberOfSteps = 0) {
    // Generate a chain based on the password.
    //printf("GRTChainRunner::generateChain\n");

    // Make sure we have what we need.
    if (!this->TableHeader) {
        printf("Must set table header before attempting to call generateChain!\n");
        exit(1);
    }

    unsigned char *password;
    unsigned char *hashInputBlock;
    unsigned char *hash;
    uint32_t i, j;

    uint32_t stepsToRun;

    // Allocate space for the hash blocks.
    password = (unsigned char *)malloc(MAX_PASSWORD_LENGTH);
    hash = (unsigned char *)malloc(this->hashLengthBytes);
    hashInputBlock = (unsigned char *)malloc(this->hashBlockSizeBytes);

    memset(password, 0, MAX_PASSWORD_LENGTH);
    memset(hash, 0, this->hashLengthBytes);
    memset(hashInputBlock, 0, this->hashBlockSizeBytes);

    memcpy(password, chain->password, MAX_PASSWORD_LENGTH);

    // Put the initial password in the hash block
    memcpy(hashInputBlock, password, this->TableHeader->getPasswordLength());

    if (numberOfSteps == 0) {
        stepsToRun = this->TableHeader->getChainLength();
    } else {
        stepsToRun = numberOfSteps;
    }

    // Loop through the chain.
    for (i = 0; i < stepsToRun; i++) {
        if (this->showEachChain) {
            printf("Step %d: \n", i);
            printf("%s:", password);
        }

        // If we are requesting a number of steps, copy the password in too.
        if (numberOfSteps) {
            memcpy(chain->password, password, MAX_PASSWORD_LENGTH);
        }

        this->hashFunction(hashInputBlock, hash);
        if (this->showEachChain) {
            for (j = 0; j < this->hashLengthBytes; j++) {
                printf("%02x", hash[j]);
            }
        printf("\n");
        }
        memset(password, 0, MAX_PASSWORD_LENGTH);
        this->reduceFunction(password, hash, i);
        memset(hashInputBlock, 0, this->hashBlockSizeBytes);
        memcpy(hashInputBlock, password, this->TableHeader->getPasswordLength());
    }
    if (this->showEachChain) {
        printf("Final Hash: ");
        for (j = 0; j < this->hashLengthBytes; j++) {
            printf("%02x", hash[j]);
        }
        printf("\n");
    }


    memcpy(chain->hash, hash, this->hashLengthBytes);
    
    free(password);
    free(hash);
    free(hashInputBlock);

    return 1;
}

// Checks to ensure a chain is correct.  Returns true if it is, false if fail.
int GRTChainRunner::verifyChain(hashPasswordData *chainToCheck) {
    hashPasswordData generatedChain;
    int bitsInHash;

    bitsInHash = this->TableHeader->getBitsInHash();


    memcpy(generatedChain.password, chainToCheck->password, MAX_PASSWORD_LENGTH);

    this->generateChain(&generatedChain);

    int j;

    if (memcmpBits(generatedChain.hash, chainToCheck->hash, bitsInHash) != 0) {
        printf("\nChain mismatch!\n");
        printf("Hash in table  : ");
        for (j = 0; j < this->hashLengthBytes; j++) {
            printf("%02x", chainToCheck->hash[j]);
        }
        printf("\nHash calculated: ");
        for (j = 0; j < this->hashLengthBytes; j++) {
            printf("%02x", generatedChain.hash[j]);
        }
        printf("\n");
        return 0;
    }
    return 1;
}

void GRTChainRunner::setShowEachChain(int newShowEachChain) {
    this->showEachChain = newShowEachChain;
}

hashPasswordData GRTChainRunner::getLinkAtChainIndex(hashPasswordData * chainToRegen, uint64_t index) {
    hashPasswordData returnChain;

    memset(&returnChain, 0, sizeof(hashPasswordData));

    this->generateChain(chainToRegen, index);

    memcpy(returnChain.hash, chainToRegen->hash, this->hashLengthBytes);

    return returnChain;
}