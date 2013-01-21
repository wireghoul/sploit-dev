


#ifndef __GRTCHAINRUNNER_H
#define __GRTCHAINRUNNER_H

#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTTableHeader.h"

// In GRTTableSearchV2.cpp
int memcmpBits(unsigned char *val1, unsigned char *val2, int bitsToCompare);

// Class for CPU chain running/verification.

class GRTChainRunner {
public:
    // HashLengthBytes: Length of the hash output.
    // HashBlockSizeBytes: Length of the hash input (typically 64).
    GRTChainRunner(int newHashLengthBytes, int newHashBlockSizeBytes);
    ~GRTChainRunner();

    // Set the table header to use for chain running.
    void setTableHeader(GRTTableHeader *);

    // Generate a chain: Take the initial password, generate the end hash.
    // Stores the end hash in the provided structure.
    // Returns 0 on failure, 1 on success.
    int generateChain(hashPasswordData *, uint64_t);

    // Verifies a chain: Returns 1 if it's correct, 0 if it's failure.
    int verifyChain(hashPasswordData *);

    void setShowEachChain(int);

    hashPasswordData getLinkAtChainIndex(hashPasswordData *, uint64_t index);

protected:
    // Storage of the table header.
    GRTTableHeader *TableHeader;

    // Output length of the hash, in bytes
    int hashLengthBytes;

    // Working size of the hash input block
    int hashBlockSizeBytes;

    unsigned char charset[512];
    int charsetLength;

    int showEachChain;

    // Hash function for whatever hash is being used
    virtual void hashFunction(unsigned char *hashInput, unsigned char *hashOutput) = 0;

    virtual void reduceFunction(unsigned char *password, unsigned char *hash, uint32_t step) = 0;
    
};

#endif

