

#ifndef __GRTCHAINRUNNERSHA256_H
#define __GRTCHAINRUNNERSHA256_H

#include "GRT_Common/GRTChainRunner.h"

class GRTChainRunnerSHA256 : public GRTChainRunner {
public:
    GRTChainRunnerSHA256();
private:
    // Hash function for whatever hash is being used
    void hashFunction(unsigned char *hashInput, unsigned char *hashOutput);

    void reduceFunction(unsigned char *password, unsigned char *hash, uint32_t step);
};

#endif