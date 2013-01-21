

#ifndef __GRTCHAINRUNNERSHA1_H
#define __GRTCHAINRUNNERSHA1_H

#include "GRT_Common/GRTChainRunner.h"

class GRTChainRunnerSHA1 : public GRTChainRunner {
public:
    GRTChainRunnerSHA1();
private:
    // Hash function for whatever hash is being used
    void hashFunction(unsigned char *hashInput, unsigned char *hashOutput);

    void reduceFunction(unsigned char *password, unsigned char *hash, uint32_t step);

};

#endif