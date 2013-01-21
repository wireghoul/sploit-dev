

#ifndef __GRTCHAINRUNNERMD5_H
#define __GRTCHAINRUNNERMD5_H

#include "GRT_Common/GRTChainRunner.h"

class GRTChainRunnerMD5 : public GRTChainRunner {
public:
    GRTChainRunnerMD5();
private:
    // Hash function for whatever hash is being used
    void hashFunction(unsigned char *hashInput, unsigned char *hashOutput);

    void reduceFunction(unsigned char *password, unsigned char *hash, uint32_t step);
    
};

#endif