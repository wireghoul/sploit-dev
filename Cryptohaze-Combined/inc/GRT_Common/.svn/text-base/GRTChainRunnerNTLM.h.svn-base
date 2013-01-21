

#ifndef __GRTCHAINRUNNERNTLM_H
#define __GRTCHAINRUNNERNTLM_H

#include "GRT_Common/GRTChainRunner.h"

class GRTChainRunnerNTLM : public GRTChainRunner {
public:
    GRTChainRunnerNTLM();
private:
    // Hash function for whatever hash is being used
    void hashFunction(unsigned char *hashInput, unsigned char *hashOutput);

    void reduceFunction(unsigned char *password, unsigned char *hash, uint32_t step);

};

#endif