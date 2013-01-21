// Fucking mtwist class!

#ifndef __CHRANDOM_H
#define __CHRANDOM_H

#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <stdint.h>

#define UNIT_TEST 0


// Random number generation for Cryptohaze tools
// Replacement for the ever-annoying mtwist.c
// Uses boost random.
class CHRandom {
public:
    CHRandom() {
        // Default constructor: Use hardware random sources
        // to set the seed value & seed the internal RNG
        boost::random_device hardwareRandom;
        this->seedValue = hardwareRandom();
        this->randomNumberSource.seed(this->seedValue);
    }

    // Sets the seed of the twister to a specified value
    void setSeed(uint32_t newSeed) {
        this->seedValue = newSeed;
        this->randomNumberSource.seed(this->seedValue);
    }

    // Gets the currently active seed
    uint32_t getSeed() {
        return this->seedValue;
    }

    // Returns a random uint32_t from the generator
    uint32_t getRandomValue() {
        return this->randomNumberSource();
    }

    // Skips the specified number of steps
    void skipSome(uint32_t numberToSkip) {
        this->randomNumberSource.discard(numberToSkip);
    }


private:
    // Boost random number generator matching
    // mtwist.c implementation
    boost::mt19937 randomNumberSource;

    // Random seed value being used for current system.
    uint32_t seedValue;
};

#endif


#if UNIT_TEST

#include <stdio.h>
#include <stdlib.h>
#include "CH_Common/mtwist.h"


int main () {
    uint32_t seedValue = 1234;
    int i;

    CHRandom RandomSequence;
    CHRandom RandomSequence2;

    RandomSequence.setSeed(seedValue);
    mt_seed32new(seedValue);

    for (i = 0; i < 100; i++) {
        printf("Value %03d: %12lu %12lu\n", i, mt_lrand(), RandomSequence.getRandomValue());
    }

    printf("\n\n");
    
    mt_seed32new(RandomSequence2.getSeed());

    for (i = 0; i < 100; i++) {
        mt_lrand();
    }
    RandomSequence2.skipSome(100);

    for (i = 0; i < 100; i++) {
        printf("Value %03d: %12lu %12lu\n", i, mt_lrand(), RandomSequence2.getRandomValue());
    }
}
#endif
