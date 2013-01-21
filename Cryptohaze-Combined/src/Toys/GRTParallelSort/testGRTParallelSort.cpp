/**
 * Test file for testing GRT parallel sort performance.
 * 
 * Usage: [binary] [megs of RAM to use (total)]
 * 
 * This will create two identical vectors, sort the first with the STL sort
 * function, and then use the new one.  It will time both, and verify the
 * correctness of the vectors after sorting.
 * 
 * Enjoy, cybergray!
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "CH_Common/CHHiresTimer.h"
#include <stdint.h>
#include <time.h>
#include "GRT_Common/GRTCommon.h"

// Hash length in bytes.  Please test with multiple lengths!
const int hashLength = 16;


// Return true if d1 is less than d2
bool hashPasswordDataSortPredicate(const hashPasswordData &d1, const hashPasswordData &d2) {
    int i;
    for (i = 0; i < MAX_HASH_LENGTH_BYTES; i++) {
        if (d1.hash[i] == d2.hash[i]) {
            continue;
        } else if (d1.hash[i] > d2.hash[i]) {
            return 0;
        } else if (d1.hash[i] < d2.hash[i]) {
            return 1;
        }
    }
    // Exactly equal = return 0.
    return 0;
}


// Return true if d1 <= d2
bool hashLessThanOrEqual(const hashPasswordData &d1, const hashPasswordData &d2) {
    int i;
    for (i = 0; i < MAX_HASH_LENGTH_BYTES; i++) {
        if (d1.hash[i] == d2.hash[i]) {
            continue;
        } else if (d1.hash[i] > d2.hash[i]) {
            return 0;
        } else if (d1.hash[i] < d2.hash[i]) {
            return 1;
        }
    }
    // Exactly equal = return 1
    return 1;
}


void sortStl(std::vector<hashPasswordData> &vectorToSort) {
    std::sort(vectorToSort.begin(), vectorToSort.end(), hashPasswordDataSortPredicate);
}

void sortRadix(std::vector<hashPasswordData> &vectorToSort) {
    // Do stuff here!
}

void checkHashVectorOrdering(std::vector<hashPasswordData> &vectorToVerify) {
    std::vector<hashPasswordData>::iterator hashIterator;
    hashPasswordData currentHash, previousHash;
    int i;
    
    // Set the previous hash to the first hash.
    previousHash = *vectorToVerify.begin();
    
    for (hashIterator = vectorToVerify.begin(); hashIterator != vectorToVerify.end(); hashIterator++) {
        if (!hashLessThanOrEqual(previousHash, *hashIterator)) {
            printf("ERROR: Hash ");
            for (i = 0; i < hashLength; i++) {
                printf("%02x", previousHash.hash[i]);
            }
            printf(" ! <= ");
            for (i = 0; i < hashLength; i++) {
                printf("%02x", hashIterator->hash[i]);
            }
            printf("\n");
        }
        previousHash = *hashIterator;
    }
    
}

int main(int argc, char *argv[]) {
    
    uint32_t megsOfRam = 0;
    uint32_t elementsPerVector = 0, element, i;
    
    // Vectors of data - one for STL sort, one for Radix sort.
    std::vector<hashPasswordData> dataToSortStl;
    std::vector<hashPasswordData> dataToSortRadix;
    
    hashPasswordData hashToAdd;
    
    double StlSortTime, RadixSortTime;
    
    
    CHHiresTimer SortTimer;
    
    if (argc != 2) {
        printf("Usage: %s [megs of RAM to use]\n", argv[0]);
        exit(1);
    }
    
    megsOfRam = atoi(argv[1]);
    
    if (megsOfRam == 0) {
        printf("Error in RAM amount!  Use an integer.\n");
        exit(1);
    }
    
    printf("Using %d MB RAM total\n", (int)megsOfRam);
    
    // Elements per vector - divide by 2 so we use the right amount of RAM.
    elementsPerVector = (megsOfRam * 1024 * 1024) / (2 * sizeof(hashPasswordData));
    printf("Using %u elements per vector.\n", elementsPerVector);
    
    // Seed the random generator.
    srand(time(NULL));
    
    // Add data to the vectors
    for (element = 0; element < elementsPerVector; element++) {
        // Clear the vector prior to adding data.
        memset(&hashToAdd, 0, sizeof(hashPasswordData));
        for (i = 0; i < hashLength; i++) {
            hashToAdd.hash[i] = (uint8_t)rand();
        }
        /*
        printf("Hash %08u: ", element);
        for (i = 0; i < hashLength; i++) {
            printf("%02x", hashToAdd.hash[i]);
        }
        printf("\n");
        */
        dataToSortStl.push_back(hashToAdd);
        dataToSortRadix.push_back(hashToAdd);
    }
    
    printf("\n\n");
    printf("Starting STL sort...\n");
    SortTimer.start();
    sortStl(dataToSortStl);
    SortTimer.stop();
    printf("Verifying STL sort...\n");
    checkHashVectorOrdering(dataToSortStl);
    StlSortTime = SortTimer.getElapsedTimeInMilliSec();
    printf("STL sort complete.  Time: %f ms\n", StlSortTime);

    
    printf("\n\n");
    printf("Starting Radix sort...\n");
    SortTimer.start();
    sortRadix(dataToSortRadix);
    SortTimer.stop();
    printf("Verifying Radix sort...\n");
    checkHashVectorOrdering(dataToSortRadix);
    RadixSortTime = SortTimer.getElapsedTimeInMilliSec();
    printf("Radix sort complete.  Time: %f ms\n", RadixSortTime);
    
    // So we don't divide by zero.
    if (RadixSortTime == 0.0) {
        RadixSortTime = 1;
    }
    
    printf("\n\n");
    printf("Summary:\n");
    printf("STL sort time: %f ms\n", StlSortTime);
    printf("Radix sort time: %f ms\n", RadixSortTime);
    printf("Speedup: %0.2f\n", (StlSortTime / RadixSortTime));
}