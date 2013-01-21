/*
Cryptohaze GPU Rainbow Tables
Copyright (C) 2011  Bitweasil (http://www.cryptohaze.com/)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

/* GRTAnalyzeTable analyzes a table file for the following:
 * - Ensures the hashes are a monotonically increasing sequence
 * - Determines the bits of uniqueness needed to identify each chain
 * - Reports the number of bits needed for the password space
 *
 * This output is designed for use in converting tables to V2 format,
 * or compressing the V2 format down.
 *
 */

#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableSearchV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableSearchV2.h"
#include "GRT_Common/GRTCommon.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

// Silence output if true.
char silent = 0;

// Maximum number of bits to compare.
#define MAX_HASH_BITS 128

uint64_t hitsPerBitDifference[MAX_HASH_BITS];
FILE *analysisLog;
uint64_t totalChainCount;


void terminate_process(int) {
    printf("\n\n");
    for (int index = 0; index < MAX_HASH_BITS; index++) {
        printf("Items in %03d bits: %lu (%0.5f %%)\n", index, hitsPerBitDifference[index],
                (100.0 * (double)hitsPerBitDifference[index] / (double)totalChainCount));
        if (analysisLog != NULL) {
            fprintf(analysisLog, "Items in %03d bits: %lu (%0.5f %%)\n", index, hitsPerBitDifference[index],
                (100.0 * (double)hitsPerBitDifference[index] / (double)totalChainCount));
        }
    }
    exit(0);
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

// Return true if d1 == d2
bool hashEqual(const hashPasswordData &d1, const hashPasswordData &d2) {
    int i;
    for (i = 0; i < MAX_HASH_LENGTH_BYTES; i++) {
        if (d1.hash[i] == d2.hash[i]) {
            // Current byte is equal... continue.
            continue;
        } else {
            // Not equal.  Return 0.
            return 0;
        }
    }
    // Exactly equal = return 1
    return 1;
}

// Converts the hash into a uint64_t value
// hashOffset is used to split out the high and low values.
// Use 0 to get the high 8 bytes, 8 to get the low 8 byes.
uint64_t convertHashToUint64(const hashPasswordData &d1, int hashOffset) {
    uint64_t hashValue = 0;
    int i;

    // Convert from the character array to a uint64_t of the first 8 bytes.
    // This treats it in standard big endian style - so 0x12, 0x34 => 0x1234
    //printf("Hash: ");
    for (i = 0; i < 8; i++) {
        //printf("%02X", d1.hash[i + hashOffset]);
        hashValue |= (uint64_t)d1.hash[i + hashOffset] << (8 * (7 - i));
    }
    //printf("\n====: %08X%08X\n", hashValue >> 32, hashValue & 0xffffffff);
    return hashValue;
}


// Determines how many bits are the same between two values.
// Returns the number of bits needed to separate the values.
// Returns 0 if they are identical, as this is meaningless in this case.
unsigned char getBitsOfSameness(uint64_t v1, uint64_t v2) {
    int i;
    uint64_t difference;

    if (v1 == v2) {
        return 64;
    }

    // Determine the sameness or difference between the values.
    difference = v1 ^ v2;

    //printf("v1: %08x%08x\n", v1 >> 32, v1 & 0xffffffff);
    //printf("v2: %08x%08x\n", v2 >> 32, v2 & 0xffffffff);
    //printf("^^: %08x%08x\n", difference >> 32, difference & 0xffffffff);


    for (i = 0; i < 64; i++) {
        //printf("i: %d    %d\n", i, difference >> (63 - i));
        if (difference >> (63 - i)) {
            return i;
        }
    }
    return 64;
}

void printChainWithIndex(uint64_t index, hashPasswordData *chain) {
    int i;

    printf("Chain %lu\n", index);
    printf("Password: %s\n", chain->password);
    printf("Hash: ");
    for (i = 0; i < 16; i++) {
        printf("%02x", chain->hash[i]);
    }
    printf("\n\n");
}

int main (int argc, char *argv[]) {
    hashPasswordData currentHash, previousHash;

    GRTTableHeader *TableHeader;
    GRTTableSearch *TableSearch;

    char verbose = 0;

    uint64_t index;
    uint64_t currentHashValueLow, currentHashValueHigh, previousHashValueLow, previousHashValueHigh;
    int maxBitsSameness = 0;
    int bitsOfSameness;

    if (argc < 2) {
        printf("Usage: %s [table filename to analyze] [verbose 0:1] [opt: analysis filename]\n", argv[0]);
        exit(1);
    }

    if (argc == 2) {
        verbose = 0;
    } else if (argv[2][0] == '0') {
        verbose = 0;
    } else if (argv[2][0] == '1') {
        verbose = 1;
    } else {
        printf("Verbose must be 0 or 1!\n");
        exit(1);
    }
    
    if (argc == 4) {
        printf("Writing analysis to %s\n", argv[2]);
        analysisLog = fopen(argv[3], "w");
    } else {
        analysisLog = NULL;
    }


    // Catch Ctrl-C and handle it gracefully
    signal(SIGINT, terminate_process);

    if (getTableVersion(argv[1]) == 1) {
        TableHeader = new GRTTableHeaderV1();
        TableSearch = new GRTTableSearchV1();
    } else if (getTableVersion(argv[1]) == 2) {
        TableHeader = new GRTTableHeaderV2();
        TableSearch = new GRTTableSearchV2();
    } else {
        printf("Table version %d not supported!\n", getTableVersion(argv[1]));
        exit(1);
    }

    if (!TableHeader->readTableHeader(argv[1])) {
        // Error will be printed.  Just exit.
        exit(1);
    }

    if (verbose) {
        TableHeader->printTableHeader();
    }

    TableSearch->SetTableFilename(argv[1]);

    TableSearch->setTableHeader(TableHeader);

    // Now we have everything set up.  Start analysis.

    memset(&currentHash, 0, sizeof(hashPasswordData));
    memset(&previousHash, 0, sizeof(hashPasswordData));
    memset(hitsPerBitDifference, 0, 64 * sizeof(uint64_t));

    if (!TableSearch->getNumberChains()) {
        printf("Table appears to have no chains... eh?\n");
        exit(1);
    }

    totalChainCount = TableSearch->getNumberChains();

    // Set hash zero if it is present as the "previous hash"
    TableSearch->getChainAtIndex(0, &previousHash);
    previousHashValueHigh = convertHashToUint64(previousHash, 0);
    previousHashValueLow  = convertHashToUint64(previousHash, 8);

    TableSearch->getChainAtIndex(0, &currentHash);

    if (verbose) {
        printChainWithIndex(0, &currentHash);
    }


    for (index = 1; index < TableSearch->getNumberChains(); index++) {
        TableSearch->getChainAtIndex(index, &currentHash);

        if (verbose) {
            printChainWithIndex(index, &currentHash);
        }

        currentHashValueHigh = convertHashToUint64(currentHash, 0);
        currentHashValueLow  = convertHashToUint64(currentHash, 8);
        // Ensure previousHash <= currentHash
        if (!hashLessThanOrEqual(previousHash, currentHash)) {
            printf("\n\nERROR: Hash at position %lu is not <= hash at position %lu!\n", index, index - 1);
            exit(1);
        }

        // Determine bits of sameness.
        // If they are equal, it's the hash length.
        // Else, figure out details.
        if (hashEqual(currentHash, previousHash)) {
            // They are equal as far out as we care.
            bitsOfSameness = 0;
        } else {
            // Check the high bits.  If this works out to 64, the upper half is identical,
            // and we check the lower half.
            bitsOfSameness = getBitsOfSameness(currentHashValueHigh, previousHashValueHigh);
            if (bitsOfSameness == 64) {
                // Add the lower half bits of sameness.
                bitsOfSameness += getBitsOfSameness(currentHashValueLow, previousHashValueLow);
            }
        }

        // Add this data to the histogram-ish structure.
        hitsPerBitDifference[bitsOfSameness]++;

        // If this is a new high water mark, add it.
        // If the hashes are identical, it doesn't matter.
        if ((bitsOfSameness != MAX_HASH_BITS) && (bitsOfSameness > maxBitsSameness)) {
            maxBitsSameness = bitsOfSameness;
            //printf("previous: %08x%08x\n", previousHashValue >> 32, previousHashValue & 0xffffffff);
            //printf("current : %08x%08x\n", currentHashValue >> 32, currentHashValue & 0xffffffff);
            //printf("bits: %d\n\n", bitsOfSameness);
        }

        if ((index % 100000) == 0) {
            printf("Progress: %lu / %lu (%0.2f%%)   Max bits of sameness: %d\r",
                    index, TableSearch->getNumberChains(),
                    100.0 * ((float)index / (float)TableSearch->getNumberChains()), maxBitsSameness);
            fflush(stdout);
        }

        previousHashValueLow = currentHashValueLow;
        previousHashValueHigh = currentHashValueHigh;
        previousHash = currentHash;
    }

    printf("\n\nMax bits of sameness: %d\n", maxBitsSameness);

    terminate_process(0);
}