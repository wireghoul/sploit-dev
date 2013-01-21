#include "GRT_Common/GRTTableSearchV1.h"
#include <fcntl.h>
#include <sys/stat.h>
#include "GRT_Common/GRTTableHeaderV1.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include "GRT_Common/GRTCommon.h"

// If we are using boost memory mapped files, include them...
#if USE_BOOST_MMAP
#include <boost/iostreams/device/mapped_file.hpp>
#endif

extern char silent;

// Fix for 64-bit stat on Windows platform
// Thanks MaddGamer for finding this problem
// with big files!
#if _MSC_VER >= 1000
#define stat    _stat64
#endif


GRTTableSearchV1::GRTTableSearchV1() {
    this->maxChainAddress = 0;
    this->tableSize = 0;
    this->tableFile = 0;
    this->tableAccess = NULL;
    this->tableMemoryBase = NULL;
    this->indexFileIsLoaded = 0;
    this->tableIndexFileAccess = NULL;
    this->Table_Header = NULL;

    this->outputFile = NULL;
    this->numberChainsWritten = 0;

    this->CrackDisplay = NULL;
}
GRTTableSearchV1::~GRTTableSearchV1() {

    if (this->indexFileIsLoaded) {
        free(this->tableIndexFileAccess);
    }

    this->chainsToRegen.clear();
}


// Sets the table filename to search.
void GRTTableSearchV1::SetTableFilename(const char *newTableFilename) {
    struct stat file_status;

    // If the table header is not present, load one.
    if (!this->Table_Header) {
        this->Table_Header = new GRTTableHeaderV1();
        this->Table_Header->readTableHeader(newTableFilename);
    }

    if(stat(newTableFilename, &file_status) != 0){
        if (this->CrackDisplay) {
            delete this->CrackDisplay;
        }
        printf("Unable to stat %s\n", newTableFilename);
        exit(1);
    }

#if USE_BOOST_MMAP
    // Use boost memory mapped files to get the table loaded.
    mapped_file_params table_params;

    table_params.path = newTableFilename;

    this->boost_mapped_source.open(table_params);
    this->tableMemoryBase = (unsigned char *)this->boost_mapped_source.data();
#else
    this->tableFile = open(newTableFilename, O_RDONLY);
    if (!this->tableFile) {
        if (this->CrackDisplay) {
            delete this->CrackDisplay;
        }
        printf("Cannot open table file %s!\n", newTableFilename);
        exit(1);
    }

    // Memory map the file.  This starts at the BASE - we deal with the offsets later.
    // Windows has some weird issues with offsets.  So we deal with that here.
    this->tableMemoryBase = (unsigned char *)mmap(0, file_status.st_size, PROT_READ, MAP_SHARED, this->tableFile, 0);

    // Advise the OS that access will be sequential.
    madvise (this->tableMemoryBase, file_status.st_size, MADV_SEQUENTIAL);
#endif



    // Copy the table size over.
    this->tableSize = file_status.st_size;

    // Now do our offset work.  Table header v1 is 8192 bytes by definition.
    this->tableAccess = (tableV1DataStructure *) (this->tableMemoryBase + 8192);

    this->maxChainAddress = (this->tableSize - 8192) / sizeof(tableV1DataStructure);

    // Attempt to load the index file
    this->LoadIndexFile(newTableFilename);

}

// Give the table searcher the list of hashes it needs to find.
void GRTTableSearchV1::SetCandidateHashes(std::vector<hashData>* newCandidateHashes) {
    this->candidateHashes = newCandidateHashes;
}

// Actually searches the table.
void GRTTableSearchV1::SearchTable() {
    // Report if an index is not loaded.
    if (this->indexFileIsLoaded) {
        this->SearchWithIndex();
    } else {
        if (!silent) {
            if (this->CrackDisplay) {
                sprintf(this->statusStrings, "Using unindexed search!");
                this->CrackDisplay->addStatusLine(this->statusStrings);
            } else {
                printf("Using unindexed search!  This is much slower!\n");
            }
        }
        this->SearchWithoutIndex();
    }
}


void GRTTableSearchV1::SearchWithoutIndex() {
    unsigned char hash[16];
    uint64_t min, max;
    int64_t check;
    int64_t hashLocation;
    int result;
    int thisHashFound = 0, isDuplicate;
    long int found = 0, i = 0, chainsToRegen = 0;
    UINT4 hashId;
    struct regenChainData *regenChainData;
    //unsigned int timer, totaltimer;
    //uint64_t previousTimerHashCount = 0;
    hashData candidateHashData;
    hashPasswordData chainToRegen;




    for (hashId = 0; hashId < this->candidateHashes->size(); hashId++) {
        if (!silent && ((hashId % 500) == 0)) {
            if (this->CrackDisplay) {
                // Set the percentage done
                this->CrackDisplay->setStagePercent(100.0 *
                        (float)hashId / (float)this->candidateHashes->size());
            } else {
                printf("Step %d / %d , %ld chains found       \r", hashId, this->candidateHashes->size(),
                    found);
                fflush(stdout);
            }
        }

        // Copy hash value into 'hash' variable.
        candidateHashData = this->candidateHashes->at(hashId);
        //printf("Hash: ");
        for (i = 0; i < 16; i++) {
            hash[i] = candidateHashData.hash[i];
            //printf("%02x", hash[i]);
        }
        //printf("\n");

        min = 0;
        max = this->maxChainAddress;
        thisHashFound = 0;

        while ((max - min) > 1) {
            check = (min + max) / 2;
            result = memcmp(&this->tableAccess[check].hash, &hash, 16);
            // Found
            if (result == 0) {
                thisHashFound = 1;
                hashLocation = check;
                goto end;
            }
            // check > hash
            else if (result > 0) {
                max = check;
            }
            else {
                min = check;
            }
        }
        // Check the previous one to deal with integer rounding issues and the 1st chain
        check--;
        if (check < 0) {check = 0;}
        result = memcmp(&this->tableAccess[check].hash, &hash, 16);
        // Found
        if (result == 0) {
            thisHashFound = 1;
            hashLocation = check;
            goto end;
        }

        end:;
        //printf("Quitting at location %d\n", check);
        if (thisHashFound) {
            // Back up to the start of the sequence or the beginning of the table
            while ((memcmp(&this->tableAccess[hashLocation].hash, &hash, 16) == 0)) {
                hashLocation--;
                // If the hash location is -1, we are at the beginning of the table.
                if (hashLocation < 0) {
                    break;
                }
            }
            hashLocation++;
            found++;
            while (memcmp(&this->tableAccess[hashLocation].hash, &hash, 16) == 0) {
                // Check to see if this hash is already present in the output.
                // This is somewhat inefficient, but a lot faster than regenning a chain twice.
                isDuplicate = 0;

                if (!isDuplicate) {
                    //printf("Found at %ld\n", hashLocation);
                    chainsToRegen++;

                    for (i = 0; i < 16; i++) {
                        chainToRegen.password[i] = this->tableAccess[hashLocation].password[i];
                        //regenChainData->chains[regenChainData->NumberOfChains * 16 + i] = fileMap[hashLocation].password[i];
                    }
                    this->chainsToRegen.push_back(chainToRegen);
                }

                hashLocation++;
            }
        }
    }

    if (!silent) {
        if (this->CrackDisplay) {
            sprintf(this->statusStrings, "Chains found: %lu", this->chainsToRegen.size());
            this->CrackDisplay->addStatusLine(this->statusStrings);
        } else {
            printf("\n\n");
            printf("Total chains found to regen before merging: %d\n", this->chainsToRegen.size());
       }
    }
    sort(this->chainsToRegen.begin(), this->chainsToRegen.end(), passwordDataSortPredicate);
    this->chainsToRegen.erase(
        unique( this->chainsToRegen.begin(), this->chainsToRegen.end(), passwordDataUniquePredicate ),
        this->chainsToRegen.end() );

    if (!silent) {
        if (this->CrackDisplay) {
            sprintf(this->statusStrings, "Merged chains: %lu", this->chainsToRegen.size());
            this->CrackDisplay->addStatusLine(this->statusStrings);
        } else {
            printf("Total chains found to regen after merging: %d\n", this->chainsToRegen.size());
       }
    }

}

// Return a list of the chains to regenerate
// This should probably get deleted after use :)
std::vector<hashPasswordData>* GRTTableSearchV1::getChainsToRegen() {
    std::vector<hashPasswordData>* returnChains;

    returnChains = new std::vector<hashPasswordData>();
    
    *returnChains = this->chainsToRegen;

    return returnChains;
}

void GRTTableSearchV1::setTableHeader(GRTTableHeader * newTableHeader) {
    this->Table_Header = (GRTTableHeaderV1 *)newTableHeader;
}

void GRTTableSearchV1::getChainAtIndex(uint64_t index, struct hashPasswordData *chainInfo) {
    // Copy the hash.
    memset(chainInfo->hash, 0, sizeof(chainInfo->hash));
    memcpy(chainInfo->hash, this->tableAccess[index].hash, 16);

    // Copy the chain start password
    memset(chainInfo->password, 0, sizeof(chainInfo->password));
    memcpy(chainInfo->password, this->tableAccess[index].password, 16);

}

uint64_t GRTTableSearchV1::getNumberChains() {
    return this->maxChainAddress;
}

int GRTTableSearchV1::LoadIndexFile(const char *newTableFilename) {
    FILE *indexFile;
    int i, j;

    // Buffer for the possible index filename
    char filenameBuffer[2000];
    struct stat file_status;

    sprintf(filenameBuffer, "%s.idx", newTableFilename);

    // Check to make sure the file is present
    if(stat(filenameBuffer, &file_status) != 0){
        if (!silent) {
            if (this->CrackDisplay) {
                sprintf(this->statusStrings, "Cannot find index file.");
                this->CrackDisplay->addStatusLine(this->statusStrings);
            } else {
                printf("Cannot find index file %s\n", filenameBuffer);
            }
        }
        return 0;
    }

    indexFile = fopen(filenameBuffer, "rb");

    if (!indexFile) {
        printf("Cannot open index file %s!\n", filenameBuffer);
        return NULL;
    }

    this->TotalIndexes = file_status.st_size / sizeof(struct indexFile);

    if (!silent) {
        if (this->CrackDisplay) {
            sprintf(this->statusStrings, "Loaded %d indexes", this->TotalIndexes);
            this->CrackDisplay->addStatusLine(this->statusStrings);
        } else {
            printf("Loaded %d indexes\n", this->TotalIndexes);
        }
    }

    this->tableIndexFileAccess = (struct indexFile*)malloc(this->TotalIndexes *
            sizeof(struct indexFile));

    if (!this->tableIndexFileAccess) {
        if (this->CrackDisplay) {
            sprintf(this->statusStrings, "Malloc fail: %ld b\n", (this->TotalIndexes * sizeof(struct indexFile)));
            this->CrackDisplay->addStatusLine(this->statusStrings);
        } else {
            printf("Malloc fail: %ld b\n", (this->TotalIndexes * sizeof(struct indexFile)));
        }
        return NULL;
    }
    //printf("malloc'd %ld bytes for index file!\n", (TotalIndexes * sizeof(struct indexFile)));

    if (fread(this->tableIndexFileAccess, sizeof(struct indexFile), this->TotalIndexes, indexFile) != this->TotalIndexes) {
        if (!silent) {
            //printf("Index file read failed.\n");
        }
        return NULL;
    }
    fclose(indexFile);
    //printf("Index file loaded successfully.\n");
    this->indexFileIsLoaded = 1;

    // Get index file bits


    // Calculate how many bits we have.
    // Probabalistically calculate it... there is a CHANCE
    // this is wrong, but it's basically zero.
    this->indexFileBits = 0;
    uint32_t HashIndex;
    for (i = 0; i < this->TotalIndexes; i++) {
        HashIndex = this->tableIndexFileAccess[i].Index;
        for (j = 0; j < 32; j++) {
            if ((HashIndex >> j) & 0x01) {
                if ((uint32_t)(32 - j) > this->indexFileBits) {
                    this->indexFileBits = (uint32_t)(32 - j);
                }
                break;
            }
        }
    }
    if (!silent) {
        if (this->CrackDisplay) {
            sprintf(this->statusStrings, "Bits of index: %d", this->indexFileBits);
            this->CrackDisplay->addStatusLine(this->statusStrings);
        } else {
            printf("Index file bits to index: %d\n", this->indexFileBits);
        }
    }
    return 1;
}


void GRTTableSearchV1::SearchWithIndex() {
    unsigned char hash[16];
    uint64_t min, max, check = 0;
    int64_t hashLocation;
    int thisHashFound = 0, isDuplicate;
    long int found = 0, i = 0, chainsToRegen = 0;
    UINT4 hashId;
    struct regenChainData *regenChainData;
    uint64_t StartOffset, EndOffset, CurrentOffset;

    uint32_t Mask, MaskedData;
    
    hashData candidateHashData;
    hashPasswordData chainToRegen;

    memset(&candidateHashData, 0, sizeof(hashData));
    memset(&chainToRegen, 0, sizeof(hashPasswordData));



    Mask = 0x00000000;
    for (i = 0; i < this->indexFileBits; i++) {
        // Add the needed bits.
        Mask |= (1 << (31 - i));
    }

    // Loop through the hashes.
    if (!silent) {
        if (!this->CrackDisplay) {
            printf("\n\n");
        }
    }


    for (hashId = 0; hashId < this->candidateHashes->size(); hashId++) {
        if (!silent && ((hashId % 500) == 0)) {
            if (this->CrackDisplay) {
                // Set the percentage done
                this->CrackDisplay->setStagePercent(100.0 *
                        (float)hashId / (float)this->candidateHashes->size());
            } else {
                printf("\rStep %d / %d (%0.2f%%) %ld chains found       ",
                    hashId, this->candidateHashes->size(),
                    (100.0 * (float)hashId / (float)this->candidateHashes->size()), found);
                fflush(stdout);
            }
        }


        // Copy hash value into 'hash' variable.
        candidateHashData = this->candidateHashes->at(hashId);
        //printf("Hash: ");
        for (i = 0; i < 16; i++) {
            hash[i] = candidateHashData.hash[i];
            //printf("%02x", hash[i]);
        }

        MaskedData = hash[0] << 24 |
                     hash[1] << 16 |
                     hash[2] << 8  |
                     hash[3];

        MaskedData &= Mask;
        //printf("Masked data: %08X\n", MaskedData);

        // Binary search the index.
        min = 0;
        max = this->TotalIndexes;


        while (max - min > 1) {
            check = (min + max) / 2;
            if (this->tableIndexFileAccess[check].Index == MaskedData) {
                goto end;
            } else if (this->tableIndexFileAccess[check].Index > MaskedData){
                max = check;
            } else {
                min = check;
            }
        }

        // Only back up if check is 1 or greater.
        if (check > 0) {
            check--;
        }

    while ((check < this->TotalIndexes) && (this->tableIndexFileAccess[check].Index < MaskedData)) {
            check++;
        }
        end:;

        // Make sure we didn't wrap over the end.
        if (check >= this->TotalIndexes) {
            check = this->TotalIndexes - 1;
        }

        thisHashFound = 0;
        StartOffset = this->tableIndexFileAccess[check].Offset;
        if (check < (this->TotalIndexes - 1)) {
            EndOffset = this->tableIndexFileAccess[check + 1].Offset;
        } else {
            EndOffset = this->maxChainAddress;
        }

        for (CurrentOffset = StartOffset; CurrentOffset <= EndOffset; CurrentOffset++) {
            if (memcmp(this->tableAccess[CurrentOffset].hash, &hash, 16) == 0) {
                thisHashFound = 1;
                //printf("Found hash!\n");
                hashLocation = CurrentOffset;
            }
        }


        if (thisHashFound) {
            // Back up to the start of the sequence or the beginning of the table
            while ((memcmp(this->tableAccess[hashLocation].hash, &hash, 16) == 0)) {
                hashLocation--;
                // If the hash location is -1, we are at the beginning of the table.
                if (hashLocation < 0) {
                    break;
                }
            }
            hashLocation++;
            found++;
            while (memcmp(this->tableAccess[hashLocation].hash, &hash, 16) == 0) {
                isDuplicate = 0;

                if (!isDuplicate) {
                    //printf("Found at %ld\n", hashLocation);
                    chainsToRegen++;

                    for (i = 0; i < 16; i++) {
                        chainToRegen.password[i] = this->tableAccess[hashLocation].password[i];
                        //regenChainData->chains[regenChainData->NumberOfChains * 16 + i] = fileMap[hashLocation].password[i];
                    }
                    this->chainsToRegen.push_back(chainToRegen);
                }

                hashLocation++;
            }
        }
    }
    if (!silent) {
        if (this->CrackDisplay) {
            sprintf(this->statusStrings, "Chains found: %lu", this->chainsToRegen.size());
            this->CrackDisplay->addStatusLine(this->statusStrings);
        } else {
            printf("\n\n");
            printf("Total chains found to regen before merging: %d\n", this->chainsToRegen.size());
       }
    }
    sort(this->chainsToRegen.begin(), this->chainsToRegen.end(), passwordDataSortPredicate);
    this->chainsToRegen.erase(
        unique( this->chainsToRegen.begin(), this->chainsToRegen.end(), passwordDataUniquePredicate ),
        this->chainsToRegen.end() );

    if (!silent) {
        if (this->CrackDisplay) {
            sprintf(this->statusStrings, "Merged chains: %lu", this->chainsToRegen.size());
            this->CrackDisplay->addStatusLine(this->statusStrings);
        } else {
            printf("Total chains found to regen after merging: %d\n", this->chainsToRegen.size());
       }
    }
}


// Open a file to output data to
int GRTTableSearchV1::openOutputFile(char *outputFilename) {

    if (!this->Table_Header) {
        printf("Cannot open output file without table header!\n");
        exit(1);
    }

    this->outputFile = fopen(outputFilename, "wb");
    if (!this->outputFile) {
        printf("Cannot open output file %s!\n", outputFilename);
        exit(1);
    }
    // Write the header to the beginning of the file
    this->Table_Header->writeTableHeader(this->outputFile);
    return 1;
}
int GRTTableSearchV1::closeOutputFile() {
    // Write the new table header and close the file.
    this->Table_Header->setNumberChains(this->numberChainsWritten);
    this->Table_Header->writeTableHeader(this->outputFile);
    fclose(this->outputFile);
	return 1;
}

// Write the chain to the opened file.
int GRTTableSearchV1::writeChain(hashPasswordData* hashToWrite) {
    if (!fwrite(hashToWrite->hash, 16, 1, this->outputFile) ||
        !fwrite(hashToWrite->password, 16, 1, this->outputFile)) {
        return 0;
    }
    this->numberChainsWritten++;
    return 1;
}

int GRTTableSearchV1::getBitsInHash() {
    return 128;
}

int GRTTableSearchV1::getBitsInPassword() {
    return 128;
}

void GRTTableSearchV1::setCrackDisplay(GRTCrackDisplay* newCrackDisplay) {
    this->CrackDisplay = newCrackDisplay;
}

