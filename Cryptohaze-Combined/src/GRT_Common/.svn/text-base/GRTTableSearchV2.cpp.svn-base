#include "GRT_Common/GRTTableSearchV1.h"
#include <fcntl.h>
#include <sys/stat.h>
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableSearchV2.h"
#include "GRT_Common/GRTCommon.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <errno.h>

// Hint threads, for now, are pthreads only.
// todo: Add boost thread support to hint threads.
#if !USE_BOOST_THREADS
#include <pthread.h>
#endif


#include "CH_Common/Timer.h"

extern char silent;

// Fix for 64-bit stat on Windows platform
// Thanks MaddGamer for finding this problem
// with big files!
#if _MSC_VER >= 1000
#define stat    _stat64
#endif



// Convert the password into a charset offset
uint64_t convertSingleCharsetToOffset(char *password, char *charset) {
    int i;
    uint64_t offset;

    int passwordLength = strlen(password);
    int charsetLength = strlen(charset);

    //printf("CharsetLength: %d\n", charsetLength);

    // Offset into the charset
    unsigned char lookupTable[256];
    memset(lookupTable, 0, 256);

    for (i = 0; i < charsetLength; i++) {
        lookupTable[charset[i]] = i;
    }

    offset = 0;

    for (i = 0; i < passwordLength ; i++) {
        //printf("Lookup for pos %d: %d\n", i, lookupTable[password[i]]);
        offset *= charsetLength;
        offset += lookupTable[password[i]];
        //printf("Offset now: %lu\n", offset);
    }
    return offset;
}

// Convert an offset back into a password
void convertOffsetToSingleCharset(uint64_t offset, char *charset, char *password, int passwordLength) {
    int charsetLength = strlen(charset);
    int i;

    for (i = passwordLength - 1; i >= 0; i--) {
        //printf("offset: %lu\n", offset);
        //printf("Char: %c\n", charset[offset % charsetLength]);
        password[i] = charset[offset % charsetLength];
        offset /= charsetLength;
    }
    //printf("Password: %s\n", password);
}

// Copies the number of bits of hash into the chain data
// They end up in the same order.
void copyHashAndOffsetIntoChain(unsigned char *chainData, hashPasswordData* hashData,
        uint64_t offset, int bitsInHash, int bitsInOffset) {

    int bitsRemaining, i;
    uint8_t maskedValue, mask;
    uint8_t offsetMSB;
    int offsetCopyStart;

    // Sanity check to ensure the bit counts are valid.
    if ((bitsInHash + bitsInOffset) % 8) {
        printf("ERROR: Bits in hash + bits in offset MUST be byte aligned!\n");
        exit(1);
    }

    // Copy the whole bytes into the output
    memcpy(chainData, hashData->hash, (bitsInHash / 8));

    // Set the offset copy start location
    offsetCopyStart = (bitsInHash / 8);

    // If there are remaining bits of hash to copy, copy them.
    if (bitsInHash % 8) {
        // Figure out how many bits remain to copy
        bitsRemaining = bitsInHash % 8;

        // Get the last byte for masking
        maskedValue = hashData->hash[bitsInHash / 8];

        // Create a mask for the high bits.
        mask = (uint8_t) 0;
        for (i = 0; i < bitsRemaining; i++) {
            mask |= (uint8_t) 1 << (7 - i);
        }
        maskedValue &= mask;

        chainData[bitsInHash / 8] = maskedValue;

        //printf("Original last byte: 0x%02x\n", hashData->hash[bitsInHash / 8]);
        //printf("Masked   last byte: 0x%02x\n", chainData[bitsInHash / 8]);

        // If we copied bits of hash into the intermediate byte, we need to copy
        // bits of offset in.

        // Get the MSB of hash
        offsetMSB = offset >> ((bitsInOffset / 8) * 8);
        //printf("Offset MSB: %02x\n", offsetMSB);

        maskedValue |= offsetMSB;

        //printf("Final intermediate byte: %02x\n", maskedValue);

        chainData[bitsInHash / 8] = maskedValue;
        // As we have an intermediate byte, the offsetCopyStart must be incremented.
        offsetCopyStart++;
    }

    // Now copy the remaining bytes of password in.
    //printf("Offset: 0x%08x%08x\n", (uint32_t)(offset >> 32), (uint32_t)(offset & 0xffffffff));
    //printf("Offset copy start: %d\n", offsetCopyStart);
    memcpy(&chainData[offsetCopyStart], &offset, (bitsInOffset / 8));
/*
    printf("Final chain: ");
    for (i = 0; i < ((bitsInHash + bitsInOffset) / 8); i++) {
        printf("%02x", chainData[i]);
    }
    printf("\n");
*/
}

// This undoes what the copyHashAndOffsetIntoChain funtion does.
// Calling is almost the same, but offset is a pointer now
void getHashAndOffsetFromChain(unsigned char *chainData, hashPasswordData* hashData,
        uint64_t *offset, int bitsInHash, int bitsInOffset) {

    int bitsRemaining, i;
    uint8_t intermediateByte, mask;
    uint8_t offsetMSB;
    int offsetCopyStart;

    // SERIOUSLY BITWEASIL!  MEMSET BEFORE COPYING BYTES IN!!!!!
    memset(offset, 0, sizeof(uint64_t));

    // Sanity check to ensure the bit counts are valid.
    if ((bitsInHash + bitsInOffset) % 8) {
        printf("ERROR: Bits in hash + bits in offset MUST be byte aligned!\n");
        exit(1);
    }

    // Clear out the chain data structure
    memset(hashData, 0, sizeof(struct hashPasswordData));

    // Copy the whole bytes from the chain into the hash
    memcpy(hashData->hash, chainData, (bitsInHash / 8));

    // Set the offset copy start location
    offsetCopyStart = (bitsInHash / 8);

        // If there are remaining bits of hash to copy, copy them.
    if (bitsInHash % 8) {
        // Figure out how many bits remain to copy
        bitsRemaining = bitsInHash % 8;

        // Get the intermediate byte
        intermediateByte = chainData[offsetCopyStart];

        // Create a mask for the high bits.
        mask = (uint8_t) 0;
        for (i = 0; i < bitsRemaining; i++) {
            mask |= (uint8_t) 1 << (7 - i);
        }

        // Copy the bits of the hash into the hash value
        hashData->hash[offsetCopyStart] = intermediateByte & mask;

        // If we copied bits of hash opt of the intermediate byte, we need to copy
        // bits of offset out.

        // Invert the mask to get the mask bits for the offset MSB
        mask = ~mask;

        // Get the MSB of hash
        *offset = (uint64_t)(intermediateByte & mask) << ((bitsInOffset / 8) * 8);

        // As we have an intermediate byte, the offsetCopyStart must be incremented.
        offsetCopyStart++;
    }

    // Now copy the remaining bytes of password in.
    memcpy(offset, &chainData[offsetCopyStart], (bitsInOffset / 8));

}

// Memcmp, just that checks a given number of bits instead of only bytes
// This checks the MSB-side of the last byte if needed.
int memcmpBits(unsigned char *val1, unsigned char *val2, int bitsToCompare) {
    int result;
    uint8_t mask;
    uint8_t byte1, byte2;
    int i;

    // Compare the full bytes
    result = memcmp(val1, val2, (bitsToCompare / 8));

    if (result > 0) {
        // Positive result - return positive.
        //printf("memcmp returning positive.\n");
        return 1;
    } else if (result < 0) {
        // Negative result - return negative.
        //printf("memcmp returning negative.\n");
        return -1;
    } else {
        // Bytes matched - check bits.
        if ((bitsToCompare % 8) == 0) {
            // If there are no more bits to compare, the two are equal
            //printf("memcmp returning equal\n");
            return 0;
        }
        //printf("memcmpBits testing more values...\n");
        // There are more bits to compare, we should check them!
        mask = 0;
        for (i = 0; i < (bitsToCompare % 8); i++) {
            mask |= (1 << (7 - i));
        }
        //printf("memcmp mask: %02x\n", mask);
        // Mask off the desired bits of the two values and compare
        byte1 = (val1[(bitsToCompare / 8)] & mask);
        byte2 = (val2[(bitsToCompare / 8)] & mask);

        //printf("Byte 1: %02x\n", byte1);
        //printf("Byte 2: %02x\n", byte2);

        if (byte1 > byte2) {
            // val1 > val2, return positive.
            //printf("memcmp returning positive\n");
            return 1;
        } else if (byte1 < byte2) {
            // val1 < val2, return negative
            //printf("memcmp returning negative.\n");
            return -1;
        } else {
            // They must be equal.  Return 0.
            //printf("memcmp returning equal\n");
            return 0;
        }
    }
    // Should NEVER get here.
    printf("MEMCMP SHOULD NOT BE HERE!\n");
}

GRTTableSearchV2::GRTTableSearchV2() {
    this->maxChainAddress = 0;
    this->tableSize = 0;
    this->tableFile = 0;
    this->tableAccess = NULL;
    this->tableMemoryBase = NULL;
    this->indexFileIsLoaded = 0;
    this->indexFileIsMemoryMapped = 0;
    this->tableIndexFileAccess = NULL;
    this->Table_Header = NULL;

    this->outputFile = NULL;
    this->numberChainsWritten = 0;

    this->bitsInHash = 0;
    this->bitsInPassword = 0;

    this->charset = NULL;
    this->currentCharset = NULL;
    this->charsetLength = NULL;

    this->CrackDisplay = NULL;

    this->prefetchThreadCount = 0;
}

GRTTableSearchV2::~GRTTableSearchV2() {
    // Free the charset if allocated
    int i;
    
    if (this->charset) {
        // Ugh.  Got the charset like this... I really need to fix this.
        for (i = 0; i < 16; i++) {
            delete[] charset[i];
        }
        delete[] charset;
    }

    if (this->indexFileIsLoaded && !this->indexFileIsMemoryMapped) {
        free(this->tableIndexFileAccess);
    }

    this->chainsToRegen.clear();

    // Unmap files as needed.
#if USE_BOOST_MMAP
    if (this->tableMemoryBase) {
        this->boost_mapped_source.close();
        this->tableMemoryBase = NULL;
    }
    
    if (this->indexFileIsMemoryMapped && this->indexFileIsLoaded) {
        this->boost_mapped_index_source.close();
    }
    
#else
    if (this->tableFile) {
        //munmap(this->tableMemoryBase, this->tableSize);
        close(this->tableFile);
        this->tableFile = 0;
    }
    
    if (this->indexFileIsLoaded && this->indexFileIsMemoryMapped) {
        //munmap(this->tableIndexFileAccess, this->TotalIndexes * sizeof(indexFile));
    }

#endif

}

// Sets the table filename to search.
void GRTTableSearchV2::SetTableFilename(const char *newTableFilename) {
    struct stat file_status;

    if(stat(newTableFilename, &file_status) != 0){
        if (this->CrackDisplay) {
            delete this->CrackDisplay;
        }
        printf("Unable to stat %s\n", newTableFilename);
        exit(1);
    }

    // If we do not have a table header, read it.
    if (!this->Table_Header) {
        this->Table_Header = new GRTTableHeaderV2();
        this->Table_Header->readTableHeader(newTableFilename);
        this->bitsInHash = this->Table_Header->getBitsInHash();
        this->bitsInPassword = this->Table_Header->getBitsInPassword();
        this->bytesInChain = (this->bitsInHash + this->bitsInPassword) / 8;
    }
    
    //this->Table_Header->printTableHeader();

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
    this->tableAccess = (unsigned char *) (this->tableMemoryBase + 8192);

    this->maxChainAddress = (this->tableSize - 8192) /
            ((this->Table_Header->getBitsInHash() + this->Table_Header->getBitsInPassword()) / 8);

    // Attempt to load the index file
    this->LoadIndexFile(newTableFilename);
}

// Give the table searcher the list of hashes it needs to find.
void GRTTableSearchV2::SetCandidateHashes(std::vector<hashData>* newCandidateHashes) {
    this->candidateHashes = newCandidateHashes;
}

// Actually searches the table.
void GRTTableSearchV2::SearchTable() {

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


void GRTTableSearchV2::SearchWithoutIndex() {
    
    unsigned char hash[32];
    uint64_t min, max;
    int64_t check;
    int64_t hashLocation;
    int result;
    int thisHashFound = 0, isDuplicate;
    long int found = 0, i = 0, chainsToRegen = 0;
    UINT4 hashId;
    struct regenChainData *regenChainData;
    hashData candidateHashData;
    hashPasswordData chainToRegen;

    int bytesOfHash;

    // Determine how many full bytes we can use to compare.
    bytesOfHash = this->bitsInHash / 8;

    // Loop through the hashes.

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
        //printf("\n\nHash: ");
        for (i = 0; i < (bytesOfHash + 1); i++) {
            hash[i] = candidateHashData.hash[i];
            //printf("%02x", hash[i]);
        }
        //printf("\n");

        min = 0;
        max = this->maxChainAddress;
        thisHashFound = 0;
        check = 0;

        while ((max - min) > 1) {
            check = (min + max) / 2;
            result = memcmpBits(&this->tableAccess[check * this->bytesInChain], (unsigned char *)&hash, this->bitsInHash);
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
        result = memcmpBits(&this->tableAccess[check * this->bytesInChain], (unsigned char *)&hash, this->bitsInHash);
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
            while ((memcmpBits(&this->tableAccess[hashLocation * this->bytesInChain], (unsigned char *)&hash, this->bitsInHash) == 0)) {
                hashLocation--;
                // If the hash location is -1, we are at the beginning of the table.
                if (hashLocation < 0) {
                    break;
                }
            }
            hashLocation++;
            found++;
            while (memcmpBits(&this->tableAccess[hashLocation * this->bytesInChain], (unsigned char *)&hash, this->bitsInHash) == 0) {
                // Check to see if this hash is already present in the output.
                // This is somewhat inefficient, but a lot faster than regenning a chain twice.
                isDuplicate = 0;

                if (!isDuplicate) {
                    //printf("Found at %ld\n", hashLocation);
                    chainsToRegen++;

                    this->getChainAtIndex(hashLocation, &chainToRegen);

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
std::vector<hashPasswordData>* GRTTableSearchV2::getChainsToRegen() {
    std::vector<hashPasswordData>* returnChains;

    returnChains = new std::vector<hashPasswordData>();

    *returnChains = this->chainsToRegen;

    return returnChains;
}

void GRTTableSearchV2::setTableHeader(GRTTableHeader * newTableHeader) {
    this->Table_Header = (GRTTableHeaderV2 *)newTableHeader;

    this->bitsInPassword = this->Table_Header->getBitsInPassword();
    this->bitsInHash = this->Table_Header->getBitsInHash();
    this->bytesInChain = (this->bitsInHash + this->bitsInPassword) / 8;
    
  
    if ((this->bitsInHash + this->bitsInPassword) % 8) {
        if (this->CrackDisplay) {
            delete this->CrackDisplay;
        }
        printf("ERROR: Must set bitsInHash + bitsInPassword to a multiple of 8 (byte aligned)!\n");
        printf("bitsInhash: %d\n", this->bitsInHash);
        printf("bitsInPassword: %d\n", this->bitsInPassword);

        // Be useful.
        int possibleNumberOfBits = 0;
        // Calculate lower byte count
        possibleNumberOfBits = (((this->bitsInHash + this->bitsInPassword) / 8) * 8) - this->bitsInPassword;
        printf("Try %d or %d for bits in hash.\n", possibleNumberOfBits, possibleNumberOfBits + 8);

        exit(1);
    }

    //printf("Set bits in pw to %d\n", this->bitsInPassword);
    //printf("hash mask: %08x%08x\n", (uint32_t)(this->hashMask >> 32), (uint32_t)(this->hashMask & 0xffffffff));
}

void GRTTableSearchV2::getChainAtIndex(uint64_t index, struct hashPasswordData *chainInfo) {

    unsigned char *dataToRead;

    uint64_t offset;
    uint64_t addressToRead;
    uint64_t readIndex;

    // Allocate data to write based on the # of bits
    dataToRead = new unsigned char[this->bytesInChain];

    memset(chainInfo, 0, sizeof(struct hashPasswordData));



    // If the charset is not instantiated, load it up.
    if (!this->charset) {
        // Get the charset from the table header, then find the first row.
        // Right now, we do not handle per-position charsets.
        this->charset = this->Table_Header->getCharset();

        // Get the single charset being used and determine it's length
        this->currentCharset = charset[0];
        this->charsetLength = strlen(currentCharset);
    }


    // Determine what character we start at.
    addressToRead = index * ((this->bitsInHash + this->bitsInPassword) / 8);

    //printf("Reading chain %lu\n", index);
    //printf("Start offset: %lu\n", addressToRead);


    for (readIndex = 0;
            readIndex < ((this->bitsInHash + this->bitsInPassword) / 8);
            readIndex++) {
        dataToRead[readIndex] = this->tableAccess[addressToRead + readIndex];
        //printf("Read %02x\n", dataToRead[readIndex]);
    }

    // Data is in the buffer.
    getHashAndOffsetFromChain(dataToRead, chainInfo, &offset, this->bitsInHash, this->bitsInPassword);
    convertOffsetToSingleCharset(offset, this->currentCharset, (char *)chainInfo->password, this->Table_Header->getPasswordLength());
    
    // Should be good to go.

    delete[] dataToRead;
}

uint64_t GRTTableSearchV2::getNumberChains() {
    return this->maxChainAddress;
}

int GRTTableSearchV2::LoadIndexFile(const char *newTableFilename) {
    FILE *indexFile;
    int i, j;

    //return this->LoadIndexFilemMapped(newTableFilename);
    
    
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
        return 0;
    }

    this->TotalIndexes = file_status.st_size / sizeof(struct indexFile);
    
    if (!silent) {
        if (this->CrackDisplay) {
            sprintf(this->statusStrings, "Loading %d indexes", this->TotalIndexes);
            this->CrackDisplay->addStatusLine(this->statusStrings);
        } else {
            printf("Loading %d indexes\n", this->TotalIndexes);
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
        //printf("Index file read failed.\n");
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

int GRTTableSearchV2::LoadIndexFilemMapped(const char *newTableFilename) {
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

#if USE_BOOST_MMAP
    // Use boost memory mapped files to get the table loaded.
    mapped_file_params table_params;

    table_params.path = filenameBuffer;

    this->boost_mapped_index_source.open(table_params);
    this->tableIndexFileAccess = (indexFile *)this->boost_mapped_index_source.data();
#else
    this->indexFileId = open(filenameBuffer, O_RDONLY);
    if (!this->indexFileId) {
        return NULL;
    }

    this->tableIndexFileAccess = (struct indexFile *)mmap(0, file_status.st_size, PROT_READ, MAP_SHARED, this->indexFileId, 0);

    // Advise the OS that access will be sequential.
    madvise (this->tableIndexFileAccess, file_status.st_size, MADV_SEQUENTIAL);
#endif
    
    this->TotalIndexes = file_status.st_size / sizeof(struct indexFile);
    
    if (!silent) {
        if (this->CrackDisplay) {
            sprintf(this->statusStrings, "Loading %d indexes", this->TotalIndexes);
            this->CrackDisplay->addStatusLine(this->statusStrings);
        } else {
            printf("Loading %d indexes\n", this->TotalIndexes);
        }
    }


    
    // Rip the file into memory
    uint64_t sum = 0;
    for (uint64_t i = 0; i < this->TotalIndexes; i++) {
        sum += this->tableIndexFileAccess[i].Index;
    }
    this->indexFileIsLoaded = (char)sum;

    this->indexFileIsLoaded = 1;
    this->indexFileIsMemoryMapped = 1;
    
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

void GRTTableSearchV2::SearchWithIndex() {
    unsigned char hash[16];
    uint64_t min = 0, max = 0, check = 0;
    int64_t hashLocation = 0;
    int thisHashFound = 0, isDuplicate;
    long int found = 0, i = 0, chainsToRegen = 0;
    UINT4 hashId;
    struct regenChainData *regenChainData;
    uint64_t StartOffset, EndOffset, CurrentOffset;

    uint32_t Mask, MaskedData;
    
    // Timing variables for calculating search rate
    uint64_t startTime;
    double searchRate;

    Timer elapsedTime;
    
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

/*
    // Hinting...

    for (hashId = 0; hashId < this->candidateHashes->size(); hashId++) {
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
        


        // Copy hash value into 'hash' variable.
        candidateHashData = this->candidateHashes->at(hashId);
        //printf("Hash: ");
        for (i = 0; i < 16; i++) {
            hash[i] = candidateHashData.hash[i];
            //printf("%02x", hash[i]);
        }
        //printf("\n");

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
                goto hint_end;
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
        hint_end:;

        // Make sure we didn't wrap over the end.
        if (check >= this->TotalIndexes) {
            check = this->TotalIndexes - 1;
        }

        StartOffset = this->tableIndexFileAccess[check].Offset;
        if (check < (this->TotalIndexes - 1)) {
            EndOffset = this->tableIndexFileAccess[check + 1].Offset;
        } else {
            EndOffset = this->maxChainAddress;
        }

        // HINT!

        int madvisereturn;
        madvisereturn = madvise((void *)((uint64_t)&this->tableAccess[StartOffset * this->bytesInChain] & 0xfffffffffffff000),
        (uint32_t)(&this->tableAccess[EndOffset * this->bytesInChain] - &this->tableAccess[StartOffset * this->bytesInChain]) + 0x1000,
        MADV_WILLNEED);

        if (madvisereturn) {
            printf("madvise error %s!\n",  strerror( errno ));
        }
    }
*/
    // Get the start of the search process
    startTime = time(NULL);
    elapsedTime.start();


#if !USE_BOOST_THREADS
    int numberOfPrefetchThreads = this->prefetchThreadCount;
    pthread_t hintThreadStructure[MAX_PREFETCH_THREADS];

    if (!silent) {
        if (this->CrackDisplay) {
            sprintf(this->statusStrings, "Prefetch threads: %d", numberOfPrefetchThreads);
            this->CrackDisplay->addStatusLine(this->statusStrings);
        } else {
            printf(this->statusStrings, "Prefetch threads: %d", numberOfPrefetchThreads);
       }
    }

// Disable barrier on OS X as it's not supported
#if USE_PREFETCH_BARRIER
    pthread_barrier_t hintBarrier;
    pthread_barrier_init(&hintBarrier, NULL, MAX_PREFETCH_THREADS);
#endif
    

    hintThreadData threadData[MAX_PREFETCH_THREADS];

    for (int thread = 0; thread < numberOfPrefetchThreads; thread++) {
        threadData[thread].Mask = Mask;
        threadData[thread].candidateHashes = this->candidateHashes;
        threadData[thread].stride = numberOfPrefetchThreads;
        threadData[thread].TotalIndexes = this->TotalIndexes;
        threadData[thread].tableIndexFileAccess = this->tableIndexFileAccess;
        threadData[thread].maxChainAddress = this->maxChainAddress;
        threadData[thread].tableAccess = this->tableAccess;
        threadData[thread].bytesInChain = this->bytesInChain;
#if USE_PREFETCH_BARRIER
        threadData[thread].hintBarrier = &hintBarrier;
#endif
        threadData[thread].threadId = thread;
    }


    for (int thread = 0; thread < numberOfPrefetchThreads; thread++) {
        pthread_create(&hintThreadStructure[thread], NULL, hintThread, (void *)&threadData[thread] );
    }
#endif

    for (hashId = 0; hashId < this->candidateHashes->size(); hashId++) {
        if (!silent && ((hashId % 500) == 0)) {
            // Determine the current time of searching in a cross-platform way.
            
            if (this->CrackDisplay) {
                // Set the percentage done
                this->CrackDisplay->setStagePercent(100.0 *
                        (float)hashId / (float)this->candidateHashes->size());

                //searchRate = (double)hashId / (double)(time(NULL) - startTime);
                searchRate = (double)hashId / elapsedTime.stop();
                this->CrackDisplay->setThreadCrackSpeed(0, CPU_THREAD, searchRate);
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
        //printf("\n");

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
/*
#if USE_BOOST_MMAP
        // Nothing supported right now
#else
        madvise((void *)((uint64_t)&this->tableAccess[StartOffset * this->bytesInChain] & 0xfffffffffffff000),
        (uint32_t)(&this->tableAccess[EndOffset * this->bytesInChain] - &this->tableAccess[StartOffset * this->bytesInChain]) + 0x1000,
        MADV_WILLNEED);
#endif
*/
        for (CurrentOffset = StartOffset; CurrentOffset <= EndOffset; CurrentOffset++) {
            if (memcmpBits(&this->tableAccess[CurrentOffset * this->bytesInChain], (unsigned char *)&hash, this->bitsInHash) == 0) {
                thisHashFound = 1;
                //printf("Found hash!\n");
                hashLocation = CurrentOffset;
            }
        }


        if (thisHashFound) {
            // Back up to the start of the sequence or the beginning of the table
            while ((memcmpBits(&this->tableAccess[hashLocation * this->bytesInChain], (unsigned char *)&hash, this->bitsInHash) == 0)) {
                hashLocation--;
                // If the hash location is -1, we are at the beginning of the table.
                if (hashLocation < 0) {
                    break;
                }
            }
            hashLocation++;
            found++;
            while (memcmpBits(&this->tableAccess[hashLocation * this->bytesInChain], (unsigned char *)&hash, this->bitsInHash) == 0) {
                isDuplicate = 0;

                if (!isDuplicate) {
                    //printf("Found at %ld\n", hashLocation);
                    chainsToRegen++;
                    this->getChainAtIndex(hashLocation, &chainToRegen);
                    this->chainsToRegen.push_back(chainToRegen);
                }

                hashLocation++;
            }
        }
    }

#if !USE_BOOST_THREADS
    for (int thread = 0; thread < numberOfPrefetchThreads; thread++) {
        pthread_join(hintThreadStructure[thread], NULL);
    }
#endif
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
int GRTTableSearchV2::openOutputFile(char *outputFilename) {

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
int GRTTableSearchV2::closeOutputFile() {
    // Write the new table header and close the file.
    this->Table_Header->setNumberChains(this->numberChainsWritten);
    this->Table_Header->writeTableHeader(this->outputFile);
    fclose(this->outputFile);
	return 1;
}

// Write the chain to the opened file.
// TODO: Handle byte aligned hashes/passwords
int GRTTableSearchV2::writeChain(hashPasswordData* hashToWrite) {
    unsigned char *dataToWrite;
    uint64_t offset;

    // Allocate data to write based on the # of bits
    dataToWrite = new unsigned char[this->bytesInChain];
    memset(dataToWrite, 0, this->bytesInChain);
    
    // If the charset is not instantiated, load it up.
    if (!this->charset) {
        // Get the charset from the table header, then find the first row.
        // Right now, we do not handle per-position charsets.
        this->charset = this->Table_Header->getCharset();

        // Get the single charset being used and determine it's length
        this->currentCharset = charset[0];
        this->charsetLength = strlen(currentCharset);
    }

    // Get the offset as needed.
    offset = convertSingleCharsetToOffset((char *)hashToWrite->password, this->currentCharset);

    //printf("Offset: %lu\n", offset);
    copyHashAndOffsetIntoChain(dataToWrite, hashToWrite, offset, this->bitsInHash, this->bitsInPassword);

    if (!fwrite(dataToWrite, this->bytesInChain, 1, this->outputFile)) {
        printf("ERROR WRITING DATA\n");
        return 0;
    }
    
    delete[] dataToWrite;
    this->numberChainsWritten++;
    return 1;
}


// Converts the hash into a uint64_t value
// hashOffset is used to split out the high and low values.
// Use 0 to get the high 8 bytes, 8 to get the low 8 byes.
uint64_t GRTTableSearchV2::convertHashToUint64(const hashPasswordData &d1, int hashOffset) {
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

int GRTTableSearchV2::getBitsInHash() {
    return this->bitsInHash;
}

int GRTTableSearchV2::getBitsInPassword() {
    return this->bitsInPassword;
}
void GRTTableSearchV2::setCrackDisplay(GRTCrackDisplay* newCrackDisplay) {
    this->CrackDisplay = newCrackDisplay;
}

#if !USE_BOOST_THREADS
void *hintThread(void *hintDataVoid) {
    unsigned char hash[16];
    uint64_t min = 0, max = 0, check = 0;
    long int found = 0, i = 0;
    UINT4 hashId;
    struct regenChainData *regenChainData;
    uint64_t StartOffset, EndOffset;

    uint32_t MaskedData;
    uint64_t stepCount = 0;

    
    hintThreadData *hintData = (hintThreadData *)hintDataVoid;

    //printf("\n\nHINTTHREAD %d checking in...\n\n", hintData->threadId);

    hashData candidateHashData;
    hashPasswordData chainToRegen;

    memset(&candidateHashData, 0, sizeof(hashData));
    memset(&chainToRegen, 0, sizeof(hashPasswordData));

    for (hashId = hintData->threadId; hashId < hintData->candidateHashes->size(); hashId += hintData->stride) {
        stepCount++;
        
#if USE_PREFETCH_BARRIER
        // Wait every 100 steps for all threads.
        if ((stepCount % 500) == 0) {
            pthread_barrier_wait(hintData->hintBarrier);
        }
#endif        
        // Copy hash value into 'hash' variable.
        candidateHashData = hintData->candidateHashes->at(hashId);
        //printf("Hash: ");
        for (i = 0; i < 16; i++) {
            hash[i] = candidateHashData.hash[i];
        }

        MaskedData = hash[0] << 24 |
                hash[1] << 16 |
                hash[2] << 8 |
                hash[3];

        MaskedData &= hintData->Mask;
        //printf("Masked data: %08X\n", MaskedData);

        // Binary search the index.
        min = 0;
        max = hintData->TotalIndexes;


        while (max - min > 1) {
            check = (min + max) / 2;
            if (hintData->tableIndexFileAccess[check].Index == MaskedData) {
                goto hint_end;
            } else if (hintData->tableIndexFileAccess[check].Index > MaskedData) {
                max = check;
            } else {
                min = check;
            }
        }

        // Only back up if check is 1 or greater.
        if (check > 0) {
            check--;
        }

        while ((check < hintData->TotalIndexes) && (hintData->tableIndexFileAccess[check].Index < MaskedData)) {
            check++;
        }
hint_end:
        ;

        // Make sure we didn't wrap over the end.
        if (check >= hintData->TotalIndexes) {
            check = hintData->TotalIndexes - 1;
        }

        StartOffset = hintData->tableIndexFileAccess[check].Offset;
        if (check < (hintData->TotalIndexes - 1)) {
            EndOffset = hintData->tableIndexFileAccess[check + 1].Offset;
        } else {
            EndOffset = hintData->maxChainAddress;
        }

        // HINT!

        int madvisereturn;
        madvisereturn = madvise((void *) ((uint64_t) &hintData->tableAccess[StartOffset * hintData->bytesInChain] & 0xfffffffffffff000),
                (uint32_t) (&hintData->tableAccess[EndOffset * hintData->bytesInChain] - & hintData->tableAccess[StartOffset * hintData->bytesInChain]) + 0x1000,
                MADV_WILLNEED);

/*        if (madvisereturn) {
            printf("madvise error %s!\n", strerror(errno));
            printf("&hintData->tableAccess: %08x\n", hintData->tableAccess);
            printf("startOffset: %08x\n", ((uint64_t) &hintData->tableAccess[StartOffset * hintData->bytesInChain] & 0xfffffffffffff000));
            printf("EndOffset: %08x\n", (uint32_t)(&hintData->tableAccess[EndOffset * hintData->bytesInChain] - & hintData->tableAccess[StartOffset * hintData->bytesInChain]) + 0x1000);
        }*/
    }
}
#endif
