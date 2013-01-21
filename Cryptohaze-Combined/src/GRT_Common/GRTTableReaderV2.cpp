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

#include "GRT_Common/GRTTableReaderV2.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

static int memcmpBits(unsigned char *val1, unsigned char *val2, int bitsToCompare) {
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


// This undoes what the copyHashAndOffsetIntoChain funtion does.
// Calling is almost the same, but offset is a pointer now
static void getHashAndOffsetFromChain(std::vector<uint8_t> chainData, hashPasswordData* hashData,
        uint64_t *offset, int bitsInHash, int bitsInOffset) {

    int bitsRemaining, i;
    uint8_t intermediateByte, mask;
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
    memcpy(hashData->hash, &chainData[0], (bitsInHash / 8));

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


GRTTableReaderV2::GRTTableReaderV2() {
    this->maxChainAddress = 0;
    this->tableSize = 0;
    this->tableFile = 0;
    this->tableAccess = NULL;
    this->tableMemoryBase = NULL;
    this->indexMask = 0;
    this->indexShiftBits = 0;
    this->Table_Header = NULL;

    this->bitsInHash = 0;
    this->bitsInPassword = 0;

    this->charset = NULL;
    this->currentCharset = NULL;
    this->charsetLength = NULL;

    this->prefetchThreadCount = 0;
    
    this->tableIsMalloced = 0;
}

// For now, this class terminates when the binary exists, so there's no 
// point to cleaning up.  TODO: CLEAN UP PROPERLY!
GRTTableReaderV2::~GRTTableReaderV2() {
    if (this->tableIsMalloced) {
        free(this->tableMemoryBase);
    }
}


int GRTTableReaderV2::OpenTableFile(std::string newTableFilename, char loadIntoMemory) {
    struct stat file_status;

    // Chekc to make sure we can access the table and get data.
    if(stat(newTableFilename.c_str(), &file_status) != 0){
        printf("Unable to stat %s\n", newTableFilename.c_str());
        return 0;
    }

    // If we do not have a table header, read it.
    if (!this->Table_Header) {
        this->Table_Header = new GRTTableHeaderV2();
        this->Table_Header->readTableHeader(newTableFilename.c_str());
        this->bitsInHash = this->Table_Header->getBitsInHash();
        this->bitsInPassword = this->Table_Header->getBitsInPassword();
        this->bytesInChain = (this->bitsInHash + this->bitsInPassword) / 8;
    }
    
    if (loadIntoMemory) {
        // Load the file into memory completely.
        FILE *tableFileAccess;
        uint64_t bytesRead;
        
        printf("Attempting to load %s into memory.\n", newTableFilename.c_str());
        tableFileAccess = fopen(newTableFilename.c_str(), "rb");
        if (!tableFileAccess) {
            printf("Cannot open table file %s!\n", newTableFilename.c_str());
            return 0;
        }
        // Round up to cover 16 byte searches at the end.
        this->tableMemoryBase = (unsigned char *)malloc(file_status.st_size + 32);
        memset(this->tableMemoryBase, 0, file_status.st_size + 32);
        if (!this->tableMemoryBase) {
            printf("Cannot allocate %lu bytes of memory!\n", file_status.st_size);
            return 0;
        }
        this->tableIsMalloced = 1;
        
        bytesRead = fread(this->tableMemoryBase, file_status.st_size, 1, tableFileAccess);
        if (bytesRead != 1) {
            printf("Table read unsuccessful.\n");
            return 0;
        }
        fclose(tableFileAccess);
        printf("Table loaded into memory successfully.\n");
    } else {
        // MMap the file like normal.
#if USE_BOOST_MMAP
        // Use boost memory mapped files to get the table loaded.
        mapped_file_params table_params;

        table_params.path = newTableFilename;

        this->boost_mapped_source.open(table_params);
        this->tableMemoryBase = (unsigned char *)this->boost_mapped_source.data();
#else
        this->tableFile = open(newTableFilename.c_str(), O_RDONLY);
        if (!this->tableFile) {
            printf("Cannot open table file %s!\n", newTableFilename.c_str());
            return 0;
        }

        // Memory map the file.  This starts at the BASE - we deal with the offsets later.
        // Windows has some weird issues with offsets.  So we deal with that here.
        this->tableMemoryBase = (unsigned char *)mmap(0, file_status.st_size, PROT_READ, MAP_SHARED, this->tableFile, 0);

        // Advise the OS that access will be sequential.
        madvise (this->tableMemoryBase, file_status.st_size, MADV_SEQUENTIAL);
#endif
    }



    // Copy the table size over.
    this->tableSize = file_status.st_size;

    // Now do our offset work.  Table header v1 is 8192 bytes by definition.
    this->tableAccess = (unsigned char *) (this->tableMemoryBase + 8192);

    this->maxChainAddress = (this->tableSize - 8192) /
            ((this->Table_Header->getBitsInHash() + this->Table_Header->getBitsInPassword()) / 8);

    // Attempt to load the index file.  Index file is REQUIRED for use, so
    // just return the success or failure of the load.
    return this->LoadIndexFile(newTableFilename);
}


int GRTTableReaderV2::LoadIndexFile(std::string newTableFilename) {
    FILE *indexFile;
    uint64_t i, j;
    uint64_t TotalIndexes;
    indexFileRv2 indexData;
    uint32_t indexOffset;

    // Buffer for the possible index filename
    char filenameBuffer[2000];
    struct stat file_status;

    sprintf(filenameBuffer, "%s.idx", newTableFilename.c_str());

    // Check to make sure the file is present
    if(stat(filenameBuffer, &file_status) != 0){
        printf("Cannot find index file %s\n", filenameBuffer);
        return 0;
    }

    indexFile = fopen(filenameBuffer, "rb");

    if (!indexFile) {
        printf("Cannot open index file %s!\n", filenameBuffer);
        return 0;
    }

    TotalIndexes = file_status.st_size / sizeof(struct indexFileRv2);
    
    printf("Loading %d indexes\n", TotalIndexes);
    
    // Determine how many bits of index there are - round up to the next power
    // of two.
    uint64_t powerIndexCount = 1;
    this->indexFileBits = 0;
    while (powerIndexCount < TotalIndexes) {
        powerIndexCount *= 2;
        this->indexFileBits++;
    }
    printf("%lu indexes in file\n", TotalIndexes);
    printf("%lu indexes at bit marker\n", powerIndexCount);
    printf("Got %d bits in index.\n", this->indexFileBits);
    
    // Calculate the mask & shifts
    this->indexMask = powerIndexCount - 1;
    this->indexShiftBits = (32 - this->indexFileBits);

    // Resize the index buffer to the right size.
    this->tableIndexOffsets.resize(powerIndexCount, 0);
    
    // Read the indexes out of the table file.
    printf("\n\n");
    while (fread(&indexData, sizeof(struct indexFileRv2), 1, indexFile) == 1) {
        //printf("Read index: %08x... -> %lu\n", indexData.Index, indexData.Offset);
        indexOffset = (indexData.Index >> this->indexShiftBits) & this->indexMask;
        //printf("indexOffset: %u\n", indexOffset);
        this->tableIndexOffsets[indexOffset] = indexData.Offset;
        if ((indexOffset % 1000) == 0) {
            printf("\r%0.2f %% done  ", 100.0 * ((float)indexOffset / (float)powerIndexCount));
            fflush(stdout);
        } 
    }
    printf("\n\n");
    // Look for null values.
    for (i = 1; i < this->tableIndexOffsets.size(); i++) {
        if (this->tableIndexOffsets[i] == 0) {
            //printf("Null at offset %lu\n", i);
            // If there is a null value, set it to the previous value.
            if (i) {
                this->tableIndexOffsets[i] = this->tableIndexOffsets[i - 1];
            }
        }
    }
    printf("Table index loaded.\n");
    return 1;
}

Rv2chain GRTTableReaderV2::getChainAtIndex(uint64_t index) {
    
    std::vector<uint8_t> dataToRead;

    uint64_t offset;
    uint64_t addressToRead;
    uint64_t readIndex;
    int i;
    hashPasswordData chainInfo;
    Rv2chain returnChain;
    

    // Resize the vector so we can memcpy into it.
    dataToRead.resize(this->bytesInChain, 0);

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
    }
    
    // Split the buffer into the hash and offset.

    // Data is in the buffer.
    getHashAndOffsetFromChain(dataToRead, &chainInfo, &offset, this->bitsInHash, this->bitsInPassword);
    
    returnChain.hash.resize(MAX_HASH_LENGTH_BYTES, 0);
    memcpy(&returnChain.hash[0], chainInfo.hash, MAX_HASH_LENGTH_BYTES);
    returnChain.password.resize(this->Table_Header->getPasswordLength());
    
    int charsetLength = strlen(this->charset[0]);

    for (i = (this->Table_Header->getPasswordLength()) - 1; i >= 0; i--) {
        returnChain.password[i]= charset[0][offset % charsetLength];
        offset /= charsetLength;
    }
    
    return returnChain;
}

int GRTTableReaderV2::validateIndex() {
    printf("Validating indexes...\n");
    
    // Read through the table files and validate the index.
    uint64_t i;
    uint64_t indexOffset;
    uint32_t chainValue;
    Rv2chain chainData;

    printf("\n");
    for (i = 0; i < this->tableIndexOffsets.size(); i++) {
        if ((i % 1000) == 0) {
            printf("\r%0.2f%% done   ", 100.0 * (float)i / (float)this->tableIndexOffsets.size());
            fflush(stdout);
        }
        //printf("Index %u\n", i);
        //printf("Getting chain at index %lu\n", this->tableIndexOffsets[i]);
        chainData = this->getChainAtIndex(this->tableIndexOffsets[i]);
        //printf("got hash: ");
        for (int j = 0; j < 8; j++) {
            //printf("%02x", (uint8_t)chainData.hash[j]);
        }
        //printf("\n");
        
        chainValue = ((uint32_t)(uint8_t)chainData.hash[0]) << 24 |
                     ((uint32_t)(uint8_t)chainData.hash[1]) << 16 |
                     ((uint32_t)(uint8_t)chainData.hash[2]) << 8  |
                     ((uint32_t)(uint8_t)chainData.hash[3]);
        
        //printf("Read chain value %08x\n", chainValue);
        
        
        
        indexOffset = (chainValue >> this->indexShiftBits) & this->indexMask;
        if (indexOffset != i) {
            printf("=========================\n");
            printf("Offset error in index %lu \n", i);
            printf("Caclulated offset: %lu\n", indexOffset);
            printf("Actual offset: %lu\n", i);
            printf("=========================\n");
        }
    }
    printf("\n\nValidating indexes completed.\n");
    return 1;
}


std::vector<std::string > GRTTableReaderV2::searchTable(std::vector<std::string> candidateHashes) {
    unsigned char hash[16];
    uint64_t min = 0, max = 0, check = 0;
    int64_t hashLocation = 0;
    int thisHashFound = 0, isDuplicate;
    long int found = 0, i = 0, chainsToRegen = 0;
    UINT4 hashId;
    uint64_t StartOffset, EndOffset, CurrentOffset;

    uint32_t indexOffset;
    
    std::string candidateHashData;
    std::vector<std::string> chainsToReturn;
    
    Rv2chain foundChain;

    for (hashId = 0; hashId < candidateHashes.size(); hashId++) {
        if (false && (hashId % 500) == 0) {
        // Determine the current time of searching in a cross-platform way.
        printf("\rStep %d / %d (%0.2f%%) %ld chains found       ",
            hashId, candidateHashes.size(),
            (100.0 * (float)hashId / (float)candidateHashes.size()), found);
        fflush(stdout);
        }


        // Copy hash value into 'hash' variable.
        candidateHashData = candidateHashes.at(hashId);
        //printf("Hash: ");
        memset(hash, 0, 16);
        for (i = 0; i < candidateHashData.length(); i++) {
            hash[i] = candidateHashData[i];
            //printf("%02x", hash[i]);
        }
        //printf("\n");

        indexOffset = hash[0] << 24 |
                      hash[1] << 16 |
                      hash[2] << 8  |
                      hash[3];
        indexOffset = (indexOffset >> this->indexShiftBits) & this->indexMask;

        // Get the start and end offsets.
        StartOffset = this->tableIndexOffsets[indexOffset];
        if (indexOffset < (this->tableIndexOffsets.size() - 1)) {
            EndOffset = this->tableIndexOffsets[indexOffset + 1];
        } else {
            EndOffset = this->maxChainAddress;
        }

#if USE_BOOST_MMAP
        // Nothing supported right now
#else
        madvise((void *)((uint64_t)&this->tableAccess[StartOffset * this->bytesInChain] & 0xfffffffffffff000),
        (uint32_t)(&this->tableAccess[EndOffset * this->bytesInChain] - &this->tableAccess[StartOffset * this->bytesInChain]) + 0x1000,
        MADV_WILLNEED);
#endif

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
                chainsToRegen++;
                foundChain = this->getChainAtIndex(hashLocation);
                chainsToReturn.push_back(foundChain.password);
                hashLocation++;
            }
        }
    }
/*
#if !USE_BOOST_THREADS
    for (int thread = 0; thread < numberOfPrefetchThreads; thread++) {
        //pthread_join(hintThreadStructure[thread], NULL);
    }
#endif
*/
    //printf("Total chains found to regen before merging: %d\n", chainsToReturn.size());

    sort(chainsToReturn.begin(), chainsToReturn.end());
    chainsToReturn.erase(
        std::unique(chainsToReturn.begin(), chainsToReturn.end()),
        chainsToReturn.end() );

    //printf("Total chains found to regen after merging: %d\n", chainsToReturn.size());
    
    return chainsToReturn;
}

std::string GRTTableReaderV2::getTableHeaderAsString() {
    // Create a string from the table base in memory.
    return std::string((const char *)this->tableMemoryBase, 8192);
}