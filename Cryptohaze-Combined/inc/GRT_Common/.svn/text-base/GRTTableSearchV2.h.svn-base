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


// Table searching for V2 tables.

#ifndef __GRTTABLESEARCHV2_H__
#define __GRTTABLESEARCHV2_H__

/*
 * V2 Table Format and logic behind it
 *
 * V2 tables are bit packed with byte aligned elements.
 * The first N bits in the table are the hash, big endian order.
 * This makes sorting/searching easier.
 * As the hashes are coming in as a character array, this is a direct
 * memcpy to the table chain, with the final byte masked off.
 * 
 * If the length of the hash is not byte aligned, the final bits of the hash 
 * are stored as the most significant bits in the "intermediate" value.
 *
 * The password offset comes in as a uint64_t with the given number of
 * significant bits.
 *
 * It is stored after the hash as follows:
 * - The intermediate value takes the least significant bits of the
 * most significant byte of the offset.  This can easily be done by shifting.
 * - The remainder of the offset is stored in the remaining bytes, little endian
 * memory order.  This allows a straight memcpy to put the value in and out
 * of the chain.
 *
 * If the hash and password lengths are byte aligned, there is no intermediate
 * value containing values from both, and they butt up against each other.
 *
 * Examples:
 *
 * Hash: 0x1122334455667788
 * Pass offset: 0x01234567
 *
 * Pass as stored in memory (byte order):
 * 0x67 45 23 01 00 00
 *
 * Bits of hash: 36 (0x112233445)
 * Bits of password: 28 (0x1234567)
 *
 * Chain stored value:
 * 0x1122334451674523
 *
 * Hash value is in hash order, password offset is little endian except the
 * most significant byte.
 */


#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTTableSearch.h"
#include "GRT_Common/GRTTableHeaderV2.h"

#define MAX_PREFETCH_THREADS 16

// OS X does not have a barrier primative we use.
#define USE_PREFETCH_BARRIER 0

#if USE_BOOST_MMAP
#include <boost/iostreams/device/mapped_file.hpp>
using namespace boost::iostreams;
#else
#include <sys/mman.h>
#endif

typedef struct hintThreadData {
    std::vector<hashData>* candidateHashes;
    uint32_t Mask;
    int stride;
    uint32_t TotalIndexes;
    indexFile *tableIndexFileAccess;
    uint64_t maxChainAddress;
    unsigned char *tableAccess;
    int bytesInChain;
#if USE_PREFETCH_BARRIER
    pthread_barrier_t *hintBarrier;
#endif
    int threadId;
} hintThreadData;

// Entry points for pthreads
extern "C" {
    void *hintThread(void *);
}




class GRTTableSearchV2 : public GRTTableSearch {
public:
    GRTTableSearchV2();
    ~GRTTableSearchV2();

    // Sets the table filename to search.
    void SetTableFilename(const char *newTableFilename);

    // Give the table searcher the list of hashes it needs to find.
    void SetCandidateHashes(std::vector<hashData>* newCandidateHashes);

    // Actually searches the table.
    void SearchTable();

    // Return a list of the chains to regenerate
    std::vector<hashPasswordData>* getChainsToRegen();

    void setTableHeader(GRTTableHeader * newTableHeader);

    void getChainAtIndex(uint64_t index, struct hashPasswordData *chainInfo);

    uint64_t getNumberChains();

    // Open a file to output data to
    int openOutputFile(char *outputFilename);
    int closeOutputFile();

    // Write the chain to the opened file.
    int writeChain(hashPasswordData*);

    int getBitsInHash();
    int getBitsInPassword();

    void setCrackDisplay(GRTCrackDisplay *);
    
    void setPrefetchThreadCount(int newPrefetchThreadCount) {
        this->prefetchThreadCount = newPrefetchThreadCount;
    }
    
private:


    // Implements a binary search without any table indexes
    void SearchWithoutIndex();

    // Attempts to load an index file and determine the bits
    int LoadIndexFile(const char *newTableFilename);
    
    // Load the index file as a memory mapped file.
    int LoadIndexFilemMapped(const char *newTableFilename);

    // Implements an indexed table search
    void SearchWithIndex();

    uint64_t convertHashToUint64(const hashPasswordData &d1, int hashOffset);

    // Non-zero if an index is loaded.
    char indexFileIsLoaded;
    // Non-zero if memory mapping is used.
    char indexFileIsMemoryMapped;
    // Bits of the hash indexed
    int indexFileBits;
    // Struct access to the index file in memory
    indexFile *tableIndexFileAccess;
    // Total number of indexes
    uint64_t TotalIndexes;


    int tableFile;
    int indexFileId;

    // This is the pointer for the memory mapped file.
    unsigned char *tableMemoryBase;

    // This is the pointer to the beginning of the actual data for access.
    unsigned char *tableAccess;

    // The table size, in bytes (including header)
    uint64_t tableSize;

    // The maximum chain number in the table.
    uint64_t maxChainAddress;

    // Bits in the password and hash
    int bitsInPassword, bitsInHash;
    // Total bytes in the chain
    int bytesInChain;
    uint64_t hashMask;

    // Needed for speeding up chain writes
    char **charset;
    char *currentCharset;
    int charsetLength;


    // Candidate hashes to search for.
    std::vector<hashData>* candidateHashes;

    // Chains we have found.
    std::vector<hashPasswordData> chainsToRegen;

    GRTTableHeaderV2 *Table_Header;
    GRTCrackDisplay *CrackDisplay;
    char statusStrings[1024];

    FILE *outputFile;
    uint64_t numberChainsWritten;

#if USE_BOOST_MMAP
    mapped_file_source boost_mapped_source;
    mapped_file_source boost_mapped_index_source;
#endif

    int prefetchThreadCount;

};



#endif