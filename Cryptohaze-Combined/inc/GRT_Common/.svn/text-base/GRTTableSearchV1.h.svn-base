

// Table searching for V1 tables.

#ifndef __GRTTABLESEARCHV1_H__
#define __GRTTABLESEARCHV1_H__

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTTableSearch.h"
#include "GRT_Common/GRTTableHeaderV1.h"

#if USE_BOOST_MMAP
#include <boost/iostreams/device/mapped_file.hpp>
using namespace boost::iostreams;
#else
#include <sys/mman.h>
#endif

// For sorting the hashes
typedef struct tableV1DataStructure {
  unsigned char hash[16];
  unsigned char password[16];
} tableV1DataStructure;


class GRTTableSearchV1 : public GRTTableSearch {
public:
    GRTTableSearchV1();
    ~GRTTableSearchV1();

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

private:


    // Implements a binary search without any table indexes
    void SearchWithoutIndex();

    // Attempts to load an index file and determine the bits
    int LoadIndexFile(const char *newTableFilename);

    // Implements an indexed table search
    void SearchWithIndex();

    // Non-zero if an index is loaded.
    char indexFileIsLoaded;
    // Bits of the hash indexed
    int indexFileBits;
    // Struct access to the index file in memory
    indexFile *tableIndexFileAccess;
    // Total number of indexes
    uint64_t TotalIndexes;


    int tableFile;

    // This is the pointer for the memory mapped file.
    unsigned char *tableMemoryBase;

    // This is the pointer to the beginning of the actual data for access.
    tableV1DataStructure *tableAccess;

    // The table size, in bytes (including header)
    uint64_t tableSize;

    // The maximum chain number in the table.
    uint64_t maxChainAddress;


    // Candidate hashes to search for.
    std::vector<hashData>* candidateHashes;

    // Chains we have found.
    std::vector<hashPasswordData> chainsToRegen;

    GRTTableHeaderV1 *Table_Header;
    GRTCrackDisplay *CrackDisplay;
    char statusStrings[1024];

    FILE *outputFile;
    uint64_t numberChainsWritten;


#if USE_BOOST_MMAP
    mapped_file_source boost_mapped_source;
#endif

};



#endif