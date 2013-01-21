

// Table searching for V3 tables.

#ifndef __GRTTABLESEARCHV3_H__
#define __GRTTABLESEARCHV3_H__

/*
 * V3 Table Format and logic behind it
 *
 * V3 tables are unsorted tables used to reduce the amount of data
 * transfer required to do distributed table gen.
 *
 * They store the specified number of bits of hash (must be byte length)
 * and store the start point/seed in the header so that passwords can be
 * regenerated to match the chains at the server.  This requires that the
 * hashes not be sorted coming off the GPU for generation!
 *
 * As such, there are no table searching functions implemented - the only
 * functions to deal with data are writeChain and getChainAtIndex.
 *
 * Should be fairly straightforward...
 *
 */


#include <vector>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTTableSearch.h"
#include "GRT_Common/GRTTableHeaderV3.h"

#if USE_BOOST_MMAP
#include <boost/iostreams/device/mapped_file.hpp>
using namespace boost::iostreams;
#else
#include <sys/mman.h>
#endif

class GRTTableSearchV3 : public GRTTableSearch {
public:
    GRTTableSearchV3();
    ~GRTTableSearchV3();

    // Sets the table filename to search.
    void SetTableFilename(const char *newTableFilename);

    // Give the table searcher the list of hashes it needs to find.
    // Not implemented - no table searching available.
    void SetCandidateHashes(std::vector<hashData>* newCandidateHashes) { };

    // Actually searches the table.
    void SearchTable() { };

    // Return a list of the chains to regenerate
    std::vector<hashPasswordData>* getChainsToRegen() {return NULL;};

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

    void setCrackDisplay(GRTCrackDisplay *) { };

private:

    int tableFile;

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

    GRTTableHeaderV3 *Table_Header;

    FILE *outputFile;
    uint64_t numberChainsWritten;

    // Storage for the initial passwords
    std::vector<std::string> initialPasswords;

#if USE_BOOST_MMAP
    mapped_file_source boost_mapped_source;
#endif

};



#endif