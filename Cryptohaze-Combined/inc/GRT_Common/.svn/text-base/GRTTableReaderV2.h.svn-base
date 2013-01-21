#ifndef __GRTTABLEREADERV2_H__
#define __GRTTABLEREADERV2_H__

#include <vector>
#include <string>
#include "GRT_Common/GRTTableHeader.h"
#include "GRT_Common/GRTTableHeaderV2.h"

#define MAX_READER_PREFETCH_THREADS 16

// OS X does not have a barrier primative we use.
#define USE_PREFETCH_BARRIER 0

#if USE_BOOST_MMAP
#include <boost/iostreams/device/mapped_file.hpp>
using namespace boost::iostreams;
#else
#include <sys/mman.h>
#endif

typedef struct hintThreadDataR2 {
    std::vector<std::string>* candidateHashes;
    uint32_t indexMask;
    uint32_t indexShift;
    int stride;
    uint64_t maxChainAddress;
    unsigned char *tableAccess;
    std::vector<uint64_t>* indexAccess;
    int bytesInChain;
#if USE_PREFETCH_BARRIER
    pthread_barrier_t *hintBarrier;
#endif
    int threadId;
} hintThreadDataR2;

// Entry points for pthreads
extern "C" {
    void *hintThread(void *);
}

// Oh GFF, structure packing.
#pragma pack(push)
#pragma pack(1)
// The index file structure
typedef struct indexFileRv2 {
    uint32_t Index;
    uint64_t Offset;
} indexFileRv2;
#pragma pack(pop)

typedef struct Rv2chain {
    std::string hash;
    std::string password;
} Rv2chain;

/**
 * This is a threadsafe, read-only class for GRTv2 tables.  It is focused on
 * high performance searching only, and is probably not Windows friendly.
 */

class GRTTableReaderV2 {
public:
    GRTTableReaderV2();

    ~GRTTableReaderV2();
    
    /**
     * Attempts to open the specified table and index.  Returns true on success,
     * false on failure.
     * 
     * @param newTableFilename The table filename to open.
     * @param loadIntoMemory If true, the table file will be fully loaded into RAM.
     * @return True on success, false on failure or no index.
     */
    int OpenTableFile(std::string newTableFilename, char loadIntoMemory = 0);
    
    /**
     * Searches the opened table file.  Takes a vector of strings (which is what
     * comes out of the protobufs), and returns a vector of strings of the
     * candidate hashes.  The vector of candidate hashes is not required to be
     * the full length.  This function is required to not have any side effects
     * and to be fully threadsafe for recurrent entry!
     * 
     * @param candidateHashes Vector of candidates to search for.
     * @return A vector of chains to regenerate.
     */
    std::vector<std::string > searchTable(std::vector<std::string> candidateHashes);

    uint64_t getNumberChains() {
        return this->maxChainAddress;
    }

    int getBitsInHash() {
        return this->bitsInHash;
    }
    int getBitsInPassword() {
        return this->bitsInPassword;
    }

    void setPrefetchThreadCount(int newPrefetchCount) {
        this->prefetchThreadCount = newPrefetchCount;
    }
    
    // Returns a chain at the specified index.
    Rv2chain getChainAtIndex(uint64_t index);
    
    // Ensures the index is actually sane.
    int validateIndex();
    
    // Returns the 8192 bytes of table header
    std::string getTableHeaderAsString();

private:
    // Attempts to load an index file and determine the bits
    // Returns 1 on success, 0 on failure.
    int LoadIndexFile(std::string newTableFilename);

    uint64_t convertHashToUint64(std::string hashData, int hashOffset);
    
    // Bits of the hash indexed
    int indexFileBits;
    uint32_t indexMask;
    uint8_t indexShiftBits;

    // Struct access to the index file in memory
    std::vector<uint64_t> tableIndexOffsets;


    int tableFile;
    int indexFileId;

    // This is the pointer for the memory mapped file.
    unsigned char *tableMemoryBase;
    
    // True if the table has been malloc'd.
    char tableIsMalloced;

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


    GRTTableHeaderV2 *Table_Header;

#if USE_BOOST_MMAP
    mapped_file_source boost_mapped_source;
    mapped_file_source boost_mapped_index_source;
#endif

    int prefetchThreadCount;
};



#endif