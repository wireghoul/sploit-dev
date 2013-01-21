#ifndef __GRTTABLESEARCH_H__
#define __GRTTABLESEARCH_H__

#include <vector>
#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTTableHeader.h"
#include "GRT_Common/GRTCrackDisplay.h"


class GRTTableSearch {
public:
    GRTTableSearch() {

    };

    virtual ~GRTTableSearch() {
        
    };
    // Sets the table filename to search.
    virtual void SetTableFilename(const char *newTableFilename) = 0;

    // Give the table searcher the list of hashes it needs to find.
    virtual void SetCandidateHashes(std::vector<hashData>* newCandidateHashes) = 0;

    // Actually searches the table.
    virtual void SearchTable() = 0;

    // Return a list of the chains to regenerate
    virtual std::vector<hashPasswordData>* getChainsToRegen() = 0;

    virtual void setTableHeader(GRTTableHeader *) = 0;

    // Returns the chain at the specified index: Hash/password combo for start/end chain.
    // This is a bit of an abuse of hashPasswordData, but it works just fine.
    virtual void getChainAtIndex(uint64_t index, struct hashPasswordData *chainInfo) = 0;

    virtual uint64_t getNumberChains() = 0;

    // Need to implement some writers here too...

    // Open a file to output data to
    virtual int openOutputFile(char *outputFilename) = 0;
    virtual int closeOutputFile() = 0;

    // Write the chain to the opened file.
    virtual int writeChain(hashPasswordData*) = 0;

    // Not needed by V1 tables, used by V1.  Returns -1 as "unused" value
    virtual int getBitsInHash() = 0;
    virtual int getBitsInPassword() = 0;

    virtual void setBitsInHash(int) {
        return;
    }

    virtual void setBitsInPassword(int) {
        return;
    }

    virtual void setCrackDisplay(GRTCrackDisplay *CrackDisplay) = 0;

    // Do nothing with this except for v2 tables
    virtual void setPrefetchThreadCount(int) { }

    // Web URL stuff - not implemented in any but VWeb
    virtual void setWebURL(std::string) { };
    virtual void setWebUsername(std::string) { };
    virtual void setWebPassword(std::string) { };

};



#endif