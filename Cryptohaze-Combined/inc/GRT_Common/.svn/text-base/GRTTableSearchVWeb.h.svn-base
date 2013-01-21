

// Table searching for VWeb tables.

#ifndef __GRTTABLESEARCHVWEB_H__
#define __GRTTABLESEARCHVWEB_H__

// We pass everything to the web. :)

#include <vector>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTTableSearch.h"
#include "GRT_Common/GRTTableHeaderVWeb.h"


class GRTTableSearchVWeb : public GRTTableSearch {
public:
    GRTTableSearchVWeb();
    ~GRTTableSearchVWeb();

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

    // Open a file to output data to: Not implemented.
    int openOutputFile(char *outputFilename) { return 0;}
    int closeOutputFile() { return 0;}

    // Write the chain to the opened file: Not implemented
    int writeChain(hashPasswordData*) { return 0;}

    int getBitsInHash();
    int getBitsInPassword();

    void setCrackDisplay(GRTCrackDisplay *);

    // Web URL stuff - not implemented in any but VWeb
    void setWebURL(std::string newWebURL) {
        this->tableURL = newWebURL;
    };
    void setWebUsername(std::string newWebUsername) {
        this->tableUsername = newWebUsername;
    };
    void setWebPassword(std::string newWebPassword) {
        this->tablePassword = newWebPassword;
    };

private:

    // Bits in the password and hash
    int bitsInPassword, bitsInHash;

    std::string tableFilename;

    // Candidate hashes to search for.
    std::vector<hashData>* candidateHashes;

    // Chains we have found.
    std::vector<hashPasswordData> chainsToRegen;

    GRTTableHeaderVWeb *Table_Header;
    GRTCrackDisplay *CrackDisplay;
    char statusStrings[1024];

    std::string tableURL;
    std::string tableUsername;
    std::string tablePassword;

};



#endif