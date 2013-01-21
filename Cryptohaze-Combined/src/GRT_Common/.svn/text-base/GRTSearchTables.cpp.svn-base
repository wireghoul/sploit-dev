// A standalone table searcher for the web interface




#include <string>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableSearchV1.h"
#include "GRT_Common/GRTTableSearchV2.h"
#include "GRT_Common/GRTCommon.h"
#include <algorithm>

char silent = 0;

int main(int argc, char *argv[]) {
    if ((argc != 4) && (argc != 5)) {
        printf("You probably don't want to use me directly.\n");
        printf("I go with the web table script.\n");
        exit(1);
    }

    // argv[0]: program name
    // argv[1]: Table path
    // argv[2]: Candidate hash file
    // argv[3]: Output passwords file
    // argv[4]: Prefetch threads, optional
    
    std::string tablePath = argv[1];
    std::string candidateHashPath = argv[2];
    std::string regenChainsPath = argv[3];

    int prefetchThreads = 1;
    if (argc == 5) {
        prefetchThreads = atoi(argv[4]);
    }

    int TableVersion;
    GRTTableHeader *TableHeader;
    GRTTableSearch *TableSearch;
    FILE *hashfile;
    FILE *regenfile;
    char buffer[1024];
    std::vector<hashPasswordData>* chainsToRegen;
    hashData candidateHash;
    std::vector<hashData> CandidateHashList;



    TableVersion = getTableVersion(tablePath.c_str());

    if (TableVersion == -1) {
        printf("Cannot open %s\n", tablePath.c_str());
        exit(1);
    }


    if (TableVersion == 1) {
        TableHeader = new GRTTableHeaderV1();
        TableSearch = new GRTTableSearchV1();
    } else if (TableVersion == 2) {
        TableHeader = new GRTTableHeaderV2();
        TableSearch = new GRTTableSearchV2();
    } else {
        printf("Unknown table version %d in file %s.\n", TableVersion, tablePath.c_str());
        exit(1);
    }

    if (!TableHeader->readTableHeader(tablePath.c_str())) {
        printf("Error reading table header for %s.\n", tablePath.c_str());
        exit(1);
    }

    TableSearch->setTableHeader(TableHeader);
    TableSearch->SetTableFilename(tablePath.c_str());

    TableSearch->setPrefetchThreadCount(prefetchThreads);

    // Try to open the candidate file and read in the hashes

    //printf("Opening hash file %s\n", filename);


    hashfile = fopen(candidateHashPath.c_str(), "r");
    if (!hashfile) {
      printf("Cannot open hash file %s.  Exiting.\n", tablePath.c_str());
      exit(1);
    }

    while (!feof(hashfile)) {
        memset(buffer, 0, 1024);
        // If fgets returns NULL, there's been an error or eof.  Continue.
        if (!fgets(buffer, 1024, hashfile)) {
            continue;
        }

        // If this is not a full line, continue (usually a trailing crlf)
        if (strlen(buffer) < (16 * 2)) {
            continue;
        }

        // Clear the structure completely
        memset(&candidateHash, 0, sizeof (candidateHash));

        convertAsciiToBinary(buffer, (unsigned char*) &candidateHash.hash, 16);

        CandidateHashList.push_back(candidateHash);
    }

    std::sort(CandidateHashList.begin(), CandidateHashList.end(), hashDataSortPredicate);

    TableSearch->SetCandidateHashes(&CandidateHashList);

    TableSearch->SearchTable();


    chainsToRegen = TableSearch->getChainsToRegen();

    regenfile = fopen(regenChainsPath.c_str(), "w");
    if (!regenfile) {
        printf("Cannot open regen file!\n");
        exit(1);
    }

    for (int i = 0; i < chainsToRegen->size(); i++) {
        for (int j = 0; j < 16; j++) {
            if (chainsToRegen->at(i).password[j]) {
                fprintf(regenfile, "%c", chainsToRegen->at(i).password[j]);
            }
        }
        fprintf(regenfile,"\n");
    }
    fclose(regenfile);
}