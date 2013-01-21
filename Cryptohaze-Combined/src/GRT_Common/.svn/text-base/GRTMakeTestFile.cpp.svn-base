// Given a RT file, generate a test file with a given number of hashes out of the
// main file.  These hashes WILL be present in the file, and all of them should
// be reported successfully.  Not finding them IS A BUG.

#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableSearchV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableSearchV2.h"
#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTChainRunnerMD5.h"
#include "GRT_Common/GRTChainRunnerNTLM.h"
#include "GRT_Common/GRTChainRunnerSHA1.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>

char silent;

int main (int argc, char *argv[]) {

    GRTTableHeader *TableHeader;
    GRTTableSearch *TableSearch;
    GRTChainRunner *ChainRunner;

    uint32_t numberOfChainsToGet, i;

    char filenamebuffer[1024];
    FILE *hashes;
    FILE *passwords;

    int bytesInHash = 16;

    hashPasswordData chainToRegen, chainStepToReport;


    if (argc != 4) {
        printf("Usage: [program] [RT file] [number of hashes] [output hash list name]\n");
        exit(1);
    }

    srand ( time(NULL) );


    if (getTableVersion(argv[1]) == 1) {
        TableHeader = new GRTTableHeaderV1();
        TableSearch = new GRTTableSearchV1();
    } else if (getTableVersion(argv[1]) == 2) {
        TableHeader = new GRTTableHeaderV2();
        TableSearch = new GRTTableSearchV2();
    } else {
        printf("Table version %d not supported!\n", getTableVersion(argv[1]));
        exit(1);
    }

    if (!TableHeader->readTableHeader(argv[1])) {
        // Error will be printed.  Just exit.
        exit(1);
    }

    TableHeader->printTableHeader();

    TableSearch->SetTableFilename(argv[1]);

    TableSearch->setTableHeader(TableHeader);

    switch (TableHeader->getHashVersion()) {
        case 0:
            ChainRunner = new GRTChainRunnerNTLM();
            break;
        case 1:
            ChainRunner = new GRTChainRunnerMD5();
            break;
        case 3:
            ChainRunner = new GRTChainRunnerSHA1();
            bytesInHash = 20;
            break;
        default:
            printf("Hash ID %d not supported!\n", TableHeader->getHashVersion());
            exit(1);
            break;
    }

    ChainRunner->setTableHeader(TableHeader);

    numberOfChainsToGet = atoi(argv[2]);

    printf("About to get %d hashes.\n", numberOfChainsToGet);

    sprintf(filenamebuffer, "%s-hashes", argv[3]);
    hashes = fopen(filenamebuffer, "w");
    if (!hashes) {
        printf("Cannot open file %s for writing!\n", filenamebuffer);
        exit(1);
    }

    sprintf(filenamebuffer, "%s-passwords", argv[3]);
    passwords = fopen(filenamebuffer, "w");
    if (!passwords) {
        printf("Cannot open file %s for writing!\n", filenamebuffer);
        exit(1);
    }


    for (i = 0; i < numberOfChainsToGet; i++) {
        uint64_t chainToGetFrom = rand() % TableHeader->getNumberChains();
        uint32_t chainIndexToGet = rand() % TableHeader->getChainLength();

        printf("Chain: %lu\n", chainToGetFrom);
        printf("Index: %u\n", chainIndexToGet);

        TableSearch->getChainAtIndex(chainToGetFrom, &chainToRegen);
        chainStepToReport = ChainRunner->getLinkAtChainIndex(&chainToRegen, chainIndexToGet);

        for (int j = 0; j < bytesInHash; j++) {
            fprintf(hashes, "%02x", chainToRegen.hash[j]);
            fprintf(passwords, "%02x", chainToRegen.hash[j]);
            printf("%02x", chainToRegen.hash[j]);
        }
        fprintf(hashes, "\n");
        fprintf(passwords, ":%s\n", chainToRegen.password);
        printf(":%s\n", chainToRegen.password);
        
    }
    fclose(hashes);
    fclose(passwords);

    printf("Done writing %d hashes.\n\n", numberOfChainsToGet);
}