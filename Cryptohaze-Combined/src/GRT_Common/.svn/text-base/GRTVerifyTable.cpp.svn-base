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


#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableSearchV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableSearchV2.h"
#include "GRT_Common/GRTTableHeaderV3.h"
#include "GRT_Common/GRTTableSearchV3.h"
#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTChainRunnerMD5.h"
#include "GRT_Common/GRTChainRunnerNTLM.h"
#include "GRT_Common/GRTChainRunnerSHA1.h"
#include "GRT_Common/GRTChainRunnerSHA256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

// Silence output if true.
char silent = 0;

uint64_t totalChainCount;


void printChainWithIndex(uint64_t index, hashPasswordData *chain) {
    int i;

    printf("Chain %lu\n", index);
    printf("Password: %s\n", chain->password);
    printf("Hash: ");
    for (i = 0; i < 16; i++) {
        printf("%02x", chain->hash[i]);
    }
    printf("\n\n");
}

int main (int argc, char *argv[]) {
    hashPasswordData currentHash;

    GRTTableHeader *TableHeader;
    GRTTableSearch *TableSearch;
    GRTChainRunner *ChainRunner;
    
    char verbose = 0;
    uint32_t stride = 1;

    uint64_t index;
    uint64_t chainsChecked = 0;

    // Set to 1 if any errors are detected.
    char tableError = 0;

    if (argc < 2) {
        printf("Usage: %s [table filename to verify] [verbose 0:1] [stride]\n", argv[0]);
        exit(1);
    }

    printf("argc: %d\n", argc);

    if (argc == 2) {
        verbose = 0;
    } else if (argv[2][0] == '0') {
        verbose = 0;
    } else if (argv[2][0] == '1') {
        verbose = 1;
    } else {
        printf("Verbose must be 0 or 1!\n");
        exit(1);
    }

    if (argc == 4) {
        stride = atoi(argv[3]);
    }

    if (getTableVersion(argv[1]) == 1) {
        TableHeader = new GRTTableHeaderV1();
        TableSearch = new GRTTableSearchV1();
    } else if (getTableVersion(argv[1]) == 2) {
        TableHeader = new GRTTableHeaderV2();
        TableSearch = new GRTTableSearchV2();
    } else if (getTableVersion(argv[1]) == 3) {
        TableHeader = new GRTTableHeaderV3();
        TableSearch = new GRTTableSearchV3();
    } else {
        printf("Table version %d not supported!\n", getTableVersion(argv[1]));
        exit(1);
    }

    if (!TableHeader->readTableHeader(argv[1])) {
        // Error will be printed.  Just exit.
        exit(1);
    }

    if (verbose || 1) {
        TableHeader->printTableHeader();
    }

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
            break;
        case 4:
            ChainRunner = new GRTChainRunnerSHA256();
            break;
        default:
            printf("Hash ID %d not supported!\n", TableHeader->getHashVersion());
            exit(1);
            break;
    }

    ChainRunner->setTableHeader(TableHeader);
    ChainRunner->setShowEachChain(verbose);
    // Start verifying chains.

    if (!TableSearch->getNumberChains()) {
        printf("Table appears to have no chains... eh?\n");
        exit(1);
    }

    totalChainCount = TableSearch->getNumberChains();



    for (index = 0; index < TableSearch->getNumberChains(); index+= stride) {
        memset(&currentHash, 0, sizeof(hashPasswordData));
        TableSearch->getChainAtIndex(index, &currentHash);

        if (verbose) {
            printChainWithIndex(index, &currentHash);
        }

        if (!ChainRunner->verifyChain(&currentHash)) {
            printf("Chain mismatch in chain %d!\n", index);
            tableError = 1;
        }
        if (chainsChecked % 100 == 0) {
            printf("\rProgress: %lu / %lu (%0.2f%%)   ", index, TableSearch->getNumberChains(),
                    100.0 * (float)index / (float)TableSearch->getNumberChains());
            fflush(stdout);
        }
        chainsChecked++;
    }
    printf("\n\nVerified %d of %d chains.\n", chainsChecked, TableSearch->getNumberChains());
    return tableError;
}