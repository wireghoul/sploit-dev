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

// This code converts a table between formats.

// Right now, it assumes you want to go from V1 to V2.

#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableSearchV1.h"
#include "GRT_Common/GRTTableSearchV2.h"
#include "GRT_Common/GRTCommon.h"

#include <stdio.h>
#include <stdlib.h>

char silent = 0;

// For now:
// Arg1: Source file
// Arg2: Bits of password to save
// Arg3: Output filename
int main(int argc, char *argv[]) {

    GRTTableHeader *SourceTableHeader;
    GRTTableHeaderV2 *DestinationTableHeader;
    GRTTableSearch *SourceTable, *DestinationTable, *ReadbackTable;
    uint64_t chainIndex;
    hashPasswordData chainData;

    // Sanity check
    if (argc != 4) {
        printf("Usage: %s [source file] [bits of hash] [destination file]\n", argv[0]);
        exit(1);
    }

    if (!atoi(argv[2])) {
        printf("Invalid number of bits...\n");
        exit(1);
    }


    if (getTableVersion(argv[1]) != 1) {
        printf("Error: GRTConvert can only convert V1 to V2 tables.\n");
        printf("To trim the number of bits of hash in a V2 table, \n");
        printf("use GRTMerge.\n");
        exit(1);
    }

    // For now, V1 to V2 conversions

    SourceTableHeader = new GRTTableHeaderV1();
    DestinationTableHeader = new GRTTableHeaderV2();

    SourceTable = new GRTTableSearchV1();
    DestinationTable = new GRTTableSearchV2();
    ReadbackTable = new GRTTableSearchV2();

    
    // Read in the source table data.
    SourceTableHeader->readTableHeader(argv[1]);
    SourceTableHeader->printTableHeader();



    SourceTable->setTableHeader(SourceTableHeader);

    SourceTable->SetTableFilename(argv[1]);

    // Should be good now to continue on.

    DestinationTableHeader->setChainLength(SourceTableHeader->getChainLength());
    DestinationTableHeader->setCharsetCount(SourceTableHeader->getCharsetCount());
    DestinationTableHeader->setCharsetLengths(SourceTableHeader->getCharsetLengths());
    DestinationTableHeader->setCharset(SourceTableHeader->getCharset());
    DestinationTableHeader->setHashName(SourceTableHeader->getHashName());
    DestinationTableHeader->setHashVersion(SourceTableHeader->getHashVersion());
    DestinationTableHeader->setNumberChains(SourceTableHeader->getNumberChains());
    DestinationTableHeader->setPasswordLength(SourceTableHeader->getPasswordLength());
    DestinationTableHeader->setTableIndex(SourceTableHeader->getTableIndex());
    DestinationTableHeader->setTableVersion(2);
    DestinationTableHeader->setBitsInHash(atoi(argv[2]));


    // Print the new table header.
    DestinationTableHeader->printTableHeader();

    DestinationTable->setTableHeader(DestinationTableHeader);
    DestinationTable->openOutputFile(argv[3]);


    printf("\n\n");
    for (chainIndex = 0; chainIndex < SourceTable->getNumberChains(); chainIndex++) {
        if ((chainIndex % 10000) == 0) {
            printf("Chain %lu / %lu  (%0.2f%%)   \r", chainIndex, SourceTable->getNumberChains(),
                    100.0 * ((float)chainIndex / (float)SourceTable->getNumberChains()));
            fflush(stdout);
        }
        SourceTable->getChainAtIndex(chainIndex, &chainData);
        //printf("Chain %d pass: %s\n", chainIndex, chainData.password);
        DestinationTable->writeChain(&chainData);
    }

    DestinationTable->closeOutputFile();
    printf("\n\n");

/*
    printf("\n\n==== READBACK ====\n\n");

    // Now try reading back...
    ReadbackTable = new GRTTableSearchV2();

    ReadbackTable->SetTableFilename(argv[3]);

    printf("Num Chains: %lu\n", ReadbackTable->getNumberChains());

    for (chainIndex = 0; chainIndex < ReadbackTable->getNumberChains(); chainIndex++) {
        ReadbackTable->getChainAtIndex(chainIndex, &chainData);
        printf("Chain %d password: %s\n", chainIndex, chainData.password);
        printf("Hash:");
        for (int i = 0 ; i < 16; i++) {
            printf("%02x", chainData.hash[i]);
        }
        printf("\n");
    }

*/
}