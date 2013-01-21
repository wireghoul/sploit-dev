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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "GRT_CUDA_host/GRTCrackCommandLineData.h"
#include "GRT_CUDA_host/GRTCandidateHashesMD5.h"
#include "GRT_CUDA_host/GRTCandidateHashesNTLM.h"
#include "GRT_CUDA_host/GRTCandidateHashesSHA1.h"
#include "GRT_CUDA_host/GRTCandidateHashesSHA256.h"
#include "CH_Common/GRTWorkunit.h"
#include "GRT_Common/GRTHashes.h"
#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableSearchV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableSearchV2.h"
#include "GRT_Common/GRTTableHeaderVWeb.h"
#include "GRT_Common/GRTTableSearchVWeb.h"
#include "CH_Common/GRTHashFilePlain.h"
#include "GRT_CUDA_host/GRTRegenerateChainsMD5.h"
#include "GRT_CUDA_host/GRTRegenerateChainsNTLM.h"
#include "GRT_CUDA_host/GRTRegenerateChainsSHA1.h"
#include "GRT_CUDA_host/GRTRegenerateChainsSHA256.h"
#include "GRT_Common/GRTCrackDisplayCurses.h"
#include "GRT_Common/GRTCrackDisplayDebug.h"

#include <boost/filesystem.hpp>

// Silence output if true.
char silent = 0;

int main(int argc, char *argv[]) {

    GRTCrackCommandLineData CommandLineData;
    GRTCandidateHashes *CandidateHash;
    hashPasswordData HashData;
    GRTWorkunit Workunit;
    GRTHashFilePlain *HashFile;

    GRTRegenerateChains *RegenChains;

    GRTTableHeader *TableHeader;
    GRTTableSearch *TableSearch;

    GRTHashes HashNames;

    // Default for Display is null: If it remains null, nothing should use it.
    GRTCrackDisplay *Display = NULL;

    TableHeader = NULL;
    TableSearch = NULL;

    std::vector<hashData> * candidateHashesVector;
    
    unsigned char *Hash;

    int i;
    const char *nextTableFilename;

    char TableVersion;

    unsigned char *hashList;

    char statusStrings[1024];

    silent = 0;

    CommandLineData.ParseCommandLine(argc, argv);

    if (CommandLineData.getIsSilent()) {
        silent = 1;
    }
    
    switch(CommandLineData.getHashType()) {
        case 0:
            CandidateHash = new GRTCandidateHashesNTLM();
            RegenChains = new GRTRegenerateChainsNTLM();
            HashFile = new GRTHashFilePlain(16);
            break;
        case 1:
            CandidateHash = new GRTCandidateHashesMD5();
            RegenChains = new GRTRegenerateChainsMD5();
            HashFile = new GRTHashFilePlain(16);
            break;
        case 3:
            CandidateHash = new GRTCandidateHashesSHA1();
            RegenChains = new GRTRegenerateChainsSHA1();
            HashFile = new GRTHashFilePlain(20);
            break;
        case 4:
            CandidateHash = new GRTCandidateHashesSHA256();
            RegenChains = new GRTRegenerateChainsSHA256();
            HashFile = new GRTHashFilePlain(32);
            break;
        default:
            printf("This hash type is not supported yet!\n");
            exit(1);
            break;
    }

    // Use display debug for now
    if (!silent) {
        if (CommandLineData.getDebug()) {
            Display = new GRTCrackDisplayDebug();
        } else {
            Display = new GRTCrackDisplayCurses();
        }

        Display->setHashName((char *)HashNames.GetHashStringFromId(CommandLineData.getHashType()));
    }



    CandidateHash->setCommandLineData(&CommandLineData);
    RegenChains->setCommandLineData(&CommandLineData);


    // Add GPUs to the Candidate Hash and Chain Regen classes
    for (i = 0; i < CommandLineData.getCudaNumberDevices(); i++) {
        if (CommandLineData.getDeveloperDebug()) {
            printf("Adding GPU ID %d\n", i);
        }
        CandidateHash->addGPUDeviceID(i);
        RegenChains->addGPUDeviceID(i);
    }


    if (CommandLineData.getUseHashFile()) {
        // Do the hash file work

        // Open the hash file
        HashFile->OpenHashFile((char *)CommandLineData.getHashFileName().c_str());
        if (Display) {
            sprintf(statusStrings, "Loaded %d hashes", HashFile->GetTotalHashCount());
            Display->addStatusLine(statusStrings);
        }
    } else {
        // Work with a single hash.
        Hash = CommandLineData.getHash();
        HashFile->AddHashBinaryString((const char *)Hash);
    }

    // Inform the hash file if we want hex output
    HashFile->SetAddHexOutput(CommandLineData.getAddHexOutput());

    if (CommandLineData.getUseOutputHashFile()) {
        HashFile->SetFoundHashesOutputFilename(CommandLineData.getOutputHashFile().c_str());
    }

    hashList = HashFile->ExportUncrackedHashList();
    for (i = 0; i < HashFile->GetTotalHashCount(); i++) {
        memcpy(HashData.hash, &hashList[i * HashFile->GetHashLength()], HashFile->GetHashLength());
        CandidateHash->addHashToCrack(&HashData, HashFile->GetHashLength());
    }
    delete[] hashList;

    // Set the total number of table files and hashes (if using display)
    if (Display) {
        Display->setTotalTables(CommandLineData.getNumberOfTableFiles());
        Display->setTotalHashes(HashFile->GetTotalHashCount());
    }

    // For each file selected, try it out.
    for (i = 0; i < CommandLineData.getNumberOfTableFiles(); i++) {
        nextTableFilename = CommandLineData.getNextTableFile();

        // Clear out any old GRTTableHeader & GRTTableSearch classes
        if (TableHeader) {
            delete TableHeader;
            TableHeader = NULL;
        }
        if (TableSearch) {
            delete TableSearch;
            TableSearch = NULL;
        }

        // If we are using the web table, do that.  Otherwise, use local tables.
        if (CommandLineData.getUseWebTable()) {
            TableHeader = new GRTTableHeaderVWeb();
            TableSearch = new GRTTableSearchVWeb();

            TableHeader->setWebURL(CommandLineData.getTableURL());
            TableHeader->setWebUsername(CommandLineData.getTableUsername());
            TableHeader->setWebPassword(CommandLineData.getTablePassword());
            TableHeader->setDisplay(Display);
            
            TableSearch->setWebURL(CommandLineData.getTableURL());
            TableSearch->setWebUsername(CommandLineData.getTableUsername());
            TableSearch->setWebPassword(CommandLineData.getTablePassword());


        } else {
            // Determine the table version for the next table
            TableVersion = getTableVersion(nextTableFilename);

            if (TableVersion == -1) {
                if (Display) {
                    delete Display;
                }
                printf("Invalid table %s!", nextTableFilename);
                exit(1);
            }

            if (TableVersion == 1) {
                TableHeader = new GRTTableHeaderV1();
                TableSearch = new GRTTableSearchV1();
            } else if (TableVersion == 2) {
                TableHeader = new GRTTableHeaderV2();
                TableSearch = new GRTTableSearchV2();
            } else {
                if (Display) {
                    delete Display;
                }
                printf("Unknown table version %d in file %s.\n", TableVersion, nextTableFilename);
                exit(1);
            }
        }



        if (Display) {
            boost::filesystem::path tablePath(nextTableFilename);
            Display->setTableFilename(tablePath.filename().string().c_str());
            Display->setCurrentTableNumber(i + 1);
        }
        if (!TableHeader->readTableHeader(nextTableFilename)) {
            if (Display) {
                delete Display;
                Display = NULL;
            }
            printf("Error reading table header for %s.\n", nextTableFilename);
            break;
        }

        if (CommandLineData.getDeveloperDebug()) {
            TableHeader->printTableHeader();
        }

        // Print out some basic table information to the status lines
        if (Display) {
            sprintf(statusStrings, "Table Info");
            Display->addStatusLine(statusStrings);
            sprintf(statusStrings, "Table Ver: %d", TableHeader->getTableVersion());
            Display->addStatusLine(statusStrings);
            sprintf(statusStrings, "Pass Len: %d", TableHeader->getPasswordLength());
            Display->addStatusLine(statusStrings);
            sprintf(statusStrings, "Index: %lu", TableHeader->getTableIndex());
            Display->addStatusLine(statusStrings);
            sprintf(statusStrings, "Chain Len: %lu", TableHeader->getChainLength());
            Display->addStatusLine(statusStrings);
            sprintf(statusStrings, "Num Chains: %lu", TableHeader->getNumberChains());
            Display->addStatusLine(statusStrings);
       }

        Workunit.CreateWorkunits(HashFile->GetUncrackedHashCount(), 0);
        if (Display) {
            Display->setWorkunitsTotal(Workunit.GetNumberOfWorkunits());
            Display->setWorkunitsCompleted(0);
        }

        // Provide the candidate hash class the table header for table info.
        CandidateHash->clearCandidates();
        CandidateHash->clearHashesToCrack();
        CandidateHash->setTableHeader(TableHeader);
        CandidateHash->setWorkunit(&Workunit);
        CandidateHash->setDisplay(Display);
		CandidateHash->SetCandidateHashesToSkip(CommandLineData.getCandidateHashesToSkip());

        // Update the current uncracked hash list.
        unsigned char *hashList;

        hashList = HashFile->ExportUncrackedHashList();

        for (int hashCount = 0; hashCount < HashFile->GetUncrackedHashCount(); hashCount++) {
            memcpy(HashData.hash, &hashList[hashCount * HashFile->GetHashLength()], HashFile->GetHashLength());
            CandidateHash->addHashToCrack(&HashData, HashFile->GetHashLength());
            if (CommandLineData.getDeveloperDebug()) {
                printf("Adding hash ");
                for (int pos = 0; pos < 16; pos++) {
                    printf("%02x", HashData.hash[pos]);
                }
                printf("\n");
            }
        }
        
        delete[] hashList;

        if (Display) {
            Display->setSystemStage(GRT_CRACK_CHGEN);
        }

        CandidateHash->generateCandidateHashes();


        // Get the vectors back.
        candidateHashesVector = CandidateHash->getGeneratedCandidates();

        if (CommandLineData.getDeveloperDebug()) {
            for (int chain = 0; chain < candidateHashesVector->size(); chain++) {
                printf("Candidate hash %d: ", chain);
                for (int pos = 0; pos < 16; pos++) {
                    printf("%02x", candidateHashesVector->at(chain).hash[pos]);
                }
                printf("\n");
            }
        }

        if (CommandLineData.getDebugDump()) {
            // Output candidate hashes to disk as well as to screen.
            FILE *candidateHashOutput;
            std::string candidateHashFilename;

            candidateHashFilename = CommandLineData.getDebugDumpFilename();
            candidateHashFilename += ".candidates";
            // Writing ASCII - just "w", not "wb" to muck up newlines.
            candidateHashOutput = fopen(candidateHashFilename.c_str(), "w");
            for (int chain = 0; chain < candidateHashesVector->size(); chain++) {
                for (int pos = 0; pos < 16; pos++) {
                    if (candidateHashOutput) {
                        fprintf(candidateHashOutput, "%02X", candidateHashesVector->at(chain).hash[pos]);
                    }
                }
                fprintf(candidateHashOutput, "\n");
            }
            fclose(candidateHashOutput);
        }

        TableSearch->setCrackDisplay(Display);

        TableSearch->setTableHeader(TableHeader);
        TableSearch->SetTableFilename(nextTableFilename);
        TableSearch->SetCandidateHashes(candidateHashesVector);
        TableSearch->setPrefetchThreadCount(CommandLineData.getNumberPrefetchThreads());

        if (Display) {
            Display->setSystemStage(GRT_CRACK_SEARCH);
        }

        TableSearch->SearchTable();

        candidateHashesVector->clear();

        delete candidateHashesVector;

        std::vector<hashPasswordData>* chainsToRegen;

        chainsToRegen = TableSearch->getChainsToRegen();

        if (CommandLineData.getDeveloperDebug()) {
            for (int chain = 0; chain < chainsToRegen->size(); chain++) {
                printf("Chain to regen %d: %s\n", chain, chainsToRegen->at(chain).password);
            }
        }

        if (CommandLineData.getDebugDump()) {
            // Output candidate hashes to disk as well as to screen.
            FILE *chainsOutput;
            std::string chainsFilename;

            chainsFilename = CommandLineData.getDebugDumpFilename();
            chainsFilename += ".chains";
            // Writing ASCII - just "w", not "wb" to muck up newlines.
            chainsOutput = fopen(chainsFilename.c_str(), "w");
            for (int chain = 0; chain < chainsToRegen->size(); chain++) {
                fprintf(chainsOutput, "%s\n", chainsToRegen->at(chain).password);
            }
            fprintf(chainsOutput, "\n");
            fclose(chainsOutput);
        }


        if (Display) {
            sprintf(statusStrings, "To regen: %lu", chainsToRegen->size());
            Display->addStatusLine(statusStrings);
        }

        // Regenerate chain process.
        RegenChains->setDisplay(Display);
        RegenChains->setHashfile(HashFile);
        RegenChains->setTableHeader(TableHeader);

        // Let's make workunits of 256 chains each, for fun for now.
        Workunit.CreateWorkunits(chainsToRegen->size(), 16);
        if (Display) {
            Display->setWorkunitsTotal(Workunit.GetNumberOfWorkunits());
        }
        RegenChains->setWorkunit(&Workunit);
        // Copy in the chains we are regenerating
        RegenChains->setChainsToRegen(chainsToRegen);
        
        if (Display) {
            Display->setSystemStage(GRT_CRACK_REGEN);
            Display->setWorkunitsCompleted(0);
            Display->setWorkunitsTotal(Workunit.GetNumberOfWorkunits());
        }
        RegenChains->regenerateChains();

        chainsToRegen->clear();
        delete chainsToRegen;
        
        delete[] nextTableFilename;

        // If there are no hashes left, exit.
        if (!HashFile->GetUncrackedHashCount()) {
            break;
        }

    }

    delete CandidateHash;
    delete RegenChains;

    if (TableHeader) {
        delete TableHeader;
        TableHeader = NULL;
    }
    if (TableSearch) {
        delete TableSearch;
        TableSearch = NULL;
    }
    
    if (Display) {
        delete Display;
    }
    HashFile->PrintAllFoundHashes();
    
    exit(1);
}
