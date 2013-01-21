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


#include "GRT_Common/GRTMergeCommandLineData.h"
#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableSearchV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableSearchV2.h"
#include "GRT_Common/GRTHashes.h"
#include "GRT_Common/GRTCommon.h"
#include <vector>
#include <string>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <algorithm>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#endif

#include <boost/filesystem.hpp>

#define DEBUG_OUTPUT_MERGE 0

// Silence output if true.
char silent = 0;


// This is the merge utility.  It should work with all the various options.
// It will sanity check to ensure that the output bits are >= the input bits.


int main(int argc, char *argv[]) {
    
    GRTMergeCommandLineData MergeParams;
    GRTHashes HashNames;
    std::vector<std::string> *filenames;

    char filenameBuffer[1024];

    // Allocate the various tables needed
    GRTTableHeader *TableHeader;
    GRTTableSearch **tableFiles;

    GRTTableHeader *outputHeaderFull = NULL, *outputHeaderPerfect = NULL;
    GRTTableSearch *outputFileFull = NULL, *outputFilePerfect = NULL;

    struct hashPasswordData tableElement, haltElement, previousWrittenElement;

    std::vector<hashPasswordData> tableSegmentToSort;

    uint64_t i;
    uint32_t table;
    // To keep track of where we are starting in the files
    uint64_t *startOffsets;

    uint64_t elementsToReadPerTable = 0;

    char haltFlag = 0;

    uint64_t merged = 0;
    uint64_t totalWrittenFull = 0;
    uint64_t totalWrittenPerfect = 0;

    uint64_t totalInputElements = 0;



    MergeParams.ParseCommandLine(argc, argv);

    printf("Output dir: %s\n", MergeParams.getOutputDirectory().c_str());


    filenames = MergeParams.getListOfFiles();

    // Create the table header based on the first table version

    if (getTableVersion(filenames->at(0).c_str()) == 1) {
        TableHeader = new GRTTableHeaderV1();
    } else if (getTableVersion(filenames->at(0).c_str()) == 2) {
        TableHeader = new GRTTableHeaderV2();
    } else {
        printf("Unknown table version %d in file %s!\n",
                getTableVersion(filenames->at(0).c_str()) == 1, filenames->at(0).c_str());
        exit(1);
    }

    // Load the table header from the first file.
    TableHeader->readTableHeader(filenames->at(0).c_str());


    // Make the output directory if needed
#ifdef _WIN32
    _mkdir(MergeParams.getOutputDirectory().c_str());
#else
    mkdir(MergeParams.getOutputDirectory().c_str(), S_IRWXO | S_IXOTH | S_IRWXU | S_IRWXG);
#endif

    // Theoretically we have a full structure now...
    if (MergeParams.getBuildFull()) {
        // Create the new output files.
        if (MergeParams.getTableVersion() == 1) {
            outputFileFull = new GRTTableSearchV1();
            outputHeaderFull = new GRTTableHeaderV1();
            outputHeaderFull->setTableVersion(1);
        } else if (MergeParams.getTableVersion() == 2) {
            outputFileFull = new GRTTableSearchV2();
            outputHeaderFull = new GRTTableHeaderV2();
            outputHeaderFull->setTableVersion(2);
            // If the user has specified the bits of output, set it.
            if (MergeParams.getBitsOfOutput()) {
                outputHeaderFull->setBitsInHash(MergeParams.getBitsOfOutput());
            } else {
                // For now, we require the user to set this.
                printf("You must set bits in output to use V2 tables!\n");
                exit(1);
            }
        }
        
        // Copy output from the source file to the full output file.
        outputHeaderFull->setChainLength(TableHeader->getChainLength());
        outputHeaderFull->setCharsetCount(TableHeader->getCharsetCount());
        outputHeaderFull->setCharsetLengths(TableHeader->getCharsetLengths());
        outputHeaderFull->setCharset(TableHeader->getCharset());
        outputHeaderFull->setHashName(TableHeader->getHashName());
        outputHeaderFull->setHashVersion(TableHeader->getHashVersion());
        outputHeaderFull->setNumberChains(TableHeader->getNumberChains());
        outputHeaderFull->setPasswordLength(TableHeader->getPasswordLength());
        outputHeaderFull->setTableIndex(TableHeader->getTableIndex());

        // Create the merged table filename.
        sprintf(filenameBuffer, "%s/%s-len%d-idx%d-chr%d-cl%d-v%d-full.grt",
                 MergeParams.getOutputDirectory().c_str(),
                 HashNames.GetHashStringFromId(TableHeader->getHashVersion()),
                 TableHeader->getPasswordLength(),
                 TableHeader->getTableIndex(),
                 TableHeader->getCharsetLengths()[0],
                 TableHeader->getChainLength(),
                 outputHeaderFull->getTableVersion());

         outputFileFull->setTableHeader(outputHeaderFull);
         outputFileFull->openOutputFile(filenameBuffer);

         printf("Writing full table to %s\n", filenameBuffer);
    }
    if (MergeParams.getBuildPerfect()) {
        // Create the new output files.
        // Create the new output files.
        if (MergeParams.getTableVersion() == 1) {
            outputFilePerfect = new GRTTableSearchV1();
            outputHeaderPerfect = new GRTTableHeaderV1();
            outputHeaderFull->setTableVersion(1);
        } else if (MergeParams.getTableVersion() == 2) {
            outputFilePerfect = new GRTTableSearchV2();
            outputHeaderPerfect = new GRTTableHeaderV2();
            outputHeaderPerfect->setTableVersion(2);
            outputHeaderPerfect->setBitsInHash(MergeParams.getBitsOfOutput());
        }

        // Copy output from the source file to the full output file.
        outputHeaderPerfect->setChainLength(TableHeader->getChainLength());
        outputHeaderPerfect->setCharsetCount(TableHeader->getCharsetCount());
        outputHeaderPerfect->setCharsetLengths(TableHeader->getCharsetLengths());
        outputHeaderPerfect->setCharset(TableHeader->getCharset());
        outputHeaderPerfect->setHashName(TableHeader->getHashName());
        outputHeaderPerfect->setHashVersion(TableHeader->getHashVersion());
        outputHeaderPerfect->setNumberChains(TableHeader->getNumberChains());
        outputHeaderPerfect->setPasswordLength(TableHeader->getPasswordLength());
        outputHeaderPerfect->setTableIndex(TableHeader->getTableIndex());

        sprintf(filenameBuffer, "%s/%s-len%d-idx%d-chr%d-cl%d-v%d-perfect.grt",
                 MergeParams.getOutputDirectory().c_str(),
                 HashNames.GetHashStringFromId(TableHeader->getHashVersion()),
                 TableHeader->getPasswordLength(),
                 TableHeader->getTableIndex(),
                 TableHeader->getCharsetLengths()[0],
                 TableHeader->getChainLength(),
                 outputHeaderPerfect->getTableVersion());


        outputFilePerfect->setTableHeader(outputHeaderPerfect);
        outputFilePerfect->openOutputFile(filenameBuffer);
        printf("Writing perfect table to %s\n", filenameBuffer);
    }

    // Allocate the pointers to the table files
    tableFiles = new GRTTableSearch*[filenames->size()];

    // Now build the array of classes.
    for (i = 0; i < filenames->size(); i++) {

        // Create the table files of the appropriate version
        if (getTableVersion(filenames->at(i).c_str()) == 1) {
            tableFiles[i] = new GRTTableSearchV1();
        } else if (getTableVersion(filenames->at(i).c_str()) == 2) {
            tableFiles[i] = new GRTTableSearchV2();
        } else {
            printf("Unknown table version %d in file %s!\n",
                getTableVersion(filenames->at(0).c_str()) == 1, filenames->at(0).c_str());
            exit(1);
        }

        tableFiles[i]->SetTableFilename(filenames->at(i).c_str());
        tableFiles[i]->getChainAtIndex(0, &tableElement);

        // Check to ensure the bits of table are greater than the bits of output.
        if (MergeParams.getTableVersion() == 2) {
            printf("Table %d bits in hash: %d, requested %d\n", i, tableFiles[i]->getBitsInHash(), MergeParams.getBitsOfOutput());
            if (tableFiles[i]->getBitsInHash() < MergeParams.getBitsOfOutput()) {
                printf("Error: Must have bits of output <= bits in all tables.\n");
                printf("Table %s: %d bits of hash.\n", filenames->at(i).c_str(), tableFiles[i]->getBitsInHash());
                printf("Desired output bits: %d\n", MergeParams.getBitsOfOutput());
                exit(1);
            }
        }

        // Add to the total number of chains we are working with.
        totalInputElements += tableFiles[i]->getNumberChains();
    }

    printf("Total elements to merge: %lu\n", totalInputElements);

    // Allocate the start offset array and zero it.
    startOffsets = new uint64_t[filenames->size()];
    memset(startOffsets, 0, sizeof(uint64_t) * filenames->size());

    // Determine the average number of elements to read per table.
    //elementsToReadPerTable = (uint64_t)MergeParams.getMegabytesToUse()
    //        * 1024 * 1024 / sizeof(hashPasswordData) / filenames->size();

    // New calculations for elements to read per table to handle mixed table sizes!
    // How many elements to read per pass - based on size.
    elementsToReadPerTable = (uint64_t)MergeParams.getMegabytesToUse()
            * 1024 * 1024 / sizeof(hashPasswordData);

    if (MergeParams.getDevDebug()) {
        printf("Elements to read total per pass: %d\n", elementsToReadPerTable);
    }
    // Calculate percentage of data to read per pass
    float fractionOfEachTableToRead = (float)elementsToReadPerTable / (float)totalInputElements;

    if (MergeParams.getDevDebug()) {
        printf("Fraction of each table to read: %f\n", fractionOfEachTableToRead);
    }
    // And then figure this out based on the first table
    elementsToReadPerTable = (uint64_t)((float)tableFiles[0]->getNumberChains() * fractionOfEachTableToRead);

    if (MergeParams.getDevDebug()) {
        //elementsToReadPerTable = 20;
        printf("Elements to read for first table: %d\n\n", elementsToReadPerTable);
    }


    // Reserve a bit more memory than we're likely to use to avoid thrashing
    tableSegmentToSort.reserve((1.10 * MergeParams.getMegabytesToUse() * 1024 * 1024) / sizeof(hashPasswordData));

    // Just loop until we're done.
    while(!haltFlag) {

        // Clear the sort segments.
        tableSegmentToSort.clear();
        tableSegmentToSort.reserve(0);
        tableSegmentToSort.reserve((1.10 * MergeParams.getMegabytesToUse() * 1024 * 1024) / sizeof(hashPasswordData));

        // First, we read elements from table 0 to find the halt element.
        // Read the requested number, then continue until a change is found.
        //printf("Reading %d elements from table 0 starting at %d...\n", elementsToReadPerTable, startOffsets[0]);

        for (i = startOffsets[0]; i < min((startOffsets[0] + elementsToReadPerTable), tableFiles[0]->getNumberChains()); i++) {
            // Read an element at the desired offset and push into the sort vector
            tableFiles[0]->getChainAtIndex(i, &tableElement);
            tableSegmentToSort.push_back(tableElement);
            if (DEBUG_OUTPUT_MERGE) {
                printf("pushed [0] %d  %02x%02x%02x%02x...\n", i,
                    tableElement.hash[0], tableElement.hash[1], tableElement.hash[2], tableElement.hash[3]);
            }
        }


        if (DEBUG_OUTPUT_MERGE) {
            printf("Finished reading at hash %d: %02x%02x%02x%02x...\n", i,
                tableElement.hash[0], tableElement.hash[1], tableElement.hash[2], tableElement.hash[3]);
        }

        // If the final count is the final chain, set the target to all 1s.
        if (i >= tableFiles[0]->getNumberChains()) {
            //printf("Table 0 is finished!\n");
            // Set the hash target to all 1s.
            memset(&haltElement, 0xff, sizeof(haltElement));
            haltFlag = 1;
        } else {
            // If not the final element, use the current search point.
            // Store the current hash in the haltElement and read until a change is found.
            haltElement = tableElement;
        }


        // Read the next element.  i is at the last element read + 1.
        tableFiles[0]->getChainAtIndex(i, &tableElement);

        // Now continue reading until a change is found.
        while ((i < tableFiles[0]->getNumberChains()) && (memcmp(haltElement.hash, tableElement.hash, sizeof(tableElement.hash)) == 0)) {
            tableSegmentToSort.push_back(tableElement);
            if (DEBUG_OUTPUT_MERGE) {
                printf("pushed [0] %d  %02x%02x%02x%02x...\n", i,
                    tableElement.hash[0], tableElement.hash[1], tableElement.hash[2], tableElement.hash[3]);
            }
            i++;
            tableFiles[0]->getChainAtIndex(i, &tableElement);
        }
        if (DEBUG_OUTPUT_MERGE || MergeParams.getDevDebug()) {
            printf("table 0 Halted at element %d (%02x%02x%02x%02x...)\n", i,
                tableElement.hash[0], tableElement.hash[1], tableElement.hash[2], tableElement.hash[3]);
        }
        
        startOffsets[0] = i;
        //haltElement = tableElement;

        if (DEBUG_OUTPUT_MERGE) {
            printf("table 0 halt element: (%02x%02x%02x%02x...)\n",
                haltElement.hash[0], haltElement.hash[1], haltElement.hash[2], haltElement.hash[3]);
        }

        // Now, for all the other tables, read until the halt element
        for (table = 1; table < filenames->size(); table++) {
            // Get their start offset.
            i = startOffsets[table];
            if (DEBUG_OUTPUT_MERGE) {printf("Table %d start offset %d\n", table, i);}
            // clear the table element so it does not match initially.
            memset(&tableElement, 0, sizeof(tableElement));
            // While tableElement is less than or equal to the halt element, loop.
            tableFiles[table]->getChainAtIndex(i, &tableElement);

            while ((i < tableFiles[table]->getNumberChains()) && (memcmp(tableElement.hash, haltElement.hash, sizeof(tableElement.hash)) <= 0)) {
                tableSegmentToSort.push_back(tableElement);
                if (DEBUG_OUTPUT_MERGE) {
                    printf("pushed [%d] %d (%02x%02x%02x%02x...)\n",table, i,
                        tableElement.hash[0], tableElement.hash[1], tableElement.hash[2], tableElement.hash[3]);
                }
                i++;
                tableFiles[table]->getChainAtIndex(i, &tableElement);
            }
            printf("\rReading %d/%d (%0.2f %% done)   ", table, filenames->size(),
                (float)(100.0 * ((float)startOffsets[0] / (float)tableFiles[0]->getNumberChains())));
            fflush(stdout);

            if (MergeParams.getDevDebug()) {
                printf("Elements read from table %d: %lu\n", table, i - startOffsets[table]);
            }

            startOffsets[table] = i;
        }

        printf("Sorting %d elements...\n", tableSegmentToSort.size());
        std::sort(tableSegmentToSort.begin(), tableSegmentToSort.end(), tableDataSortPredicate);
        

        // Write the full table stage.
        if (MergeParams.getBuildFull()) {
            printf("Writing full table output...\n");
            for (i = 0; i < tableSegmentToSort.size(); i++) {
                if ((i % 1000000) == 0) {
                    printf("\rWritten %lu / %lu  (%0.2f%%)   ", i, tableSegmentToSort.size(),
                            100.0 * (float)i / (float)tableSegmentToSort.size());
                    fflush(stdout);
                }
                tableElement = tableSegmentToSort.at(i);

                if (!outputFileFull->writeChain(&tableElement)) {
                    printf("Disk write failed: Disk full?\n\n");
                    exit(1);
                }
                totalWrittenFull++;
            }
        }

        if (MergeParams.getBuildPerfect()) {
            printf("\nWriting perfect table output...\n");
            // Clear the previously written element so we write the first one.
            memset(&previousWrittenElement, 0, sizeof(previousWrittenElement));

            for (i = 0; i < tableSegmentToSort.size(); i++) {
                if ((i % 1000000) == 0) {
                    printf("\rWritten %lu / %lu  (%0.2f%%)   ", i, tableSegmentToSort.size(),
                            100.0 * (float)i / (float)tableSegmentToSort.size());
                    fflush(stdout);
                }
                tableElement = tableSegmentToSort.at(i);

                // If the values are not equal, write the chain
                if (memcmp(&tableElement.hash, &previousWrittenElement.hash, 16) != 0) {
                    if (!outputFilePerfect->writeChain(&tableElement)) {
                        printf("Disk write failed: Disk full?\n\n");
                        exit(1);
                    }
                    totalWrittenPerfect++;
                } else {
                    merged++;
                }
                previousWrittenElement = tableElement;
            }
            printf("\n\nWritten: %lu  merged: %lu  (%0.2f%% merged)\n", totalWrittenPerfect, merged,
                100.0 * ((float)merged / ((float)totalWrittenPerfect + (float)merged)));
        }

    }

    // Final fixup of the files if needed, and close them.

    if (MergeParams.getBuildFull()) {
        // Update the total # of chains written.
        outputHeaderFull->setNumberChains(totalWrittenFull);
        // Close the output file, which will rewrite the header.
        outputFileFull->closeOutputFile();
    }

    if (MergeParams.getBuildPerfect()) {
        // Update the total # of chains written.
        outputHeaderPerfect->setNumberChains(totalWrittenFull);
        // Close the output file, which will rewrite the header.
        outputFilePerfect->closeOutputFile();
    }

    // Move the output files if requested
    if (MergeParams.getMoveSortedFiles()) {
        std::string moveToFilePath;
        // Create the move directory if needed
#ifdef _WIN32
        _mkdir(MergeParams.getMoveSortedFilesDirectory().c_str());
#else
        mkdir(MergeParams.getMoveSortedFilesDirectory().c_str(), S_IRWXO | S_IXOTH | S_IRWXU | S_IRWXG);
#endif
        for (i = 0; i < filenames->size(); i++) {
            // Set up the filesystem path for getting the base name
            boost::filesystem::path sourceTablePath(filenames->at(i).c_str());
            moveToFilePath.clear();
            moveToFilePath += MergeParams.getMoveSortedFilesDirectory();
            moveToFilePath += "/";
            moveToFilePath += sourceTablePath.filename().string();
            if (rename(filenames->at(i).c_str(), moveToFilePath.c_str())) {
                // Failure - report this.
                printf("Could not move source file %s!\n", filenames->at(i).c_str());
            }
        }
    }

    return 0;
}