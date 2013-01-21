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


/*
 * This program bulk-converts a bunch of tables from one format to another.
 * Typically, this will be used to go from V1 to V2, or more usefully
 * from V3 to V2. It can also be used to reduce V2 table bits-in-hash.
 * Right now, it will only go to V2 tables...
 *
 * Current use/arguments:
 * GRTBulkConvert
 *   [list of files] - list of input files to convert
 *   -o [output directory] - path to put the files.
 *   --bits [bits] - bits to copy to the output files
 *
 */


#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableHeaderV3.h"
#include "GRT_Common/GRTTableSearchV1.h"
#include "GRT_Common/GRTTableSearchV2.h"
#include "GRT_Common/GRTTableSearchV3.h"
#include "GRT_Common/GRTCommon.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#endif

#include <argtable2.h>

#include <boost/filesystem.hpp>

char silent = 0;

// Converts a V1 or V2 table to a V2 table
void ConvertV1V2ToV2(GRTTableSearch *SourceTable, GRTTableSearch *DestinationTable) {
    uint64_t chainIndex;
    hashPasswordData chainData;

    for (chainIndex = 0; chainIndex < SourceTable->getNumberChains(); chainIndex++) {
        if ((chainIndex % 10000) == 0) {
            printf("Chain %lu / %lu  (%0.2f%%)   \r", chainIndex, SourceTable->getNumberChains(),
                    100.0 * ((float)chainIndex / (float)SourceTable->getNumberChains()));
            fflush(stdout);
        }
        SourceTable->getChainAtIndex(chainIndex, &chainData);
        DestinationTable->writeChain(&chainData);
    }
}

// Converts a V3 table to a V2 table
void ConvertV3ToV2(GRTTableSearch *SourceTable, GRTTableSearch *DestinationTable) {
    uint64_t chainIndex;
    hashPasswordData chainData;
    std::vector<hashPasswordData> tableData;

    // Pre-allocate space for speed reasons
    tableData.reserve(SourceTable->getNumberChains());

    // Get all the source chains
    for (chainIndex = 0; chainIndex < SourceTable->getNumberChains(); chainIndex++) {
        SourceTable->getChainAtIndex(chainIndex, &chainData);
        tableData.push_back(chainData);
    }

    // Sort the chains
    printf("Sorting V3 table chains...\n");
    std::sort(tableData.begin(), tableData.end(), tableDataSortPredicate);

    for (chainIndex = 0; chainIndex < tableData.size(); chainIndex++) {
        DestinationTable->writeChain(&tableData[chainIndex]);
    }


}

int main(int argc, char *argv[]) {

    // Current table we are converting
    uint32_t tableNumber;
    int tableVersion;
    
    GRTTableHeader *SourceTableHeader, *DestinationTableHeader;
    GRTTableSearch *SourceTable, *DestinationTable;

    std::string outputFilename, moveDestinationFilename;


    // Figure out what to convert.
    struct arg_file *table_files = arg_filen(NULL,NULL,"<file>", 0, 100000, "GRT Tables to convert");
    struct arg_str  *output_directory = arg_str1("o", "outputdir", "<path>", "Directory for output files");
    struct arg_int  *bits = arg_int1(NULL, "bits", "<n>", "Bits to copy into the output file");
    struct arg_str  *move_source = arg_str0(NULL, "movesource", "<path>", "Move the source file here after converting successfully");
    struct arg_end  *end = arg_end(20);
    void *argtable[] = {table_files, output_directory, bits, move_source, end};

    // Get arguments, collect data, check for basic errors.
    if (arg_nullcheck(argtable) != 0) {
      printf("error: insufficient memory\n");
    }
    // Look for errors
    int nerrors = arg_parse(argc,argv,argtable);
    if (nerrors > 0) {
      // Print errors, exit.
      arg_print_errors(stdout,end,argv[0]);
      // Print help.
      printf("\n\nOptions: \n");
      arg_print_glossary(stdout,argtable,"  %-20s %s\n");
      exit(1);
    }


    // In theory, we're good to go now!
    printf("Starting bulk convert of %d tables.\n", table_files->count);
    
    // Create the output directory, and if needed, the move-to directory.
    boost::filesystem::create_directory(
        boost::filesystem::path(output_directory->sval[0]));
    if (move_source->count) {
        boost::filesystem::create_directory(
            boost::filesystem::path(move_source->sval[0]));
    }
    
    for (tableNumber = 0; tableNumber < table_files->count; tableNumber++) {
        printf("Converting table %u/%d...\n", tableNumber + 1, table_files->count);
        printf("Source table: %s\n", table_files->filename[tableNumber]);

        tableVersion = getTableVersion(table_files->filename[tableNumber]);

        // Check for flat out invalid tables
        if (tableVersion == -1) {
            printf("Table file is not valid!  Skipping...\n");
            continue;
        }

        // Based on the source table version, do what we need to do.
        switch(tableVersion) {
            case 1:
                SourceTableHeader = new GRTTableHeaderV1();
                SourceTable = new GRTTableSearchV1();
                break;
            case 2:
                SourceTableHeader = new GRTTableHeaderV2();
                SourceTable = new GRTTableSearchV2();
                break;
            case 3:
                SourceTableHeader = new GRTTableHeaderV3();
                SourceTable = new GRTTableSearchV3();
                break;
            default:
                printf("Unsupported table version %d: Continuing.\n", tableVersion);
                continue;
        }

        // Now we have the source tables.  Create the destination tables.
        DestinationTableHeader = new GRTTableHeaderV2();
        DestinationTable = new GRTTableSearchV2();

        // Open the source table.
        SourceTableHeader->readTableHeader(table_files->filename[tableNumber]);
        printf("Source table header:\n");
        SourceTableHeader->printTableHeader();
        SourceTable->setTableHeader(SourceTableHeader);
        SourceTable->SetTableFilename(table_files->filename[tableNumber]);

        boost::filesystem::path sourceTablePath(table_files->filename[tableNumber]);


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
        DestinationTableHeader->setBitsInHash(*bits->ival);

        printf("Destination table header:\n");
        DestinationTableHeader->printTableHeader();
        DestinationTable->setTableHeader(DestinationTableHeader);

        //TODO: Do a better job of creating output filenames here.
        outputFilename = output_directory->sval[0];
        outputFilename += "/";
        outputFilename += sourceTablePath.filename().string();
        outputFilename += "_converted";
        
        if (move_source) {
            moveDestinationFilename = move_source->sval[0];
            moveDestinationFilename += "/";
            moveDestinationFilename += sourceTablePath.filename().string();
        }
                
        printf("Output filename: %s\n", outputFilename.c_str());
        DestinationTable->openOutputFile((char *)outputFilename.c_str());

        if (tableVersion == 3) {
            // Convert a V3 table - more mess.
            printf("Doing V3 convert\n");
            ConvertV3ToV2(SourceTable, DestinationTable);
        } else {
            // Convert a V1 or V2 table - straightforward
            printf("Doing V1 or V2 to convert\n");
            ConvertV1V2ToV2(SourceTable, DestinationTable);
        }

        DestinationTable->closeOutputFile();
        printf("\n\n");
        
        // Move the source file if needed.
        if (move_source->count) {
            boost::filesystem::path destPath(moveDestinationFilename);
            
            boost::filesystem::rename(sourceTablePath, destPath);
        }
        
        // End of loop: Clean up.
        delete SourceTableHeader;
        delete SourceTable;
        delete DestinationTableHeader;
        delete DestinationTable;
    }
    return 0;
}