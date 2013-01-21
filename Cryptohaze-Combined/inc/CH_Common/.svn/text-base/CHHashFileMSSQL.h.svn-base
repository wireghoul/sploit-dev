/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
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

// Implements the hash file type for MSSQL
/*
 * Hash format: 0x[94 characters]
 * 4/2: 0100 - header
 * 8/4: [salt]
 * 40/20: [Full case SHA1]
 * 40/20: [Uppercase SHA1]
 */

#ifndef _CHHASHFILETYPEMSSQL_H
#define _CHHASHFILETYPEMSSQL_H

#include "CH_Common/CHHashFileTypes.h"
#include "Multiforcer_Common/CHCommon.h"


// Hex-string length with 0x prefix
#define MSSQL_HASH_LENGTH_HEX 96
#define MSSQL_SHA1_LENGTH 20

class CHHashFileMSSQL : public CHHashFileTypes {
private:
    typedef struct HashMSSQL {
        unsigned char hashUppercase[MSSQL_SHA1_LENGTH]; // The uppercase hash segment
        unsigned char hashFullcase[MSSQL_SHA1_LENGTH];  // The full case hash segment
        uint32_t salt;
        unsigned char password[32];
        char passwordReported;
        char passwordFound;
        char passwordOutputToFile;
    } HashMSSQL;

    HashMSSQL **HashList;
    uint64_t TotalHashes;
    uint64_t TotalHashesFound;
    uint64_t TotalHashesRemaining;

    void SortHashList();  // Sorts by the uppercase segment, since this is on the GPU

    // Related to file output
    int outputFoundHashesToFile;
    char outputFilename[1000];
    FILE *outputFile;

public:

    CHHashFileMSSQL();
    ~CHHashFileMSSQL();

    int OpenHashFile(char *filename);

    unsigned char *ExportUncrackedHashList();

    int ReportFoundPassword(unsigned char *Hash, unsigned char *Password);

    void PrintAllFoundHashes();

    void PrintNewFoundHashes();

    void SetFoundHashesOutputFilename(char *filename);
    int OutputFoundHashesToFile();

    int OutputUnfoundHashesToFile(char *filename);

    unsigned long GetTotalHashCount();

    unsigned long GetCrackedHashCount();
    unsigned long GetUncrackedHashCount();

    // MSSQL specific functions
    // Normalize the password
    void NormalizeHash(HashMSSQL *HashToNormalize);

    // Exports the salt list for placing in constant memory
    uint32_t *GetSaltList();

} ;



#endif
