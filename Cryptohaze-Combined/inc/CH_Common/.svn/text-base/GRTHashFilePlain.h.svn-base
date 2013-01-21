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

// Implements the hash file type for simple/plain hashes (MD4/MD5/NTLM/SHA1/etc).
// This is a simplified version of the hash file from the Multiforcer, as the
// other hash file types make no sense for rainbow tables.  This can be extended
// to longer hashes if needed.

#ifndef _GRTHASHFILETYPEPLAIN_H
#define _GRTHASHFILETYPEPLAIN_H

#include <stdio.h>
#include <vector>
#include <string>
#include "GRT_Common/GRTCommon.h"

using namespace std;

// Maximum hash length supported is specified in GRTCommon.h
// Same for maximum password length.


class GRTHashFilePlain {
protected:
    // This is defined here because it is only used internally.
    // Keep track of the hashes we are using.
    typedef struct GRTHash {
        unsigned char hash[MAX_HASH_LENGTH_BYTES];
        unsigned char password[MAX_PASSWORD_LENGTH];
        char passwordReported;
        char passwordFound;
        char passwordOutputToFile;
    } GRTHash;

    // Vector to contain the hashes present.
    vector<GRTHash> HashList;
    uint64_t TotalHashes;
    uint64_t TotalHashesFound;
    uint64_t TotalHashesRemaining;

    // Contains the hash length being used.
    int hashLength;

    // Related to file output
    int outputFoundHashesToFile;
    string outputFilename;
    FILE *outputFile;

    char AddHexOutput;

    static bool GRTHashSortPredicate(const GRTHash &d1, const GRTHash &d2);

public:


    GRTHashFilePlain(int);
    ~GRTHashFilePlain();

    int OpenHashFile(char *filename);

    unsigned char *ExportUncrackedHashList();

    int ReportFoundPassword(unsigned char *Hash, unsigned char *Password);

    void PrintAllFoundHashes();

    void PrintNewFoundHashes();

    int AddHashBinaryString(const char *hashString);

    void SetFoundHashesOutputFilename(const char *filename);
    int OutputFoundHashesToFile();

    int OutputUnfoundHashesToFile(char *filename);

    uint64_t GetTotalHashCount();
    uint64_t GetCrackedHashCount();
    uint64_t GetUncrackedHashCount();

    int GetHashLength();
    void SetAddHexOutput(char);
};

#endif
