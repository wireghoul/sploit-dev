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

// Implements the hash file type for up to 32 byte byte hashes (MD4/MD5/NTLM/SHA1/etc).

#ifndef _CHHASHFILETYPELM_H
#define _CHHASHFILETYPELM_H

#include "CH_Common/CHHashFileTypes.h"
#include "Multiforcer_Common/CHCommon.h"

using namespace std;

#include <vector>

class CHHashFileLM : public CHHashFileTypes {
protected:
    // This structure contains the full hash data - both pieces.
    // This is used to report out passwords as desired.
    typedef struct LMFullHashData {
        unsigned char fullHash[32];
        unsigned char password[MAX_PASSWORD_LEN];
        char passwordReported;
        char passwordPart1Inserted; // Part 1 is inserted
        char passwordPart2Inserted; // Part 2 is inserted
        char passwordPart1IsNull; // Part 1 is null
        char passwordPart2IsNull; // Part 2 is null
        char passwordFound;
        char passwordOutputToFile;
    } LMFullHashData;

    // This structure contains each half of the hashes.  As the system
    // cracks the half-hashes, this is what actually gets fed out to the
    // rest of the cracking framework.
    typedef struct LMFragmentHashData {
        unsigned char halfHash[16];
        unsigned char password[8]; // Null termination space
        char passwordReported;
        char passwordFound;
        char passwordOutputToFile;
    } LMFragmentHashData;


    std::vector<LMFullHashData> HashList;
    std::vector<LMFragmentHashData> halfHashList;

    // This counts the fragments - not the full ones.
    // We only care about the fragments, technically.
    uint64_t TotalHashes;
    uint64_t TotalHashesFound;
    uint64_t TotalHashesRemaining;

    virtual void SortHashList();
    static bool LMHalfDataSortPredicate(const LMFragmentHashData &d1, const LMFragmentHashData &d2);
    static bool halfHashDataUniquePredicate(const LMFragmentHashData &d1, const LMFragmentHashData &d2);

    void MergeHalfPartsIntoFullPasswords();

    int hashLength;

    // Related to file output
    int outputFoundHashesToFile;
    char outputFilename[1000];
    FILE *outputFile;

public:

    CHHashFileLM();
    ~CHHashFileLM();

    virtual int OpenHashFile(char *filename);

    virtual unsigned char *ExportUncrackedHashList();

    virtual int ReportFoundPassword(unsigned char *Hash, unsigned char *Password);

    virtual void PrintAllFoundHashes();

    virtual void PrintNewFoundHashes();

    virtual void SetFoundHashesOutputFilename(char *filename);
    virtual int OutputFoundHashesToFile();

    virtual int OutputUnfoundHashesToFile(char *filename);

    virtual unsigned long GetTotalHashCount();

    virtual unsigned long GetCrackedHashCount();
    virtual unsigned long GetUncrackedHashCount();

    virtual int GetHashLength();

    virtual void importHashListFromRemoteSystem(unsigned char *hashData, uint32_t numberHashes);

#if USE_NETWORK
    virtual void submitFoundHashToNetwork(unsigned char *Hash, unsigned char *Password);
#endif

} ;

#endif
