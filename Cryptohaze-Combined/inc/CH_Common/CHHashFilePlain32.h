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

#ifndef _CHHASHFILETYPEPLAIN32_H
#define _CHHASHFILETYPEPLAIN32_H

#include "CH_Common/CHHashFileTypes.h"
#include "Multiforcer_Common/CHCommon.h"

class CHHashFilePlain32 : public CHHashFileTypes {
protected:
    typedef struct Hash32 {
        unsigned char hash[32];
        unsigned char hashProcessed[32];
        unsigned char password[MAX_PASSWORD_LEN];
        char passwordReported;
        char passwordFound;
        char passwordOutputToFile;
    } Hash32;

    Hash32 **HashList;
    uint64_t TotalHashes;
    uint64_t TotalHashesFound;
    uint64_t TotalHashesRemaining;

    virtual void SortHashList();

    int hashLength;

    // Related to file output
    int outputFoundHashesToFile;
    char outputFilename[1000];
    FILE *outputFile;
    
public:

    CHHashFilePlain32(int);
    ~CHHashFilePlain32();

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
