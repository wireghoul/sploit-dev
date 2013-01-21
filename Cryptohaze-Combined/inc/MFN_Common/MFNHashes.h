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

#ifndef _MFNHASH_H
#define _MFNHASH_H

#include "Multiforcer_Common/CHCommon.h"




// Define the various hash types we are using.

// The Plain MD5 class of hashes: 0-19
#define CH_HASH_TYPE_MD5_PLAIN       0
#define CH_HASH_TYPE_MD5_SINGLE      1
#define CH_HASH_TYPE_DOUBLE_MD5      2
#define CH_HASH_TYPE_TRIPLE_MD5      3
#define CH_HASH_TYPE_DUPLICATED_MD5  4
#define CH_HASH_TYPE_MD5_PASS_SALT   5
#define CH_HASH_TYPE_MD5_SALT_PASS   6

// The Windows class of hashes: 20-29
#define CH_HASH_TYPE_NTLM            20
#define CH_HASH_TYPE_DUPLICATED_NTLM 21
#define CH_HASH_TYPE_LM              22

// The SHA class of hashes: 30-39
#define CH_HASH_TYPE_SHA1_PLAIN      30
#define CH_HASH_TYPE_SHA             31
#define CH_HASH_TYPE_SSHA            32
#define CH_HASH_TYPE_MSSQL           33
#define CH_HASH_TYPE_SHA256          34
#define CH_HASH_TYPE_SL3             35

// Other stuff: 40-49
#define CH_HASH_TYPE_MYSQL323        40
#define CH_HASH_TYPE_MD5_OF_SHA1     41
#define CH_HASH_TYPE_SHA1_OF_MD5     42
#define CH_HASH_TYPE_MD4_PLAIN       43

// Defined in CHCommon - currently 100
#define MAX_HASH_ID_VALUE MAX_HASH_TYPES




// This contains data about the various hash types.
// Alignment on this doesn't matter, but probably isn't pretty.
typedef struct CHHashTypeData {
    // This is the descriptive string for the hash.  MD5, NTLM, etc.
    char HashString[MAX_HASH_STRING_LENGTH];
    // This is a text description of the hash, notes, etc.
    char HashDescription[MAX_HASH_DESCRIPTION_LENGTH];
    // This is an algorithm for the hash if needed to clarify.
    char HashAlgorithm[MAX_HASH_ALGORITHM_LENGTH];
    // Minimum and maximum supported lengths
    uint8_t MinSupportedLength;
    uint8_t MaxSupportedLength;
    // Network support functioning or not?
    uint8_t NetworkSupportEnabled;
    // Default workunit size, in bits.
    uint8_t DefaultWorkunitSize;
    // Maximum number of hashes supported by this algorithm.
    // 0: Unlimited
    uint32_t MaxHashCount;
    
} CHHashTypeData;

class CHHashesV {
private:
    int CurrentHashId;
    struct CHHashTypeData HashTypes[MAX_HASH_TYPES];
    //char Hashes[MAX_HASH_TYPES][MAX_HASH_STRING_LENGTH];
    int NumberOfHashes;
public:
    // Basic functions to support hashes
    int GetHashIdFromString(const char* HashString);
    void SetHashId(int);
    int GetNumberOfHashes();
    char *GetHashStringFromID(int);
    int GetHashId();
    // Returns 0 if hash is not set.
    uint8_t GetMinSupportedLength();
    uint8_t GetMaxSupportedLength();
    uint8_t GetIsNetworkSupported();
    uint8_t GetDefaultWorkunitSizeBits();
    uint32_t GetMaxHashCount();

    // Returns the appropriate hash type/hash file type for the hash.
    CHHashFileTypes *GetHashFile();
    CHHashType *GetHashType();

    void PrintAllHashTypes();

    CHHashesV();
};




#endif
