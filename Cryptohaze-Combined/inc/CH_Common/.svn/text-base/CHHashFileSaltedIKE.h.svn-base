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

/* Implements the hash file type for IKE cracking.
 */

#ifndef _CHHASHFILETYPESALTEDIKE_H
#define _CHHASHFILETYPESALTEDIKE_H

//#include "CH_Common/CHHashFileTypes.h"
#include <stdint.h>
#include <vector>
#include <string>
#include <stdio.h>

class CHHashFileSaltedIKE/* : public CHHashFileTypes*/ {
protected:
    typedef struct SaltedHashIKE {
        // Parameters we will pull in from the file.
        std::vector<uint8_t> skeyid_data;
        std::vector<uint8_t> hash_r_data;
        // Target hash we are looking for.
        std::vector<uint8_t> hash_r;
        // The found password if discovered
        std::vector<uint8_t> password;
        char passwordReported;
        char passwordFound;
        char passwordOutputToFile;
    } SaltedHashIKE;

    std::vector<SaltedHashIKE> HashList;

    uint64_t TotalHashes;
    uint64_t TotalHashesFound;
    uint64_t TotalHashesRemaining;

    void SortHashList();

    int hashLength;
    int saltLength;

    // Related to file output
    int outputFoundHashesToFile;
    char outputFilename[1000];
    FILE *outputFile;

public:

    CHHashFileSaltedIKE();
    ~CHHashFileSaltedIKE();

    int OpenHashFile(std::string filename);
/*
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

    unsigned char *GetSaltList();
    unsigned char *GetSaltLengths();


    virtual int GetHashLength();
    virtual int GetSaltLength();*/
} ;

#endif
