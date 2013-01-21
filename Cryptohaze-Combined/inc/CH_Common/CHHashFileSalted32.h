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

/* Implements the hash file type for up to 32 byte salted hashes.
 *
 * Max salt and hash length is 32 bytes.  Note that this, if combined,
 * cannot be >55 for most hash types.
 *
 * SaltIsFirst: Set to 1 if the salt is first in the file.
 * LiteralSalt: Set to 1 if the salt is literal in the file, else it will be
 * interpreted as ascii-hex.
 */

#ifndef _CHHASHFILETYPESALTED32_H
#define _CHHASHFILETYPESALTED32_H

#define CHHASHFILESALTED32_MAX_HASH_LENGTH 32
#define CHHASHFILESALTED32_MAX_SALT_LENGTH 64

#include "CH_Common/CHHashFileTypes.h"
#include "Multiforcer_Common/CHCommon.h"


class CHHashFileSalted32 : public CHHashFileTypes {
protected:
    typedef struct SaltedHash32 {
        unsigned char hash[CHHASHFILESALTED32_MAX_HASH_LENGTH];
        unsigned char salt[CHHASHFILESALTED32_MAX_SALT_LENGTH];
        unsigned char password[MAX_PASSWORD_LEN];
        char saltLength;
        char passwordReported;
        char passwordFound;
        char passwordOutputToFile;
    } SaltedHash32;

    SaltedHash32 **HashList;
    uint64_t TotalHashes;
    uint64_t TotalHashesFound;
    uint64_t TotalHashesRemaining;

    void SortHashList();

    int hashLength;
    int saltLength;
    char saltIsFirst;
    char literalSalt;

    // Related to file output
    int outputFoundHashesToFile;
    char outputFilename[1000];
    FILE *outputFile;

public:

    CHHashFileSalted32(int, int, char, char);
    ~CHHashFileSalted32();

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

    unsigned char *GetSaltList();
    unsigned char *GetSaltLengths();


    virtual int GetHashLength();
    virtual int GetSaltLength();
} ;



#endif
