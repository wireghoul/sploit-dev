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

#include "CH_Common/CHHashFileMSSQL.h"
#include "Multiforcer_Common/CHCommon.h"
#include "Multiforcer_CUDA_host/CHHashTypeMSSQL_CPU.h"

// Converts a character to lower if requested.
char charToLower(char possibleUpper, char convertToLower) {
    // If it's an uppercase character, return the lowercase version if requested
    if (convertToLower && (possibleUpper >= 0x41) && (possibleUpper <= 0x5a)) {
        return (possibleUpper + 0x20);
    }
    // else just return it as is.
    return possibleUpper;
}


CHHashFileMSSQL::CHHashFileMSSQL() {
    this->outputFoundHashesToFile = 0;
}

CHHashFileMSSQL::~CHHashFileMSSQL() {

}



int CHHashFileMSSQL::OpenHashFile(char *filename){
    FILE *hashfile;
    long int estimated_hash_number;
    char buffer[1024];
    long int i;
    int hash_offset; // For keeping track of where we are parsing the hash
    uint32_t original;


    //printf("Opening hash file %s\n", filename);


    // Estimate number of hashes - this *WILL* be high.
    estimated_hash_number = (file_size(filename) / MSSQL_HASH_LENGTH_HEX) + 10; // Add some slack.

    hashfile = fopen(filename, "r");
    if (!hashfile) {
      printf("Cannot open hash file %s.  Exiting.\n", filename);
      exit(1);
    }

    // Allocate new memory.  Return 0 on failure, not an exception.
    this->HashList = new (std::nothrow) HashMSSQL*[estimated_hash_number];
    if (this->HashList == 0) {
        printf("Cannot allocate memory for hash list!\n");
        exit(1);
    }

    for (i = 0; i < estimated_hash_number; i++) {
        this->HashList[i] = new (std::nothrow) HashMSSQL;
        if (this->HashList[i] == 0) {
            printf("Cannot allocate memory for hash list!\n");
            exit(1);
        }
        memset(this->HashList[i], 0, sizeof(HashMSSQL));
    }
    this->TotalHashes = 0;
    this->TotalHashesFound = 0;



    while (!feof(hashfile)) {
      // If fgets returns NULL, there's been an error or eof.  Continue.
      if (!fgets(buffer, 1024, hashfile)) {
        continue;
      }

      // If this is not a full line, continue (usually a trailing crlf)
      // We subtract 2 just in case there's no "0x" prefix here.
      if (strlen(buffer) < (MSSQL_HASH_LENGTH_HEX - 2) ) {
        continue;
      }

      hash_offset = 0;

      // Check for an 0x prefix and skip it if needed.
      if (buffer[0] == '0' && buffer[1] == 'x') {
          hash_offset += 2;
      }

      // Check for the header (0100).  If not found, it is not a valid hash.
      if ((buffer[hash_offset] == '0') && (buffer[hash_offset + 1] == '1') &&
              (buffer[hash_offset + 2] == '0') && (buffer[hash_offset + 3] == '0')) {
          hash_offset += 4;
      } else {
          printf("Valid MSSQL header not found!\n");
          continue;
      }

      convertAsciiToBinary(&buffer[hash_offset], (unsigned char *)&this->HashList[this->TotalHashes]->salt, 4);
      // Reverse this so it's correct in memory
      original = this->HashList[this->TotalHashes]->salt;
      this->HashList[this->TotalHashes]->salt = ((original >> 24) & 0xff) | (((original >> 16) & 0xff) << 8) |
        (((original >> 8) & 0xff) << 16) | (((original) & 0xff) << 24);

      hash_offset += 8;
      convertAsciiToBinary(&buffer[hash_offset], this->HashList[this->TotalHashes]->hashFullcase, 20);
      hash_offset += 40;
      convertAsciiToBinary(&buffer[hash_offset], this->HashList[this->TotalHashes]->hashUppercase, 20);

      //convertAsciiToBinary(buffer, (unsigned char*)&this->HashList[this->TotalHashes]->hash, this->hashLength);
      this->TotalHashes++;
    }
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = this->TotalHashes;
    this->SortHashList();

    this->PrintAllFoundHashes();
    return 1;
}


// Note: We export the UPPERCASE hash!
unsigned char *CHHashFileMSSQL::ExportUncrackedHashList() {
    unsigned char *hashListReturn;
    uint64_t i, count;
    uint32_t j;
    
    this->LockMutex();

    hashListReturn = new unsigned char[MSSQL_SHA1_LENGTH * this->TotalHashesRemaining];
    count = 0;

    // Iterate through all the hashes.
    for (i = 0; i < this->TotalHashes; i++) {
        // If the password is found, do NOT export it.
        if (this->HashList[i]->passwordFound) {
            continue;
        }
        // There is *probably* a faster way to do this...
        // Copy the hash into the return list.
        for (j = 0; j < MSSQL_SHA1_LENGTH; j++) {
            hashListReturn[count * MSSQL_SHA1_LENGTH + j] = this->HashList[i]->hashUppercase[j];
        }
        count++;
    }
    this->UnlockMutex();
    return hashListReturn;
}

int CHHashFileMSSQL::ReportFoundPassword(unsigned char *Hash, unsigned char *Password){
    // TODO: Optimize this...
    uint64_t i;
    int j;

    this->LockMutex();

    // TODO: Add salt checks, and normalization
    for (i = 0; i < this->TotalHashes; i++) {
        if (memcmp(Hash, this->HashList[i]->hashUppercase, MSSQL_SHA1_LENGTH) == 0) {
            // Only do this if the password is not already reported.
            if (!this->HashList[i]->passwordFound) {
                for (j = 0; j < strlen((const char *)Password); j++) {
                    this->HashList[i]->password[j] = Password[j];
                }
                this->NormalizeHash(((HashMSSQL *)this->HashList[i]));
                this->HashList[i]->passwordFound = 1;
                this->TotalHashesFound++;
                this->TotalHashesRemaining--;
                // Output to a file if needed.
                this->OutputFoundHashesToFile();
                this->UnlockMutex();
                return 1;
            }
        }
    }
    this->UnlockMutex();
    return 0;
}

void CHHashFileMSSQL::PrintAllFoundHashes(){
    uint64_t i;
    int j;

    for (i = 0; i < this->TotalHashes; i++) {
        if (this->HashList[i]->passwordFound) {
            printf("%08X:", this->HashList[i]->salt);
            for (j = 0; j < MSSQL_SHA1_LENGTH; j++) {
                printf("%02X", this->HashList[i]->hashFullcase[j]);
            }
            printf(":");
            for (j = 0; j < strlen((const char *)this->HashList[i]->password); j++) {
                    printf("%c", this->HashList[i]->password[j]);
            }
            if (this->AddHexOutput) {
                printf(":0x");
                for (j = 0; j < strlen((const char *)this->HashList[i]->password); j++) {
                        printf("%02x", this->HashList[i]->password[j]);
                }
            }
            printf("\n");
        }
    }
}

void CHHashFileMSSQL::PrintNewFoundHashes(){

}

void CHHashFileMSSQL::SetFoundHashesOutputFilename(char *filename){
    strncpy(this->outputFilename, filename, 1000);
    this->outputFoundHashesToFile = 1;
    this->outputFile = fopen(filename, "a");
    if (!this->outputFile) {
        printf("Cannot open output file %s\n!", filename);
        exit(1);
    }
}
int CHHashFileMSSQL::OutputFoundHashesToFile(){
    uint64_t i;
    int j;
    if (this->outputFoundHashesToFile && this->outputFile) {
        for (i = 0; i < this->TotalHashes; i++) {
            if (this->HashList[i]->passwordFound && (!this->HashList[i]->passwordOutputToFile)) {
                fprintf(this->outputFile, "%08X:", this->HashList[i]->salt);
                for (j = 0; j < MSSQL_SHA1_LENGTH; j++) {
                    fprintf(this->outputFile, "%02X", this->HashList[i]->hashFullcase[j]);
                }
                fprintf(this->outputFile, ":");
                for (j = 0; j < strlen((const char *)this->HashList[i]->password); j++) {
                        fprintf(this->outputFile, "%c", this->HashList[i]->password[j]);
                }
                if (this->AddHexOutput) {
                    fprintf(this->outputFile,":0x");
                    for (j = 0; j < strlen((const char *)this->HashList[i]->password); j++) {
                            fprintf(this->outputFile,"%02x", this->HashList[i]->password[j]);
                    }
                }
                fprintf(this->outputFile, "\n");
                this->HashList[i]->passwordOutputToFile = 1;
            }
        }
        fflush(this->outputFile);
        return 1;
    }
    return 0;
}

int CHHashFileMSSQL::OutputUnfoundHashesToFile(char *filename){
    uint64_t i;
    int j;
    FILE *UnfoundHashes;

    // If we can't open it... oh well?
    UnfoundHashes = fopen(filename, "w");
    if (!UnfoundHashes) {
        return 0;
    }

    for (i = 0; i < this->TotalHashes; i++) {
        if (!this->HashList[i]->passwordFound) {
            fprintf(UnfoundHashes, "0x0100");
            fprintf(UnfoundHashes, "%08X", this->HashList[i]->salt);
            for (j = 0; j < MSSQL_SHA1_LENGTH; j++) {
                fprintf(UnfoundHashes, "%02X", this->HashList[i]->hashFullcase[j]);
            }
            for (j = 0; j < MSSQL_SHA1_LENGTH; j++) {
                fprintf(UnfoundHashes, "%02X", this->HashList[i]->hashUppercase[j]);
            }
            fprintf(UnfoundHashes, "\n");
        }
    }

    // We can do this at the end for speed.
    fflush(this->outputFile);
    return 1;
}

unsigned long CHHashFileMSSQL::GetTotalHashCount(){
    return this->TotalHashes;
}

unsigned long CHHashFileMSSQL::GetCrackedHashCount(){
    return this->TotalHashesFound;
}
unsigned long CHHashFileMSSQL::GetUncrackedHashCount(){
    return this->TotalHashesRemaining;
}

// MSSQL specific functions
// Normalize the password
void CHHashFileMSSQL::NormalizeHash(HashMSSQL *HashToNormalize){
    uint32_t passwordNormalizeValues = 0, passwordNormalizeMax = 0;
    char testPassword[MAX_PASSWORD_LEN];
    int passwordLength;
    int passwordPosition;
    int i;

    passwordLength = strlen((const char *)HashToNormalize->password);

    passwordNormalizeMax = pow((float)2.0, passwordLength);

    // Loop through all possible mutations
    for (passwordNormalizeValues = 0; passwordNormalizeValues < passwordNormalizeMax; passwordNormalizeValues++) {
        memset(testPassword, 0, MAX_PASSWORD_LEN);
        // Iterate through the password length
        for (passwordPosition = 0; passwordPosition <= passwordLength; passwordPosition++) {
            testPassword[passwordPosition] = charToLower(HashToNormalize->password[passwordPosition],
                    passwordNormalizeValues & (1 << passwordPosition));
        }
        if (CPU_MSSQL(testPassword, HashToNormalize->salt, HashToNormalize->hashFullcase)) {
            for (i = 0; i < passwordLength; i++) {
                HashToNormalize->password[i] = testPassword[i];
            }
        }
    }

}

// Exports the salt list for placing in constant memory
uint32_t *CHHashFileMSSQL::GetSaltList(){
    uint32_t *saltListReturn;
    uint64_t i, count;
    uint32_t original, reversed;

    saltListReturn = new uint32_t[this->TotalHashesRemaining];
    count = 0;

    // Iterate through all the hashes.
    for (i = 0; i < this->TotalHashes; i++) {
        // If the password is found, do NOT export it.
        if (this->HashList[i]->passwordFound) {
            continue;
        }
        // There is *probably* a faster way to do this...
        // Copy the hash into the return list.
        // First, reverse the ordering to make this faster on CUDA
        //saltListReturn[count] = this->HashList[i]->salt;
        original = this->HashList[i]->salt;
        saltListReturn[count] = ((original >> 24) & 0xff) | (((original >> 16) & 0xff) << 8) |
                (((original >> 8) & 0xff) << 16) | (((original) & 0xff) << 24);
        count++;
    }
    return saltListReturn;
}

// Woo radix sort!
void CHHashFileMSSQL::SortHashList() {
    uint64_t count[256];
    HashMSSQL** TempHashMSSQLList;

    int j;

    uint64_t *cp, s, c, i;

    TempHashMSSQLList = new HashMSSQL*[this->TotalHashes];

    for (j = (MSSQL_SHA1_LENGTH - 1); j >= 0; j--) {

        cp = count;
        for (i = 256; i > 0; --i, ++cp)
                *cp = 0;

        for (i = 0; i < this->TotalHashes; i++) {
                count[this->HashList[i]->hashUppercase[j]]++;
        }

        s = 0;
        cp = count;
        for (i = 256; i > 0; --i, ++cp) {
                c = *cp;
                *cp = s;
                s += c;
        }

        for (i = 0; i < this->TotalHashes; i++) {
            TempHashMSSQLList[count[this->HashList[i]->hashUppercase[j]]] = this->HashList[i];
            count[this->HashList[i]->hashUppercase[j]]++;
        }
        for (i = 0; i < this->TotalHashes; i++) {
                 this->HashList[i] = TempHashMSSQLList[i];
        }
    }
    delete[] TempHashMSSQLList;
}