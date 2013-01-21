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

#include "CH_Common/CHHashFilePlain32.h"
#include "Multiforcer_Common/CHCommon.h"


#if USE_NETWORK
#include "Multiforcer_Common/CHNetworkClient.h"
#endif


CHHashFilePlain32::CHHashFilePlain32(int newHashLength) {
    // Programmer error, just bail out.
    if (newHashLength > 32) {
        printf("Error: Cannot use CHHashFilePlain32 for hash types > 32 bytes long!\n");
        exit(1);
    }
    this->hashLength = newHashLength;
    this->outputFoundHashesToFile = 0;
#if USE_NETWORK
    this->NetworkClient = NULL;
#endif
}

CHHashFilePlain32::~CHHashFilePlain32() {

}

int CHHashFilePlain32::OpenHashFile(char *filename){
    FILE *hashfile;
    long int estimated_hash_number;
    char buffer[1024];
    long int i;

    // Estimate number of hashes - this *WILL* be high.
    estimated_hash_number = (file_size(filename) / this->hashLength) + 10; // Add some slack.

    hashfile = fopen(filename, "r");
    if (!hashfile) {
      printf("Cannot open hash file %s.  Exiting.\n", filename);
      exit(1);
    }

    // Allocate new memory.  Return 0 on failure, not an exception.
    this->HashList = new (std::nothrow) Hash32*[estimated_hash_number];
    if (this->HashList == 0) {
        printf("Cannot allocate memory for hash list!\n");
        exit(1);
    }

    for (i = 0; i < estimated_hash_number; i++) {
        this->HashList[i] = new (std::nothrow) Hash32;
        if (this->HashList[i] == 0) {
            printf("Cannot allocate memory for hash list!\n");
            exit(1);
        }
        memset(this->HashList[i], 0, sizeof(Hash32));
    }
    this->TotalHashes = 0;
    this->TotalHashesFound = 0;



    while (!feof(hashfile)) {
      // If fgets returns NULL, there's been an error or eof.  Continue.
      if (!fgets(buffer, 1024, hashfile)) {
        continue;
      }

      // If this is not a full line, continue (usually a trailing crlf)
      if (strlen(buffer) < (this->hashLength * 2) ) {
        continue;
      }
      convertAsciiToBinary(buffer, (unsigned char*)&this->HashList[this->TotalHashes]->hash, this->hashLength);
      this->TotalHashes++;
    }
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = this->TotalHashes;
    this->SortHashList();
    return 1;
}

unsigned char *CHHashFilePlain32::ExportUncrackedHashList(){
    unsigned char *hashListReturn;
    uint64_t i, count;
    uint32_t j;

    this->LockMutex();

    hashListReturn = new unsigned char[this->hashLength * this->TotalHashesRemaining];
    count = 0;

    // Iterate through all the hashes.
    for (i = 0; i < this->TotalHashes; i++) {
        // If the password is found, do NOT export it.
        if (this->HashList[i]->passwordFound) {
            continue;
        }
        // There is *probably* a faster way to do this...
        // Copy the hash into the return list.
        for (j = 0; j < this->hashLength; j++) {
            hashListReturn[count * this->hashLength + j] = this->HashList[i]->hash[j];
        }
        count++;
    }
    this->UnlockMutex();
    return hashListReturn;
}

int CHHashFilePlain32::ReportFoundPassword(unsigned char *Hash, unsigned char *Password){
    // TODO: Optimize this...
    uint64_t i;
    int j;

#if USE_NETWORK
    if (this->NetworkClient) {
        this->submitFoundHashToNetwork(Hash, Password);
        return 1;
    }
#endif

    this->LockMutex();

    for (i = 0; i < this->TotalHashes; i++) {
        if (memcmp(Hash, this->HashList[i]->hash, this->hashLength) == 0) {
            // Only do this if the password is not already reported.
            if (!this->HashList[i]->passwordFound) {
                for (j = 0; j < strlen((const char *)Password); j++) {
                    this->HashList[i]->password[j] = Password[j];
                }
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

void CHHashFilePlain32::PrintAllFoundHashes(){
    uint64_t i;
    int j;

    for (i = 0; i < this->TotalHashes; i++) {
        if (this->HashList[i]->passwordFound) {
            for (j = 0; j < this->hashLength; j++) {
                printf("%02X", this->HashList[i]->hash[j]);
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

void CHHashFilePlain32::PrintNewFoundHashes(){
    uint64_t i;
    int j;

    for (i = 0; i < this->TotalHashes; i++) {
        if ((this->HashList[i]->passwordFound) && (!this->HashList[i]->passwordReported)) {
            for (j = 0; j < this->hashLength; j++) {
                printf("%02X", this->HashList[i]->hash[j]);
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
            this->HashList[i]->passwordReported = 1;
        }
    }
}

void CHHashFilePlain32::SetFoundHashesOutputFilename(char *filename) {
    strncpy(this->outputFilename, filename, 1000);
    this->outputFoundHashesToFile = 1;
    this->outputFile = fopen(filename, "a");
    if (!this->outputFile) {
        printf("Cannot open output file %s\n!", filename);
        exit(1);
    }
}

int CHHashFilePlain32::OutputFoundHashesToFile(){
    uint64_t i;
    int j;

    if (this->outputFoundHashesToFile && this->outputFile) {
        for (i = 0; i < this->TotalHashes; i++) {
            if ((this->HashList[i]->passwordFound) && (!this->HashList[i]->passwordOutputToFile)) {
                for (j = 0; j < this->hashLength; j++) {
                    fprintf(this->outputFile, "%02X", this->HashList[i]->hash[j]);
                }
                fprintf(this->outputFile, ":");
                for (j = 0; j < strlen((const char *)this->HashList[i]->password); j++) {
                        fprintf(this->outputFile, "%c", this->HashList[i]->password[j]);
                }
                if (this->AddHexOutput) {
                    fprintf(this->outputFile, ":0x");
                    for (j = 0; j < strlen((const char *)this->HashList[i]->password); j++) {
                            fprintf(this->outputFile, "%02x", this->HashList[i]->password[j]);
                    }
                }
                fprintf(this->outputFile, "\n");
                this->HashList[i]->passwordOutputToFile = 1;
                fflush(this->outputFile);
            }
        }
        return 1;
    }
    return 0;
}

int CHHashFilePlain32::OutputUnfoundHashesToFile(char *filename) {
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
            for (j = 0; j < this->hashLength; j++) {
                fprintf(UnfoundHashes, "%02X", this->HashList[i]->hash[j]);
            }
            fprintf(UnfoundHashes, "\n");
        }
    }
    // We can do this at the end for speed.
    fflush(this->outputFile);
    return 1;
}

unsigned long CHHashFilePlain32::GetTotalHashCount(){
    return this->TotalHashes;
}
unsigned long CHHashFilePlain32::GetCrackedHashCount(){
    return this->TotalHashesFound;
}
unsigned long CHHashFilePlain32::GetUncrackedHashCount(){
    return this->TotalHashesRemaining;
}

// Hacked up radix sort
void CHHashFilePlain32::SortHashList() {
    uint64_t count[256];
    Hash32** TempHash32List;

    int j;

    uint64_t *cp, *sp, s, c, i;

    TempHash32List = new Hash32*[this->TotalHashes];

    for (j = (this->hashLength - 1); j >= 0; j--) {

        cp = count;
        for (i = 256; i > 0; --i, ++cp)
                *cp = 0;

        for (i = 0; i < this->TotalHashes; i++) {
                count[this->HashList[i]->hash[j]]++;
        }

        s = 0;
        cp = count;
        for (i = 256; i > 0; --i, ++cp) {
                c = *cp;
                *cp = s;
                s += c;
        }

        for (i = 0; i < this->TotalHashes; i++) {
            TempHash32List[count[this->HashList[i]->hash[j]]] = this->HashList[i];
            count[this->HashList[i]->hash[j]]++;
        }
        for (i = 0; i < this->TotalHashes; i++) {
                 this->HashList[i] = TempHash32List[i];
        }
    }
    delete[] TempHash32List;
}


int CHHashFilePlain32::GetHashLength() {
    return this->hashLength;
}


void CHHashFilePlain32::importHashListFromRemoteSystem(unsigned char *hashData, uint32_t numberHashes) {
    //printf("Trying to load %d hashes...\n", numberHashes);

    uint32_t hash, i;


    this->TotalHashes = numberHashes;
    this->TotalHashesRemaining = numberHashes;
    
    this->HashList = new (std::nothrow) Hash32*[numberHashes];
    if (this->HashList == 0) {
        printf("Cannot allocate memory for hash list!\n");
        exit(1);
    }

    for (i = 0; i < numberHashes; i++) {
        this->HashList[i] = new (std::nothrow) Hash32;
        if (this->HashList[i] == 0) {
            printf("Cannot allocate memory for hash list!\n");
            exit(1);
        }
        memset(this->HashList[i], 0, sizeof(Hash32));
    }


    for (hash = 0; hash < numberHashes; hash++) {
        memcpy(this->HashList[hash]->hash, &hashData[hash * this->hashLength], this->hashLength);
    }

    this->SortHashList();

    //printf("Done sorting hash list!\n");

    /*(for (i = 0; i < this->TotalHashes; i++) {
        printf("%d: %02x%02x%02x%02x...\n", i, this->HashList[i]->hash[0], this->HashList[i]->hash[1], this->HashList[i]->hash[2], this->HashList[i]->hash[3]);
    }*/
  
}

#if USE_NETWORK
void CHHashFilePlain32::submitFoundHashToNetwork(unsigned char *Hash, unsigned char *Password) {
    this->NetworkClient->reportNetworkFoundPassword(Hash, Password);
}
#endif