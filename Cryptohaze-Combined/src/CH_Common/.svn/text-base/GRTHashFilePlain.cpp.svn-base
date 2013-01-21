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

// This file impelements the GRTHashFilePlain methods.

#include "CH_Common/GRTHashFilePlain.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include "GRT_Common/GRTCommon.h"
#include <new>
#include <algorithm>

extern char silent;


GRTHashFilePlain::GRTHashFilePlain(int newHashLength) {
    // Programmer error, just bail out.
    if (newHashLength > MAX_HASH_LENGTH_BYTES) {
        printf("Error: Cannot use GRTHashFilePlain for hash types > %d bytes long!\n", MAX_HASH_LENGTH_BYTES);
        exit(1);
    }
    this->hashLength = newHashLength;
    this->outputFoundHashesToFile = 0;

    this->TotalHashes = 0;
    this->TotalHashesFound = 0;

    this->AddHexOutput = 0;
}

GRTHashFilePlain::~GRTHashFilePlain() {

}

int GRTHashFilePlain::OpenHashFile(char *filename){
    FILE *hashfile;
    char buffer[1024];

    GRTHash HashToInsert;

    //printf("Opening hash file %s\n", filename);


    hashfile = fopen(filename, "r");
    if (!hashfile) {
      printf("Cannot open hash file %s.  Exiting.\n", filename);
      exit(1);
    }

    while (!feof(hashfile)) {
      // If fgets returns NULL, there's been an error or eof.  Continue.
      if (!fgets(buffer, 1024, hashfile)) {
        continue;
      }

      // If this is not a full line, continue (usually a trailing crlf)
      if (strlen(buffer) < (this->hashLength * 2) ) {
        continue;
      }
      // Clear the structure completely
      memset(&HashToInsert, 0, sizeof(HashToInsert));
      
      convertAsciiToBinary(buffer, (unsigned char*)&HashToInsert.hash, this->hashLength);
      this->HashList.push_back(HashToInsert);
      this->TotalHashes++;
    }
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = this->TotalHashes;


    std::sort(this->HashList.begin(), this->HashList.end(), &GRTHashFilePlain::GRTHashSortPredicate);
    
    return 1;
}

// Return true if d1 is less than d2
bool GRTHashFilePlain::GRTHashSortPredicate(const GRTHash& d1, const GRTHash& d2) {
    int i;
    for (i = 0; i < MAX_HASH_LENGTH_BYTES; i++) {
        if (d1.hash[i] > d2.hash[i]) {
            return 0;
        } else if (d1.hash[i] < d2.hash[i]) {
            return 1;
        }
    }
    // Exactly equal = return 0.
    return 0;
}


unsigned char *GRTHashFilePlain::ExportUncrackedHashList(){
    unsigned char *hashListReturn;
    uint64_t i, count;
    uint32_t j;

    hashListReturn = new unsigned char[this->hashLength * this->TotalHashesRemaining];
    count = 0;

    // Iterate through all the hashes.
    for (i = 0; i < this->TotalHashes; i++) {
        // If the password is found, do NOT export it.
        if (this->HashList[i].passwordFound) {
            continue;
        }
        // There is *probably* a faster way to do this...
        // Copy the hash into the return list.
        for (j = 0; j < this->hashLength; j++) {
            hashListReturn[count * this->hashLength + j] = this->HashList[i].hash[j];
        }
        count++;
    }
    return hashListReturn;
}

int GRTHashFilePlain::ReportFoundPassword(unsigned char *Hash, unsigned char *Password){
    // TODO: Optimize this...
    uint64_t i;
    uint32_t j;
    uint32_t numberPasswordsFound = 0;

    for (i = 0; i < this->TotalHashes; i++) {
        if (memcmp(Hash, this->HashList[i].hash, this->hashLength) == 0) {
            // Only do this if the password is not already reported.
            if (!this->HashList[i].passwordFound) {
                for (j = 0; j < strlen((const char *)Password); j++) {
                    this->HashList[i].password[j] = Password[j];
                }
                this->HashList[i].passwordFound = 1;
                this->TotalHashesFound++;
                this->TotalHashesRemaining--;
                //this->PrintNewFoundHashes();
                // Output to a file if needed.
                this->OutputFoundHashesToFile();
                numberPasswordsFound++;
            }
        }
    }
    return numberPasswordsFound;
}

void GRTHashFilePlain::PrintAllFoundHashes(){
    uint64_t i;
    int j;

    for (i = 0; i < this->TotalHashes; i++) {
        if (this->HashList[i].passwordFound) {
            for (j = 0; j < this->hashLength; j++) {
                printf("%02X", this->HashList[i].hash[j]);
            }
            printf(":");
            for (j = 0; j < strlen((const char *)this->HashList[i].password); j++) {
                    printf("%c", this->HashList[i].password[j]);
            }
            if (this->AddHexOutput) {
                printf(":0x");
                for (j = 0; j < strlen((const char *)this->HashList[i].password); j++) {
                        printf("%02x", this->HashList[i].password[j]);
                }
            }
            printf("\n");
        }
    }
}

void GRTHashFilePlain::PrintNewFoundHashes(){
    uint64_t i;
    int j;

    for (i = 0; i < this->TotalHashes; i++) {
        if (/*!silent &&*/ (this->HashList[i].passwordFound) && (!this->HashList[i].passwordReported)) {
            //printf("\n");
            for (j = 0; j < this->hashLength; j++) {
                printf("%02X", this->HashList[i].hash[j]);
            }
            printf(":");
            for (j = 0; j < strlen((const char *)this->HashList[i].password); j++) {
                    printf("%c", this->HashList[i].password[j]);
            }
            if (this->AddHexOutput) {
                printf(":0x");
                for (j = 0; j < strlen((const char *)this->HashList[i].password); j++) {
                        printf("%02x", this->HashList[i].password[j]);
                }
            }
            printf("\n");
            this->HashList[i].passwordReported = 1;
        }
    }
    fflush(stdout);
}

void GRTHashFilePlain::SetFoundHashesOutputFilename(const char *filename) {
    this->outputFilename = filename;
    this->outputFoundHashesToFile = 1;
    this->outputFile = fopen(filename, "a");
    if (!this->outputFile) {
        printf("Cannot open output file %s\n!", filename);
        exit(1);
    }
}

int GRTHashFilePlain::OutputFoundHashesToFile(){
    uint64_t i;
    int j;

    if (this->outputFoundHashesToFile && this->outputFile) {
        for (i = 0; i < this->TotalHashes; i++) {
            if ((this->HashList[i].passwordFound) && (!this->HashList[i].passwordOutputToFile)) {
                for (j = 0; j < this->hashLength; j++) {
                    fprintf(this->outputFile, "%02X", this->HashList[i].hash[j]);
                }
                fprintf(this->outputFile, ":");
                for (j = 0; j < strlen((const char *)this->HashList[i].password); j++) {
                        fprintf(this->outputFile, "%c", this->HashList[i].password[j]);
                }
                if (this->AddHexOutput) {
                    fprintf(this->outputFile, ":0x");
                    for (j = 0; j < strlen((const char *)this->HashList[i].password); j++) {
                            fprintf(this->outputFile, "%02x", this->HashList[i].password[j]);
                    }
                }
                fprintf(this->outputFile, "\n");
                this->HashList[i].passwordOutputToFile = 1;
                fflush(this->outputFile);
            }
        }
        return 1;
    }
    return 0;
}

int GRTHashFilePlain::OutputUnfoundHashesToFile(char *filename) {
    uint64_t i;
    int j;
    FILE *UnfoundHashes;
    // If we can't open it... oh well?
    UnfoundHashes = fopen(filename, "w");
    if (!UnfoundHashes) {
        return 0;
    }

    for (i = 0; i < this->TotalHashes; i++) {
        if (!this->HashList[i].passwordFound) {
            for (j = 0; j < this->hashLength; j++) {
                fprintf(UnfoundHashes, "%02X", this->HashList[i].hash[j]);
            }
            fprintf(UnfoundHashes, "\n");
        }
    }
    // We can do this at the end for speed.
    fflush(this->outputFile);
    return 1;
}

uint64_t GRTHashFilePlain::GetTotalHashCount(){
    return this->TotalHashes;
}
uint64_t GRTHashFilePlain::GetCrackedHashCount(){
    return this->TotalHashesFound;
}
uint64_t GRTHashFilePlain::GetUncrackedHashCount(){
    return this->TotalHashesRemaining;
}

int GRTHashFilePlain::GetHashLength() {
    return this->hashLength;
}

// Takes a string & adds it as a hash.
int GRTHashFilePlain::AddHashBinaryString(const char *hashString) {
    GRTHash HashToInsert;


    // Clear the structure completely
    memset(&HashToInsert, 0, sizeof(HashToInsert));

    memcpy(HashToInsert.hash, hashString, this->hashLength);
    this->HashList.push_back(HashToInsert);

    this->TotalHashes++;
    this->TotalHashesRemaining = this->TotalHashes;

    std::sort(this->HashList.begin(), this->HashList.end(), &GRTHashFilePlain::GRTHashSortPredicate);
    
    return 1;
}

void GRTHashFilePlain::SetAddHexOutput(char newAddHexOutput) {
    this->AddHexOutput = newAddHexOutput;
}