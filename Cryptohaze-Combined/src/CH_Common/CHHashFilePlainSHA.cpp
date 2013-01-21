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

#include "CH_Common/CHHashFilePlainSHA.h"
#include "Multiforcer_Common/CHCommon.h"



// Init the plain hash file type with len 20 - that's what we're using.
CHHashFilePlainSHA::CHHashFilePlainSHA() : CHHashFilePlain32(20) {
}

CHHashFilePlainSHA::~CHHashFilePlainSHA() {

}

int CHHashFilePlainSHA::OpenHashFile(char *filename){
    FILE *hashfile;
    long int estimated_hash_number;
    char buffer[1024];
    long int i;



    //printf("Opening hash file %s\n", filename);


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

      // Check for the {SHA} prefix: If not found, continue
      if ((buffer[0] != '{') || (buffer[1] != 'S') || (buffer[2] != 'H')
              || (buffer[3] != 'A') || (buffer[4] != '}')) {
          continue;
      }

      // If this is not a full line, continue (usually a trailing crlf)
      // 28 is the length of the base64
      // Adding the additional characters for "{SHA}"
      if (strlen(buffer) < (28 + 5)) {
        continue;
      }
      // Decode the base64 magic
      UnBase64(this->HashList[this->TotalHashes]->hash, (unsigned char *)&buffer[5], 28);



      this->TotalHashes++;
    }
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = this->TotalHashes;
    this->SortHashList();
    return 1;
}



void CHHashFilePlainSHA::PrintAllFoundHashes(){
    uint64_t i;
    int j;
    unsigned char buffer[1024];

    for (i = 0; i < this->TotalHashes; i++) {
        if (this->HashList[i]->passwordFound) {
            memset(buffer, 0, 1024);
            Base64(this->HashList[i]->hash, buffer, 20);
            printf("{SHA}");
            for (j = 0; j < 28; j++) {
                printf("%c", buffer[j]);
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

void CHHashFilePlainSHA::PrintNewFoundHashes(){
    uint64_t i;
    int j;
    unsigned char buffer[1024];

    for (i = 0; i < this->TotalHashes; i++) {
        if ((this->HashList[i]->passwordFound) && (!this->HashList[i]->passwordReported)) {
            memset(buffer, 0, 1024);
            Base64(this->HashList[i]->hash, buffer, 20);
            printf("{SHA}");
            for (j = 0; j < 28; j++) {
                printf("%c", buffer[j]);
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


int CHHashFilePlainSHA::OutputFoundHashesToFile(){
    uint64_t i;
    int j;
    unsigned char buffer[1024];
    
    if (this->outputFoundHashesToFile && this->outputFile) {
        for (i = 0; i < this->TotalHashes; i++) {
            if ((this->HashList[i]->passwordFound) && (!this->HashList[i]->passwordOutputToFile)) {
                memset(buffer, 0, 1024);
                Base64(this->HashList[i]->hash, buffer, 20);
                fprintf(this->outputFile, "{SHA}");
                for (j = 0; j < 28; j++) {
                    fprintf(this->outputFile, "%c", buffer[j]);
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
                fflush(this->outputFile);
            }
        }
        return 1;
    }
    return 0;
}

int CHHashFilePlainSHA::OutputUnfoundHashesToFile(char *filename) {
    uint64_t i;
    int j;
    FILE *UnfoundHashes;
    unsigned char buffer[1024];

    // If we can't open it... oh well?
    UnfoundHashes = fopen(filename, "w");
    if (!UnfoundHashes) {
        return 0;
    }

    for (i = 0; i < this->TotalHashes; i++) {
        if (!this->HashList[i]->passwordFound) {
                memset(buffer, 0, 1024);
                Base64(this->HashList[i]->hash, buffer, 20);
                fprintf(UnfoundHashes, "{SHA}");
                for (j = 0; j < 28; j++) {
                    fprintf(UnfoundHashes, "%c", buffer[j]);
                }
            fprintf(UnfoundHashes, "\n");
        }
    }
    // We can do this at the end for speed.
    fflush(UnfoundHashes);
    return 1;
}

