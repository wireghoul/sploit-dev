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

#include "CH_Common/CHHashFileSaltedSSHA.h"
#include "Multiforcer_Common/CHCommon.h"




// Init the plain hash file type with len 20 - that's what we're using.
// Max salt length 16, and the salt is first/literal salt don't matter - we
// override functions that use them.
CHHashFileSaltedSSHA::CHHashFileSaltedSSHA() : CHHashFileSalted32(20, 16, 0, 1) {
}

CHHashFileSaltedSSHA::~CHHashFileSaltedSSHA() {

}

int CHHashFileSaltedSSHA::OpenHashFile(char *filename){
    FILE *hashfile;
    long int estimated_hash_number;
    char buffer[1024];
    unsigned char hexbuffer[1024];
    long int i;
    int decoded_length; // For the length of the base64 decoded output



    // Estimate number of hashes - this *WILL* be high.
    estimated_hash_number = (file_size(filename) / this->hashLength) + 10; // Add some slack.

    hashfile = fopen(filename, "r");
    if (!hashfile) {
      printf("Cannot open hash file %s.  Exiting.\n", filename);
      exit(1);
    }

    // Allocate new memory.  Return 0 on failure, not an exception.
    this->HashList = new (std::nothrow) SaltedHash32*[estimated_hash_number];
    if (this->HashList == 0) {
        printf("Cannot allocate memory for hash list!\n");
        exit(1);
    }

    for (i = 0; i < estimated_hash_number; i++) {
        this->HashList[i] = new (std::nothrow) SaltedHash32;
        if (this->HashList[i] == 0) {
            printf("Cannot allocate memory for hash list!\n");
            exit(1);
        }
        memset(this->HashList[i], 0, sizeof(SaltedHash32));
    }
    this->TotalHashes = 0;
    this->TotalHashesFound = 0;



    while (!feof(hashfile)) {
        memset(buffer, 0, 1024);
      // If fgets returns NULL, there's been an error or eof.  Continue.
      if (!fgets(buffer, 1024, hashfile)) {
        continue;
      }

      // Get rid of any newlines
      chomp(buffer);

      // Check for the {SSHA} prefix: If not found, continue
      if ((buffer[0] != '{') || (buffer[1] != 'S') || (buffer[2] != 'S')
              || (buffer[3] != 'H') || (buffer[4] != 'A') || (buffer[5] != '}')) {
          continue;
      }

      // If this is not a full line, continue (usually a trailing crlf)
      // 28 is the length of the base64
      // Adding the additional characters for "{SHA}"
      if (strlen(buffer) < (28 + 6)) {
        continue;
      }
      memset(hexbuffer, 0, 1024);
      // Decode the base64 magic
      decoded_length = UnBase64(hexbuffer, (unsigned char *)&buffer[6], strlen(buffer) - 6);

      // Set the hash data for SHA1 - 20 bytes
      for (i = 0; i < 20; i++) {
          this->HashList[this->TotalHashes]->hash[i] = hexbuffer[i];
      }
      // Set the salt - anything after the 20 bytes of SHA1
      for (i = 20; i < decoded_length; i++) {
          this->HashList[this->TotalHashes]->salt[i - 20] = hexbuffer[i];
      }
      this->HashList[this->TotalHashes]->saltLength = decoded_length - 20;

      this->TotalHashes++;
    }
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = this->TotalHashes;
    this->SortHashList();
    return 1;
}



void CHHashFileSaltedSSHA::PrintAllFoundHashes(){
    uint64_t i;
    int j;
    unsigned char hexbuffer[1024];
    unsigned char buffer[1024];

    for (i = 0; i < this->TotalHashes; i++) {
        if (this->HashList[i]->passwordFound) {
            memset(buffer, 0, 1024);
            memset(hexbuffer, 0, 1024);

            for (j = 0; j < 20; j++) {
                hexbuffer[j] = this->HashList[i]->hash[j];
            }
            for (j = 0; j < this->HashList[i]->saltLength; j++) {
                hexbuffer[j + 20] = this->HashList[i]->salt[j];
            }

            Base64(hexbuffer, buffer, 20 + this->HashList[i]->saltLength);
            printf("{SSHA}");
            for (j = 0; j < strlen((const char *)buffer); j++) {
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

void CHHashFileSaltedSSHA::PrintNewFoundHashes(){
    uint64_t i;
    int j;
    unsigned char hexbuffer[1024];
    unsigned char buffer[1024];

    for (i = 0; i < this->TotalHashes; i++) {
        if ((this->HashList[i]->passwordFound) && (!this->HashList[i]->passwordReported)) {
            memset(buffer, 0, 1024);
            memset(hexbuffer, 0, 1024);

             for (j = 0; j < 20; j++) {
                hexbuffer[j] = this->HashList[i]->hash[j];
            }
            for (j = 0; j < this->HashList[i]->saltLength; j++) {
                hexbuffer[j + 20] = this->HashList[i]->salt[j];
            }

            Base64(hexbuffer, buffer, 20 + this->HashList[i]->saltLength);
            printf("{SSHA}");
            for (j = 0; j < strlen((const char *)buffer); j++) {
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


int CHHashFileSaltedSSHA::OutputFoundHashesToFile(){
    uint64_t i;
    int j;
    unsigned char buffer[1024];
    unsigned char hexbuffer[1024];

    if (this->outputFoundHashesToFile && this->outputFile) {
        for (i = 0; i < this->TotalHashes; i++) {
            if ((this->HashList[i]->passwordFound) && (!this->HashList[i]->passwordOutputToFile)) {
                memset(buffer, 0, 1024);
                memset(hexbuffer, 0, 1024);

                for (j = 0; j < 20; j++) {
                    hexbuffer[j] = this->HashList[i]->hash[j];
                }
                for (j = 0; j < this->HashList[i]->saltLength; j++) {
                    hexbuffer[j + 20] = this->HashList[i]->salt[j];
                }

                Base64(hexbuffer, buffer, 20 + this->HashList[i]->saltLength);
                fprintf(this->outputFile, "{SSHA}");
                for (j = 0; j < strlen((const char *)buffer); j++) {
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

int CHHashFileSaltedSSHA::OutputUnfoundHashesToFile(char *filename) {
    uint64_t i;
    int j;
    FILE *UnfoundHashes;
    unsigned char buffer[1024];
    unsigned char hexbuffer[1024];

    // If we can't open it... oh well?
    UnfoundHashes = fopen(filename, "w");
    if (!UnfoundHashes) {
        return 0;
    }

    for (i = 0; i < this->TotalHashes; i++) {
        if (!this->HashList[i]->passwordFound) {
                memset(buffer, 0, 1024);
                memset(hexbuffer, 0, 1024);

                for (j = 0; j < 20; j++) {
                    hexbuffer[j] = this->HashList[i]->hash[j];
                }
                for (j = 0; j < this->HashList[i]->saltLength; j++) {
                    hexbuffer[j + 20] = this->HashList[i]->salt[j];
                }

                Base64(hexbuffer, buffer, 20 + this->HashList[i]->saltLength);
                fprintf(UnfoundHashes, "{SSHA}");
                for (j = 0; j < strlen((const char *)buffer); j++) {
                    fprintf(UnfoundHashes, "%c", buffer[j]);
                }
                fprintf(UnfoundHashes, ":");
                for (j = 0; j < strlen((const char *)this->HashList[i]->password); j++) {
                        fprintf(UnfoundHashes, "%c", this->HashList[i]->password[j]);
                }
                fprintf(UnfoundHashes, "\n");
        }
    }
    // We can do this at the end for speed.
    fflush(UnfoundHashes);
    return 1;
}

