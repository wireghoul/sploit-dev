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

#include "CH_Common/CHHashFileSalted32.h"
#include "Multiforcer_Common/CHCommon.h"

extern void chomp(char *s);

// Salt length is a max.  Can be longer than actual salt.
CHHashFileSalted32::CHHashFileSalted32(int newHashLength, int newSaltLength, 
        char newSaltIsFirst, char newLiteralSalt) {
    // Programmer error, just bail out.
    if (newHashLength > CHHASHFILESALTED32_MAX_HASH_LENGTH) {
        printf("Error: Cannot use CHHashFileSalted32 for hash types > %d bytes long!\n",
                CHHASHFILESALTED32_MAX_HASH_LENGTH);
        exit(1);
    }
    if (newSaltLength > 64) {
        printf("Error: Cannot use CHHashFileSalted32 for salt types > %d bytes long!\n",
                CHHASHFILESALTED32_MAX_SALT_LENGTH);
        exit(1);
    }
    this->hashLength = newHashLength;
    this->saltLength = newSaltLength;
    this->saltIsFirst = newSaltIsFirst;
    this->literalSalt = newLiteralSalt;
    this->outputFoundHashesToFile = 0;
}

CHHashFileSalted32::~CHHashFileSalted32() {

}

int CHHashFileSalted32::OpenHashFile(char *filename){
    FILE *hashfile;
    long int estimated_hash_number;
    char buffer[1024];
    long int i;
    int colonPos;

    printf("Opening hash file %s\n", filename);

    // Estimate number of hashes - this *WILL* be high.
    // Especially for salted hashes.  Oh well.
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
      // If fgets returns NULL, there's been an error or eof.  Continue.
      if (!fgets(buffer, 1024, hashfile)) {
        continue;
      }

      colonPos = 0;
      for (i = 0; i < strlen(buffer); i++) {
          if (buffer[i] == ':') {
              colonPos = i;
              printf("Found split at %d\n", i);
              break;
          }
      }

      if (strlen(buffer) < this->hashLength) {
          continue;
      }

      // Remove the newlines for length counting.
      chomp(buffer);

      if (this->saltIsFirst) {

          // Check to make sure the hash length is correct.
          // Account for colon.
          if ((strlen(buffer) - (colonPos + 1)) != (this->hashLength * 2)) {
              printf("Hash not correct length! %d\n", (strlen(buffer) - (colonPos + 1)));
              continue;
          }

          // Salt is first in the file
          if (this->literalSalt) {
              // If the salt is literal, the length is the literal length.
              this->HashList[this->TotalHashes]->saltLength = colonPos;
              // And copy the salt in.
              for (i = 0; i < colonPos; i++) {
                  this->HashList[this->TotalHashes]->salt[i] = buffer[i];
              }
          } else {
              // Else, a ascii-hex salt.
              this->HashList[this->TotalHashes]->saltLength = colonPos / 2;
              // Convert the salt and put it in the salt.
              convertAsciiToBinary(buffer, 
                      (unsigned char*)&this->HashList[this->TotalHashes]->salt,
                      this->HashList[this->TotalHashes]->saltLength);
          }
          // Now copy the hash in.
          // Start at the address after the colon
          convertAsciiToBinary(&buffer[colonPos + 1],
                  (unsigned char*)&this->HashList[this->TotalHashes]->hash,
                  this->hashLength);
      } else {
          // Salt is second in the file
          // Copy the hash in first.
          convertAsciiToBinary(buffer,
                  (unsigned char*)&this->HashList[this->TotalHashes]->hash,
                  this->hashLength);
          if (this->literalSalt) {
              // If the salt is literal, the length is the literal length.
              this->HashList[this->TotalHashes]->saltLength = (strlen(buffer) - (colonPos + 1));
              // And copy the salt in.
              for (i = 0; i < this->HashList[this->TotalHashes]->saltLength; i++) {
                  this->HashList[this->TotalHashes]->salt[i] = buffer[colonPos + 1 + i];
              }
          } else {
              // Else, a ascii-hex salt.
              this->HashList[this->TotalHashes]->saltLength = (strlen(buffer) - (colonPos + 1)) / 2;
              // Convert the salt and put it in the salt.
              convertAsciiToBinary(&buffer[colonPos + 1],
                      (unsigned char*)&this->HashList[this->TotalHashes]->salt,
                      this->HashList[this->TotalHashes]->saltLength);
          }

      }

      printf("Salt (len %d): ", this->HashList[this->TotalHashes]->saltLength);
      for (i = 0; i < this->HashList[this->TotalHashes]->saltLength; i++) {
          printf("%02x", this->HashList[this->TotalHashes]->salt[i]);
      }
      printf("\nHash: ");
      for (i = 0; i < this->hashLength; i++) {
          printf("%02x", this->HashList[this->TotalHashes]->hash[i]);
      }
      printf("\n");

      this->TotalHashes++;
    }
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = this->TotalHashes;
    this->SortHashList();
    return 1;
}

unsigned char *CHHashFileSalted32::ExportUncrackedHashList(){
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

int CHHashFileSalted32::ReportFoundPassword(unsigned char *Hash, unsigned char *Password){
    // TODO: Optimize this...
    uint64_t i;
    int j;

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

void CHHashFileSalted32::PrintAllFoundHashes(){
    uint64_t i;
    int j;
    // So we can build them ahead of time and paste them together.
    char saltBuffer[1024];
    char hashBuffer[1024];

    for (i = 0; i < this->TotalHashes; i++) {
        if (this->HashList[i]->passwordFound) {
            memset(saltBuffer, 0, 1024);
            memset(hashBuffer, 0, 1024);
            if (this->literalSalt) { 
                // Copy the literal salt into the buffer.
                strncpy(saltBuffer,
                        (const char *)this->HashList[i]->salt,
                        1024);
            } else {
                // Print the salt in hex.
                for (j = 0; j < this->HashList[i]->saltLength; j++) {
                    sprintf(saltBuffer, "%s%02X", saltBuffer,
                            this->HashList[i]->salt[j]);
                }
            }

            // Build the hash
            for (j = 0; j < this->hashLength; j++) {
                sprintf(hashBuffer, "%s%02X", hashBuffer, this->HashList[i]->hash[j]);
            }
            // And print whatever order we need.
            if (this->saltIsFirst) {
                printf("%s:%s:%s", saltBuffer, hashBuffer, this->HashList[i]->password);
            } else {
                printf("%s:%s:%s", hashBuffer, saltBuffer, this->HashList[i]->password);
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

void CHHashFileSalted32::PrintNewFoundHashes(){
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

void CHHashFileSalted32::SetFoundHashesOutputFilename(char *filename) {
    strncpy(this->outputFilename, filename, 1000);
    this->outputFoundHashesToFile = 1;
    this->outputFile = fopen(filename, "a");
    if (!this->outputFile) {
        printf("Cannot open output file %s\n!", filename);
        exit(1);
    }
}

int CHHashFileSalted32::OutputFoundHashesToFile(){
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

int CHHashFileSalted32::OutputUnfoundHashesToFile(char *filename) {
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

unsigned long CHHashFileSalted32::GetTotalHashCount(){
    return this->TotalHashes;
}
unsigned long CHHashFileSalted32::GetCrackedHashCount(){
    return this->TotalHashesFound;
}
unsigned long CHHashFileSalted32::GetUncrackedHashCount(){
    return this->TotalHashesRemaining;
}

// Hacked up radix sort
void CHHashFileSalted32::SortHashList() {
    uint64_t count[256];
    SaltedHash32** TempHash32List;

    int j;

    uint64_t *cp, *sp, s, c, i;

    TempHash32List = new SaltedHash32*[this->TotalHashes];

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

// Exports the salt list for placing in constant memory
unsigned char *CHHashFileSalted32::GetSaltList(){
    unsigned char *saltListReturn;
    uint64_t i, count;
    int j;
   
    this->LockMutex();

    saltListReturn = new unsigned char[this->TotalHashesRemaining * this->saltLength];
    count = 0;

    // Iterate through all the hashes.
    for (i = 0; i < this->TotalHashes; i++) {
        // If the password is found, do NOT export it.
        if (this->HashList[i]->passwordFound) {
            continue;
        }
        for (j = 0; j < this->saltLength; j++) {
            saltListReturn[count * this->saltLength + j] = this->HashList[i]->salt[j];
        }
        count++;
    }
    this->UnlockMutex();
    return saltListReturn;
}

unsigned char *CHHashFileSalted32::GetSaltLengths() {
    unsigned char *saltLengthReturn;
    uint64_t i, count;

    this->LockMutex();

    saltLengthReturn = new unsigned char[this->TotalHashesRemaining];
    count = 0;

    for (i = 0; i < this->TotalHashes; i++) {
        // If the password is found, do NOT export it.
        if (this->HashList[i]->passwordFound) {
            continue;
        }

        saltLengthReturn[count] = this->HashList[i]->saltLength;
        count++;
    }
    this->UnlockMutex();
    return saltLengthReturn;
}


int CHHashFileSalted32::GetHashLength() {
    return this->hashLength;
}
int CHHashFileSalted32::GetSaltLength() {
    return this->saltLength;
}