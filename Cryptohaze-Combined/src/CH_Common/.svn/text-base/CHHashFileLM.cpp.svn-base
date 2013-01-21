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

#include "CH_Common/CHHashFileLM.h"
#include "Multiforcer_Common/CHCommon.h"


#if USE_NETWORK
#include "Multiforcer_Common/CHNetworkClient.h"
#endif

bool CHHashFileLM::halfHashDataUniquePredicate(const LMFragmentHashData &d1, const LMFragmentHashData &d2) {
    // If they're equal, return true.
    if (memcmp(d1.halfHash, d2.halfHash, 8) == 0) {
        return 1;
    }
    return 0;
}

CHHashFileLM::CHHashFileLM() {
    // Hash length is 8 bytes long for each half.
    this->hashLength = 8;
    this->outputFoundHashesToFile = 0;
    this->TotalHashes = 0;
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = 0;

#if USE_NETWORK
    this->NetworkClient = NULL;
#endif
}

CHHashFileLM::~CHHashFileLM() {

}

int CHHashFileLM::OpenHashFile(char *filename){
    FILE *hashfile;
    char buffer[1024];
    int i;

    // This is the null hash - often seen in the second part.
    const unsigned char nullHash[8] = {0xAA, 0xD3, 0xB4, 0x35, 0xB5, 0x14, 0x04, 0xEE};

    LMFullHashData FullHash;
    LMFragmentHashData HalfHash;

    memset(&FullHash, 0, sizeof(LMFullHashData));
    memset(&HalfHash, 0, sizeof(LMFragmentHashData));

    //printf("Opening hash file %s\n", filename);

    hashfile = fopen(filename, "r");
    if (!hashfile) {
      printf("Cannot open hash file %s.  Exiting.\n", filename);
      exit(1);
    }

    this->TotalHashes = 0;
    this->TotalHashesFound = 0;

    while (!feof(hashfile)) {
      // If fgets returns NULL, there's been an error or eof.  Continue.
      if (!fgets(buffer, 1000, hashfile)) {
        continue;
      }

      // If this is not a full line, continue (usually a trailing crlf)
      if (strlen(buffer) < 32 ) {
        continue;
      }

      // Load the full hash into the list
      memset(&FullHash, 0, sizeof(LMFullHashData));
      convertAsciiToBinary(buffer, (unsigned char*)FullHash.fullHash, 32);
      for (i = 0; i < 14; i++) {
          FullHash.password[i] = 'x';
      }
      this->HashList.push_back(FullHash);

      // If the first half is not the null hash, add it.
      if (memcmp(&FullHash.fullHash[0], nullHash, 8)) {
          memset(&HalfHash, 0, sizeof(LMFragmentHashData));
          memcpy(HalfHash.halfHash, &FullHash.fullHash[0], 8);
          this->halfHashList.push_back(HalfHash);
      }

      if (memcmp(&FullHash.fullHash[8], nullHash, 8)) {
          memset(&HalfHash, 0, sizeof(LMFragmentHashData));
          memcpy(HalfHash.halfHash, &FullHash.fullHash[8], 8);
          this->halfHashList.push_back(HalfHash);
      }
    }

	// Sort removes uniques... do this first.
    this->SortHashList();
    this->TotalHashes = this->halfHashList.size();
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = this->TotalHashes;
    return 1;
}

unsigned char *CHHashFileLM::ExportUncrackedHashList(){
    unsigned char *hashListReturn;
    uint64_t i, count;
    uint32_t j;

    this->LockMutex();
    
    hashListReturn = new unsigned char[this->hashLength * (this->TotalHashesRemaining + 5)];
    count = 0;

    // Iterate through all the hashes.
    for (i = 0; i < this->TotalHashes; i++) {
        // If the password is found, do NOT export it.
        if (this->halfHashList[i].passwordFound) {
            continue;
        }
        // There is *probably* a faster way to do this...
        // Copy the hash into the return list.
        for (j = 0; j < this->hashLength; j++) {
            hashListReturn[count * this->hashLength + j] = this->halfHashList[i].halfHash[j];
        }
        count++;
    }
    this->UnlockMutex();
    return hashListReturn;
}

int CHHashFileLM::ReportFoundPassword(unsigned char *Hash, unsigned char *Password){
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
        if (memcmp(Hash, this->halfHashList[i].halfHash, this->hashLength) == 0) {
            // Only do this if the password is not already reported.
            if (!this->halfHashList[i].passwordFound) {
                for (j = 0; j < strlen((const char *)Password); j++) {
                    this->halfHashList[i].password[j] = Password[j];
                }
                this->halfHashList[i].passwordFound = 1;
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

void CHHashFileLM::PrintAllFoundHashes(){
    uint64_t i;
    int j;

    this->MergeHalfPartsIntoFullPasswords();

    for (i = 0; i < this->HashList.size(); i++) {
        // Print hashes in the following:
        // - Either part 1 or part 2 is found
        // - If part1 is found & part 2 is null, print
        if ((this->HashList[i].passwordPart1Inserted || this->HashList[i].passwordPart2Inserted) &&
                !(this->HashList[i].passwordPart2IsNull && !this->HashList[i].passwordPart1Inserted)) {
            for (j = 0; j < (this->hashLength * 2); j++) {
                printf("%02X", this->HashList[i].fullHash[j]);
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
    this->OutputFoundHashesToFile();
}

void CHHashFileLM::PrintNewFoundHashes(){
    uint64_t i;
    int j;

    for (i = 0; i < this->HashList.size(); i++) {
        if ((this->HashList[i].passwordFound) && (!this->HashList[i].passwordReported)) {
            for (j = 0; j < (this->hashLength * 2); j++) {
                printf("%02X", this->HashList[i].fullHash[j]);
            }
            printf(":");
            for (j = 0; j < strlen((const char *)this->HashList[i].password); j++) {
                    printf("%c", this->HashList[i].password[j]);
            }
            printf("\n");
            this->HashList[i].passwordReported = 1;
        }
    }
}

void CHHashFileLM::SetFoundHashesOutputFilename(char *filename) {
    strncpy(this->outputFilename, filename, 1000);
    this->outputFoundHashesToFile = 1;
    this->outputFile = fopen(filename, "a");
    if (!this->outputFile) {
        printf("Cannot open output file %s\n!", filename);
        exit(1);
    }
}

int CHHashFileLM::OutputFoundHashesToFile() {
    uint64_t i;
    int j;

    if (this->outputFoundHashesToFile && this->outputFile) {
        for (i = 0; i < this->HashList.size(); i++) {
            if ((this->HashList[i].passwordPart1Inserted || this->HashList[i].passwordPart2Inserted) &&
                    !(this->HashList[i].passwordPart2IsNull && !this->HashList[i].passwordPart1Inserted)) {
                for (j = 0; j < (this->hashLength * 2); j++) {
                    fprintf(this->outputFile, "%02X", this->HashList[i].fullHash[j]);
                }
                fprintf(this->outputFile, ":");
                for (j = 0; j < strlen((const char *) this->HashList[i].password); j++) {
                    fprintf(this->outputFile, "%c", this->HashList[i].password[j]);
                }
                if (this->AddHexOutput) {
                    fprintf(this->outputFile,":0x");
                    for (j = 0; j < strlen((const char *)this->HashList[i].password); j++) {
                            fprintf(this->outputFile,"%02x", this->HashList[i].password[j]);
                    }
                }
                fprintf(this->outputFile, "\n");
                this->HashList[i].passwordOutputToFile = 1;
            }
        }
        return 1;
    }
    return 0;
}

int CHHashFileLM::OutputUnfoundHashesToFile(char *filename) {
    uint64_t i;
    int j;
    
    FILE *UnfoundHashes;
    // If we can't open it... oh well?
    UnfoundHashes = fopen(filename, "w");
    if (!UnfoundHashes) {
        return 0;
    }

    for (i = 0; i < this->HashList.size(); i++) {
        if (!this->HashList[i].passwordFound) {
            for (j = 0; j < (this->hashLength * 2); j++) {
                fprintf(UnfoundHashes, "%02X", this->HashList[i].fullHash[j]);
            }
            fprintf(UnfoundHashes, "\n");
        }
    }
    // We can do this at the end for speed.
    fflush(this->outputFile);
    return 1;
}

unsigned long CHHashFileLM::GetTotalHashCount(){
    return this->TotalHashes;
}
unsigned long CHHashFileLM::GetCrackedHashCount(){
    return this->TotalHashesFound;
}
unsigned long CHHashFileLM::GetUncrackedHashCount(){
    return this->TotalHashesRemaining;
}

// Hacked up radix sort
void CHHashFileLM::SortHashList() {
    std::sort(this->halfHashList.begin(), this->halfHashList.end(), CHHashFileLM::LMHalfDataSortPredicate);
    this->halfHashList.erase(
        unique( this->halfHashList.begin(), this->halfHashList.end(), halfHashDataUniquePredicate ),
        this->halfHashList.end() );
}


int CHHashFileLM::GetHashLength() {
    return this->hashLength;
}


void CHHashFileLM::importHashListFromRemoteSystem(unsigned char *hashData, uint32_t numberHashes) {
    //printf("Trying to load %d hashes...\n", numberHashes);

    uint32_t hash;
    LMFragmentHashData incomingHalfHash;



    memset(&incomingHalfHash, 0, sizeof(LMFragmentHashData));

    for (hash = 0; hash < numberHashes; hash++) {
        memset(&incomingHalfHash, 0, sizeof(LMFragmentHashData));
        memcpy(incomingHalfHash.halfHash, &hashData[hash * this->hashLength], this->hashLength);
        this->halfHashList.push_back(incomingHalfHash);
    }

    this->SortHashList();
	this->TotalHashes = this->halfHashList.size();
    this->TotalHashesRemaining = this->halfHashList.size();

    //printf("Done sorting hash list!\n");
}

#if USE_NETWORK
void CHHashFileLM::submitFoundHashToNetwork(unsigned char *Hash, unsigned char *Password) {
    this->NetworkClient->reportNetworkFoundPassword(Hash, Password);
}
#endif

// Return true if d1 is less than d2
bool CHHashFileLM::LMHalfDataSortPredicate(const LMFragmentHashData &d1, const LMFragmentHashData &d2) {
    int i;
    for (i = 0; i < 8; i++) {
        if (d1.halfHash[i] == d2.halfHash[i]) {
            continue;
        } else if (d1.halfHash[i] > d2.halfHash[i]) {
            return 0;
        } else if (d1.halfHash[i] < d2.halfHash[i]) {
            return 1;
        }
    }
    // Exactly equal = return 0.
    return 0;
}

// Merge the half hashes back into the full passwords
void CHHashFileLM::MergeHalfPartsIntoFullPasswords() {
    int i, j;
    const unsigned char nullHash[8] = {0xAA, 0xD3, 0xB4, 0x35, 0xB5, 0x14, 0x04, 0xEE};
    LMFragmentHashData SearchValue;
    pair<vector<LMFragmentHashData>::iterator,vector<LMFragmentHashData>::iterator> valueBounds;

    // Loop through all the "full" hashes
    // Be a bit more intelligent to avoid taking all eternity...
    for (i = 0; i < this->HashList.size(); i++) {
        // If the password has been filled in, continue - no point in checking.
        if (this->HashList.at(i).passwordFound) {
            continue;
        }

        // If the first part is not updated, try for it.
        if (!this->HashList.at(i).passwordPart1Inserted) {
            // Check for empty hashes.
            if (memcmp(&this->HashList.at(i).fullHash[0], nullHash, 8) == 0) {
                memset(&this->HashList.at(i).password[0], 0, 7);
                this->HashList.at(i).passwordPart1Inserted = 1;
                this->HashList.at(i).passwordPart1IsNull = 1;
            } else {
                // Copy the value we're looking for into the search variable.
                memcpy(SearchValue.halfHash, &this->HashList.at(i).fullHash[0], 8);
                valueBounds = equal_range(this->halfHashList.begin(), this->halfHashList.end(), 
                        SearchValue, CHHashFileLM::LMHalfDataSortPredicate);

                // Start loop
                for (j = valueBounds.first - this->halfHashList.begin(); j < valueBounds.second - this->halfHashList.begin(); j++) {
                    // Check for a match.  If it matches, copy the password bit in.
                    if (this->halfHashList.at(j).passwordFound &&
                            (memcmp(this->halfHashList.at(j).halfHash, &this->HashList.at(i).fullHash[0], 8) == 0)) {
                        memcpy(&this->HashList.at(i).password[0], this->halfHashList.at(j).password, 7);
                        this->HashList.at(i).passwordPart1Inserted = 1;
                    }
                } // End loop
            }
        }

        // If the second part is not updated, try for it.
        if (!this->HashList.at(i).passwordPart2Inserted) {
            // Check for empty hashes.
            if (memcmp(&this->HashList.at(i).fullHash[8], nullHash, 8) == 0) {
                memset(&this->HashList.at(i).password[7], 0, 7);
                this->HashList.at(i).passwordPart2Inserted = 1;
                this->HashList.at(i).passwordPart2IsNull = 1;
            } else {
                // Copy the value we're looking for into the search variable.
                memcpy(SearchValue.halfHash, &this->HashList.at(i).fullHash[8], 8);
                valueBounds = equal_range(this->halfHashList.begin(), this->halfHashList.end(),
                        SearchValue, CHHashFileLM::LMHalfDataSortPredicate);

                for (j = valueBounds.first - this->halfHashList.begin(); j < valueBounds.second - this->halfHashList.begin(); j++) {
                    // Check for a match.  If it matches, copy the password bit in.
                    if (this->halfHashList.at(j).passwordFound &&
                            (memcmp(this->halfHashList.at(j).halfHash, &this->HashList.at(i).fullHash[8], 8) == 0)) {
                        memcpy(&this->HashList.at(i).password[7], this->halfHashList.at(j).password, 7);
                        this->HashList.at(i).passwordPart2Inserted = 1;
                    }
                }
            }
        }

        if (this->HashList.at(i).passwordPart1Inserted && this->HashList.at(i).passwordPart2Inserted) {
            this->HashList.at(i).passwordFound = 1;
        }
    }
}