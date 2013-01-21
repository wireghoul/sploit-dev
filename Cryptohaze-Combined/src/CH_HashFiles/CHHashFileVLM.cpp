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

#include "CH_HashFiles/CHHashFileVLM.h"
#include "CH_HashFiles/CHHashFileVPlain.h"
#include "MFN_Common/MFNDebugging.h"

#include <vector>
#include <algorithm>

CHHashFileVPlainLM::CHHashFileVPlainLM() : CHHashFileV() {
    printf("CHHashFileVPlainLM::CHHashFileVPlainLM()\n");
    // No actions needed.  Just call the public class.
}


int CHHashFileVPlainLM::OpenHashFile(std::string filename) {
    printf("CHHashFileVPlainLM::OpenHashFile(%s)\n", filename.c_str());
    
    std::ifstream hashFile;
    std::string fileLine;

    // This is the null hash - often seen in the second part.
    const unsigned char nullHash[8] = {0xAA, 0xD3, 0xB4, 0x35, 0xB5, 0x14, 0x04, 0xEE};

    LMFullHashData FullHashVectorEntry;
    LMFragmentHashData HalfHashVectorEntry;
    
    std::string whitespaces (" \t\f\v\n\r");
    size_t found;
    
    FullHashVectorEntry.hash.resize(16, 0);
    FullHashVectorEntry.passwordFound = 0;
    FullHashVectorEntry.passwordOutputToFile = 0;
    FullHashVectorEntry.passwordPart1Inserted = 0;
    FullHashVectorEntry.passwordPart1IsNull = 0;
    FullHashVectorEntry.passwordPart2Inserted = 0;
    FullHashVectorEntry.passwordPart2IsNull = 0;
    FullHashVectorEntry.passwordReported = 0;
    
    // Set the password to 14 'x' characters.  As they will never happen in a 
    // legit LM hash, they are usable as flags.
    FullHashVectorEntry.password.resize(14, 'x');
    
    // Will overwrite hash 
    memset(HalfHashVectorEntry.password, 0, sizeof(HalfHashVectorEntry.password));
    HalfHashVectorEntry.passwordFound = 0;
    HalfHashVectorEntry.passwordOutputToFile = 0;
    HalfHashVectorEntry.passwordReported = 0;
    
    
    hashFile.open(filename.c_str(), std::ios_base::in);
    if (!hashFile.good())
    {
        
        std::cout << "ERROR: Cannot open hashfile " << filename <<"\n";
        exit(1);
    }
    
    while (std::getline(hashFile, fileLine)) {
        FullHashVectorEntry.hash.clear();
        found=fileLine.find_last_not_of(whitespaces);
        if (found!=std::string::npos)
            fileLine.erase(found+1);
        else
            fileLine.clear();  
        //printf("Hash length: %d\n", (int)fileLine.length());
        // If the line is not empty and is not the right length, throw error.
        // 16 ASCII characters per full line.
        if ((fileLine.length() > 0) && (fileLine.length() != 32))
        {
            std::cout << "Hash in line "<< this->fullHashList.size() <<" incorrect length!\n";
            exit(1);
        }
        
        // If it's a valid line, do the work.
        if (fileLine.length() > 0) {
            // Convert the hash to binary.
            FullHashVectorEntry.hash = this->convertAsciiToBinary(fileLine);
            if (FullHashVectorEntry.hash.size() == 0) {
                std::cout << "Hash in line "<< this->fullHashList.size() <<" invalid hash!\n";
                exit(1);
            }
            // Push the full hash back.
            this->fullHashList.push_back(FullHashVectorEntry);
            
            // Break it into half hashes & push them back if needed.
            
            // If the first half is not the null hash, add it.
            if (memcmp(&FullHashVectorEntry.hash[0], nullHash, 8)) {
                memcpy(HalfHashVectorEntry.halfHash, &FullHashVectorEntry.hash[0], 8);
                this->halfHashList.push_back(HalfHashVectorEntry);
            }
            // Check the second half-hash
            if (memcmp(&FullHashVectorEntry.hash[8], nullHash, 8)) {
                memcpy(HalfHashVectorEntry.halfHash, &FullHashVectorEntry.hash[8], 8);
                this->halfHashList.push_back(HalfHashVectorEntry);
            }
        }
    }
    
    this->SortHashes();

    // Set the total number of hashes to the number of non-null half-hashes, as
    // this is what the system works on.
    this->TotalHashes = this->halfHashList.size();
    this->TotalHashesRemaining = this->TotalHashes;
    
    hashFile.close();
    return 1;    
}

int CHHashFileVPlainLM::OutputFoundHashesToFile() {
    printf("CHHashFileVPlainLM::OutputFoundHashesToFile()\n");
    return 1;
}

void CHHashFileVPlainLM::MergeHalfPartsIntoFullPasswords() {
    printf("CHHashFileVPlainLM::MergeHalfPartsIntoFullPasswords()\n");
    const unsigned char nullHash[8] = {0xAA, 0xD3, 0xB4, 0x35, 0xB5, 0x14, 0x04, 0xEE};
    LMFragmentHashData SearchValue;

    std::vector<LMFullHashData>::iterator HashListIterator;
    std::vector<LMFragmentHashData>::iterator HalfHashListIterator;
    std::pair<std::vector<LMFragmentHashData>::iterator,std::vector<LMFragmentHashData>::iterator> valueBounds;

    // Loop through all the "full" hashes
    // Be a bit more intelligent to avoid taking all eternity...
    for (HashListIterator = this->fullHashList.begin();
            HashListIterator < this->fullHashList.end(); HashListIterator++) {
        // If the password has been filled in, continue - no point in checking.
        if (HashListIterator->passwordFound) {
            continue;
        }

        // If the first part is not updated, try for it.
        if (!HashListIterator->passwordPart1Inserted) {
            // Check for empty hashes.
            if (memcmp(&HashListIterator->hash[0], nullHash, 8) == 0) {
                memset(&HashListIterator->password[0], 0, 7);
                HashListIterator->passwordPart1Inserted = 1;
                HashListIterator->passwordPart1IsNull = 1;
            } else {
                // Copy the value we're looking for into the search variable.
                memcpy(SearchValue.halfHash, &HashListIterator->hash[0], 8);
                valueBounds = std::equal_range(this->halfHashList.begin(), this->halfHashList.end(),
                        SearchValue, CHHashFileVPlainLM::LMHashHalfSortPredicate);
                // Start loop
                for (HalfHashListIterator = valueBounds.first;
                        HalfHashListIterator < valueBounds.second; HalfHashListIterator++) {
                    // Check for a match.  If it matches, copy the password bit in.
                    if (HalfHashListIterator->passwordFound &&
                            (memcmp(HalfHashListIterator->halfHash, &HashListIterator->hash[0], 8) == 0)) {
                        memcpy(&HashListIterator->password[0], HalfHashListIterator->password, 7);
                        HashListIterator->passwordPart1Inserted = 1;
                    }
                } // End loop
            }
        }

        // If the second part is not updated, try for it.
        if (!HashListIterator->passwordPart2Inserted) {
            // Check for empty hashes.
            if (memcmp(&HashListIterator->hash[8], nullHash, 8) == 0) {
                memset(&HashListIterator->password[7], 0, 7);
                HashListIterator->passwordPart2Inserted = 1;
                HashListIterator->passwordPart2IsNull = 1;
            } else {
                // Copy the value we're looking for into the search variable.
                memcpy(SearchValue.halfHash, &HashListIterator->hash[8], 8);
                valueBounds = std::equal_range(this->halfHashList.begin(), this->halfHashList.end(),
                        SearchValue, CHHashFileVPlainLM::LMHashHalfSortPredicate);

                for (HalfHashListIterator = valueBounds.first;
                        HalfHashListIterator < valueBounds.second; HalfHashListIterator++) {
                    // Check for a match.  If it matches, copy the password bit in.
                    if (HalfHashListIterator->passwordFound &&
                            (memcmp(HalfHashListIterator->halfHash, &HashListIterator->hash[8], 8) == 0)) {
                        memcpy(&HashListIterator->password[7], HalfHashListIterator->password, 7);
                        HashListIterator->passwordPart2Inserted = 1;
                    }
                }
            }
        }

        if (HashListIterator->passwordPart1Inserted && HashListIterator->passwordPart2Inserted) {
            HashListIterator->passwordFound = 1;
        }
    }
}

std::vector<std::vector<uint8_t> > CHHashFileVPlainLM::ExportUncrackedHashList() {
    printf("CHHashFileVPlainLM::ExportUncrackedHashList()\n");

    std::vector<std::vector<uint8_t> > returnVector;
    std::vector<LMFragmentHashData>::iterator currentHash;
    std::vector<uint8_t> currentHashVector;
    
    this->HashFileMutex.lock();
    currentHashVector.resize(8, 0);
    
    // Loop through all current hashes.
    for (currentHash = this->halfHashList.begin(); currentHash < this->halfHashList.end(); currentHash++) {
        // If it's already found, continue.
        if (currentHash->passwordFound) {
            continue;
        }
        // If not, add it to the current return vector.
        memcpy(&currentHashVector[0], currentHash->halfHash, 8);
        returnVector.push_back(currentHashVector);
    }
    
    this->HashFileMutex.unlock();
    return returnVector;
}

int CHHashFileVPlainLM::ReportFoundPassword(std::vector<uint8_t> hash, std::vector<uint8_t> password) {
    printf("CHHashFileVPlainLM::ReportFoundPassword()\n");
    printf("hash: ");
    for (int i = 0; i < hash.size(); i++) {
        printf("%02x", hash[i]);
    }
    printf("  pass: ");
    for (int i = 0; i < password.size(); i++) {
        printf("%c", password[i]);
    }
    printf("\n");


    std::vector<LMFragmentHashData>::iterator currentHash;

    this->HashFileMutex.lock();

    // TODO: Optimize with a search in the list.  std::search?

    for (currentHash = this->halfHashList.begin(); currentHash < this->halfHashList.end(); currentHash++) {
        // Compare 8 bytes - if matching, store it.
        if (memcmp(&hash[0], &currentHash->halfHash[0], 8) == 0) {
            // Only do this if the password is not already reported.
            if (!currentHash->passwordFound) {
                // Copy the bytes in.  This has been previously memset to 0.
                memcpy(&currentHash->password[0], &password[0], password.size());
                currentHash->passwordFound = 1;
                this->TotalHashesFound++;
                this->TotalHashesRemaining--;
                // Output to a file if needed.
                this->OutputFoundHashesToFile();

                // Hashes should be unique.  Therefore, if we find it once,
                // we should not need to continue looking.
                this->HashFileMutex.unlock();
                return 1;
            }
        }
    }

    this->HashFileMutex.unlock();
    return 0;
}

void CHHashFileVPlainLM::PrintAllFoundHashes() {
    printf("CHHashFileVPlainLM::PrintAllFoundHashes()\n");
    std::vector<LMFullHashData>::iterator HashListIterator;
    int j;

    this->MergeHalfPartsIntoFullPasswords();

    for (HashListIterator = this->fullHashList.begin();
            HashListIterator < this->fullHashList.end(); HashListIterator++) {
        // Print hashes in the following:
        // - Either part 1 or part 2 is found
        // - If part1 is found & part 2 is null, print
        if ((HashListIterator->passwordPart1Inserted || HashListIterator->passwordPart2Inserted) &&
                !(HashListIterator->passwordPart2IsNull && !HashListIterator->passwordPart1Inserted)) {
            for (j = 0; j < 16; j++) {
                printf("%02X", HashListIterator->hash[j]);
            }
            printf(":");
            for (j = 0; j < strlen((const char *)&HashListIterator->password[0]); j++) {
                    printf("%c", HashListIterator->password[j]);
            }
            if (this->AddHexOutput) {
                printf(":0x");
                for (j = 0; j < strlen((const char *)&HashListIterator->password[0]); j++) {
                        printf("%02x", HashListIterator->password[j]);
                }
            }
            printf("\n");
            HashListIterator->passwordReported = 1;
        }
    }
    this->OutputFoundHashesToFile();
}

void CHHashFileVPlainLM::PrintNewFoundHashes() {
    printf("CHHashFileVPlainLM::PrintNewFoundHashes()\n");
}

int CHHashFileVPlainLM::OutputUnfoundHashesToFile(std::string filename) {
    printf("CHHashFileVPlainLM::OutputUnfoundHashesToFile()\n");
    return 1;
}

void CHHashFileVPlainLM::ImportHashListFromRemoteSystem(std::string& remoteData) {
    printf("CHHashFileVPlainLM::ImportHashListFromRemoteSystem()\n");
}

void CHHashFileVPlainLM::ExportHashListToRemoteSystem(std::string* exportData) {
    printf("CHHashFileVPlainLM::ExportHashListToRemoteSystem()\n");
}

void CHHashFileVPlainLM::SortHashes() {
    printf("CHHashFileVPlainLM::SortHashes()\n");
    
    // Sort both sets of hashes.
    
    // Sort the full hashes by hash.
    std::sort(this->fullHashList.begin(), this->fullHashList.end(), CHHashFileVPlainLM::LMHashFullSortPredicate);
    this->fullHashList.erase(
        std::unique(this->fullHashList.begin(), this->fullHashList.end(), CHHashFileVPlainLM::LMHashFullUniquePredicate),
        this->fullHashList.end());
    
    // Sort the half hashes by hash.
    std::sort(this->halfHashList.begin(), this->halfHashList.end(), CHHashFileVPlainLM::LMHashHalfSortPredicate);
    this->halfHashList.erase(
        std::unique(this->halfHashList.begin(), this->halfHashList.end(), CHHashFileVPlainLM::LMHashHalfUniquePredicate),
        this->halfHashList.end());
}


bool CHHashFileVPlainLM::LMHashFullSortPredicate(const LMFullHashData &d1, const LMFullHashData &d2) {
    int i;
    for (i = 0; i < d1.hash.size(); i++) {
        if (d1.hash[i] == d2.hash[i]) {
            continue;
        } else if (d1.hash[i] > d2.hash[i]) {
            return 0;
        } else if (d1.hash[i] < d2.hash[i]) {
            return 1;
        }
    }
    // Exactly equal = return 0.
    return 0;
}
bool CHHashFileVPlainLM::LMHashHalfSortPredicate(const LMFragmentHashData &d1, const LMFragmentHashData &d2) {
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

bool CHHashFileVPlainLM::LMHashFullUniquePredicate(const LMFullHashData &d1, const LMFullHashData &d2) {
    if (memcmp(&d1.hash[0], &d2.hash[0], d1.hash.size()) == 0) {
        return 1;
    }
    return 0;
}

bool CHHashFileVPlainLM::LMHashHalfUniquePredicate(const LMFragmentHashData &d1, const LMFragmentHashData &d2) {
    if (memcmp(&d1.halfHash[0], &d2.halfHash[0], 8) == 0) {
        return 1;
    }
    return 0;
}

//#define UNIT_TEST

#ifdef UNIT_TEST

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    printf("LM Hash File Unit Test\n");

    if (argc != 2) {
        printf("Usage: %s [LM hash file]\n", argv[0]);
        exit(1);
    }
    
    CHHashFileVPlainLM *LMHashes;
    std::string filename;
    std::vector<std::vector<uint8_t> > sortedHashes;
    std::vector<std::vector<uint8_t> >::iterator sortedHashesIterator;
    int i;

    LMHashes = new CHHashFileVPlainLM();
    
    filename = argv[1];
    
    if (!LMHashes->OpenHashFile(filename)) {
        printf("Error opening hash file!\n");
        exit(1);
    }

    sortedHashes = LMHashes->ExportUncrackedHashList();

    for (sortedHashesIterator = sortedHashes.begin(); sortedHashesIterator < sortedHashes.end(); sortedHashesIterator++) {
        for (i = 0; i < 8; i++) {
            printf("%02x", sortedHashesIterator->at(i));
        }
        printf("\n");
    }

    LMHashes->PrintAllFoundHashes();
}

#endif