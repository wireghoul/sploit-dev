/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
Copyright (C) 2012  Bitweasil (http://www.cryptohaze.com/)

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

#include <fstream>

#include "CH_HashFiles/CHHashFilePlain.h"
#include "CH_HashFiles/CHHashFileLM.h"
#include "MFN_Common/MFNDebugging.h"

// LM hashes are 8 bytes long for each half hash fragment.
CHHashFileLM::CHHashFileLM() : CHHashFilePlain(8){ }

void CHHashFileLM::parseFileLine(std::string fileLine, size_t lineNumber) {
    trace_printf("CHHashFileLM::parseFileLine()\n");
    
    // This is the null hash - often seen in the second part.
    const unsigned char nullHash[8] = {0xAA, 0xD3, 0xB4, 0x35, 0xB5, 0x14, 0x04, 0xEE};

    // 
    LMFullHashData FullHashElement;
    HashPlain HalfHashElement;
    
    FullHashElement.passwordFound = 0;
    FullHashElement.passwordPrinted = 0;
    FullHashElement.passwordOutputToFile = 0;
    FullHashElement.passwordPart1Inserted = 0;
    FullHashElement.passwordPart1IsNull = 0;
    FullHashElement.passwordPart2Inserted = 0;
    FullHashElement.passwordPart2IsNull = 0;
    
    // Set the password to 14 'x' characters.  As they will never happen in a 
    // legit LM hash, they are usable as flags.
    FullHashElement.password.resize(14, 'x');
    
    HalfHashElement.passwordFound = 0;
    HalfHashElement.passwordOutputToFile = 0;
    HalfHashElement.passwordPrinted = 0;
    HalfHashElement.algorithmType = 0;
    // Set the hash to be 8 bytes of null - will copy data in later.
    HalfHashElement.hash.resize(8, 0);
    
    
    if ((fileLine.length() > 0) && (fileLine.length() != 32))
    {
        printf("Hash in line %u: incorrect length!\n", (unsigned int) lineNumber);
        return;
    }

    // If it's a valid line, do the work.
    if (fileLine.length() > 0) {
        FullHashElement.originalHash = fileLine;
        
        // Convert the hash to binary.
        FullHashElement.hash = this->convertAsciiToBinary(fileLine);
        if (FullHashElement.hash.size() == 0) {
            printf("Hash in line %u invalid hash!\n", (unsigned int) lineNumber);
            return;
        }
        // Push the full hash back.
        this->fullHashList.push_back(FullHashElement);

        // Break it into half hashes & push them back if needed.

        // If the first half is not the null hash, add it.
        if (memcmp(&FullHashElement.hash[0], nullHash, 8)) {
            memcpy(&HalfHashElement.hash[0], &FullHashElement.hash[0], 8);
            this->Hashes.push_back(HalfHashElement);
        }
        // Check the second half-hash
        if (memcmp(&FullHashElement.hash[8], nullHash, 8)) {
            memcpy(&HalfHashElement.hash[0], &FullHashElement.hash[8], 8);
            this->Hashes.push_back(HalfHashElement);
        }
    }
}

int CHHashFileLM::outputNewFoundHashesToFile() {
    trace_printf("CHHashFileLM::outputNewFoundHashesToFile()\n");
    std::vector<LMFullHashData>::iterator currentHash;

    // Lock is held by the password reporting function
    this->MergeHalfPartsIntoFullPasswords();

    // Ensure the output file is opened for access before trying to write to it.
    if (this->OutputFile) {
        for (currentHash = this->fullHashList.begin();
                currentHash < this->fullHashList.end(); currentHash++) {
            // Skip if already reported.
            if (currentHash->passwordFound && !currentHash->passwordOutputToFile) {
                fprintf(this->OutputFile, "%s\n",
                    this->formatHashToPrint(*currentHash).c_str());
                // Mark hash as reported.
                currentHash->passwordOutputToFile = 1;
            }
        }
    }
    fflush(this->OutputFile);

    return 1;
}

void CHHashFileLM::MergeHalfPartsIntoFullPasswords() {
    trace_printf("CHHashFileLM::MergeHalfPartsIntoFullPasswords()\n");
    const unsigned char nullHash[8] = {0xAA, 0xD3, 0xB4, 0x35, 0xB5, 0x14, 0x04, 0xEE};
    HashPlain SearchValue;
    SearchValue.hash.resize(8, 0);

    std::vector<LMFullHashData>::iterator HashListIterator;
    std::vector<HashPlain>::iterator HalfHashListIterator;
    std::pair<std::vector<HashPlain>::iterator,std::vector<HashPlain>::iterator> valueBounds;

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
                memcpy(&SearchValue.hash[0], &HashListIterator->hash[0], 8);
                valueBounds = std::equal_range(this->Hashes.begin(), this->Hashes.end(),
                        SearchValue, CHHashFilePlain::plainHashSortPredicate);
                // Start loop
                for (HalfHashListIterator = valueBounds.first;
                        HalfHashListIterator < valueBounds.second; HalfHashListIterator++) {
                    // Check for a match.  If it matches, copy the password bit in.
                    if (HalfHashListIterator->passwordFound &&
                            (memcmp(&HalfHashListIterator->hash[0], &HashListIterator->hash[0], 8) == 0)) {
                        memcpy(&HashListIterator->password[0], &HalfHashListIterator->password[0],
                                HalfHashListIterator->password.size());
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
                memcpy(&SearchValue.hash[0], &HashListIterator->hash[8], 8);
                valueBounds = std::equal_range(this->Hashes.begin(), this->Hashes.end(),
                        SearchValue, CHHashFilePlain::plainHashSortPredicate);

                for (HalfHashListIterator = valueBounds.first;
                        HalfHashListIterator < valueBounds.second; HalfHashListIterator++) {
                    // Check for a match.  If it matches, copy the password bit in.
                    if (HalfHashListIterator->passwordFound &&
                            (memcmp(&HalfHashListIterator->hash[0], &HashListIterator->hash[8], 8) == 0)) {
                        memcpy(&HashListIterator->password[7], &HalfHashListIterator->password[0],
                                HalfHashListIterator->password.size());
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

void CHHashFileLM::SortHashes() {
    trace_printf("CHHashFileLM::SortHashes()\n");
    
    // Sort both sets of hashes.
    
    // Sort the full hashes by hash.
    std::sort(this->fullHashList.begin(), this->fullHashList.end(), CHHashFileLM::LMHashFullSortPredicate);
    this->fullHashList.erase(
        std::unique(this->fullHashList.begin(), this->fullHashList.end(), CHHashFileLM::LMHashFullUniquePredicate),
        this->fullHashList.end());
    
    // Sort plain hashes
    CHHashFilePlain::sortHashes();
}

bool CHHashFileLM::LMHashFullSortPredicate(const LMFullHashData &d1, const LMFullHashData &d2) {
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

bool CHHashFileLM::LMHashFullUniquePredicate(const LMFullHashData &d1, const LMFullHashData &d2) {
    if (memcmp(&d1.hash[0], &d2.hash[0], d1.hash.size()) == 0) {
        return 1;
    }
    return 0;
}

int CHHashFileLM::outputUnfoundHashesToFile(std::string filename) {
    trace_printf("CHHashFileLM::outputUnfoundHashesToFile()\n");
    std::ofstream hashFile;
    std::vector<LMFullHashData>::iterator it;

    this->HashFileMutex.lock();
    
    hashFile.open(filename.c_str(), std::ios_base::out);
    //Hope you didn't need the previous contents of this file for anything.
    
    if (hashFile.good()) {
        for (it = this->fullHashList.begin(); it < this->fullHashList.end(); it++) {
            if (!it->passwordFound)
            {
                hashFile << it->originalHash << std::endl;
            } 
        }            
    }
    this->HashFileMutex.unlock();
    if (hashFile.good()) {
        hashFile.close();
        return 1;
    } else {
        return 0;
    }
}

void CHHashFileLM::printAllFoundHashes() {
    trace_printf("CHHashFileLM::printAllFoundHashes()\n");
    std::vector<LMFullHashData>::iterator currentHash;
    
    this->HashFileMutex.lock();
    // Lock is held by the password reporting function
    this->MergeHalfPartsIntoFullPasswords();
    
    for (currentHash = this->fullHashList.begin();
            currentHash < this->fullHashList.end(); currentHash++) {
        if (currentHash->passwordFound) {
            printf("%s\n", this->formatHashToPrint(*currentHash).c_str());
        }
    }
    this->HashFileMutex.unlock();
}

void CHHashFileLM::printNewFoundHashes() {
    trace_printf("CHHashFileLM::printNewFoundHashes()\n");
    std::vector<LMFullHashData>::iterator currentHash;

    this->HashFileMutex.lock();
    this->MergeHalfPartsIntoFullPasswords();
    
    // Loop through all hashes.
    for (currentHash = this->fullHashList.begin(); currentHash < this->fullHashList.end(); currentHash++) {
        // Skip if already found.
        if (currentHash->passwordFound && !currentHash->passwordPrinted) {
            printf("%s\n", this->formatHashToPrint(*currentHash).c_str());
            currentHash->passwordPrinted = 1;
        }
    }
    this->HashFileMutex.unlock();
}


std::string CHHashFileLM::formatHashToPrint(const LMFullHashData &hash) {
    trace_printf("CHHashFileLM::formatHashToPrint()\n");
    
    // Easier to sprintf into a buffer than muck with streams for formatting.
    char stringBuffer[1024];

    memset(stringBuffer, 0, sizeof (stringBuffer));

    // If the printAlgorithm flag is specified, output the algorithm.
    if (this->printAlgorithm) {
        sprintf(stringBuffer, "LM%c", this->OutputSeparator);
    }

    // Print the hash as we found it.
    sprintf(stringBuffer, "%s%s%c", stringBuffer,
            hash.originalHash.c_str(),
            this->OutputSeparator);

    // Print the password, and if needed the hex output.
    for (size_t i = 0; i < hash.password.size(); i++) {
        sprintf(stringBuffer, "%s%c", stringBuffer, hash.password[i]);
    }
    if (this->AddHexOutput) {
        sprintf(stringBuffer, "%s%c", stringBuffer, this->OutputSeparator);
        sprintf(stringBuffer, "%s0x", stringBuffer);
        for (size_t i = 0; i < hash.password.size(); i++) {
            sprintf(stringBuffer, "%s%02x", stringBuffer, hash.password[i]);
        }
    }

    return std::string(stringBuffer);
}


//#define UNIT_TEST 1

#if UNIT_TEST
#include <string.h>

char foundPasswordString[] = "Passwor";

int main(int argc, char *argv[]) {
    
    CHHashFileLM HashFile;
    CHHashFileLM HashFile2;
    std::string transfer;
    //HashFile.testPHPPassHash();
    /*
    const std::string s = "ADP GmbH\nAnalyse Design & Programmierung\nGesellschaft mit beschränkter Haftung\0";
    //const std::string s = "abcdefghijklmnopqrstuvwxyz";
    std::vector<uint8_t> sourceData, encodedData, decodedData;
    
    for (int i = 0; i < s.size(); i++) {
        sourceData.push_back(s[i]);
    }
    encodedData = HashFile.base64Encode(sourceData, base64_chars);
    printf("Encoded data: ");
    for (int i = 0; i < encodedData.size(); i++) {
        printf("%c", encodedData[i]);
    }
    printf("\n\n");
    
    decodedData = HashFile.base64Decode(encodedData, base64_chars);
    printf("Decoded data: ");
    for (int i = 0; i < decodedData.size(); i++) {
        printf("%c", decodedData[i]);
    }
    printf("\n\n");
    */
    
    
    
    std::vector<std::vector<uint8_t> > Hashes;
    std::vector<uint8_t> FoundPassword;
    uint32_t i;
    
    for (i = 0; i < strlen(foundPasswordString); i++) {
        FoundPassword.push_back(foundPasswordString[i]);
    }
    
    if (argc != 2) {
        printf("program hashfile\n");
        exit(1);
    }
    
    HashFile.openHashFile(argv[1]);
    printf("Loaded hash file.\n");
    Hashes = HashFile.exportUncrackedHashList();
    
    printf("Exported hashes: \n");
    for (i = 0; i < Hashes.size(); i++) {
        for (int j = 0; j < Hashes[i].size(); j++) {
            printf("%02x", Hashes[i][j]);
        }
        printf("\n");
    }
    
    // Report every other hash as found.
    for (i = 0; i < Hashes.size(); i += 2) {
        HashFile.reportFoundPassword(Hashes[i], FoundPassword);
    }
    HashFile.setAddHexOutput(true);
    HashFile.setUseJohnOutputStyle(true);
    //HashFile.SetOutputSeparator('-');
    
    HashFile.printAllFoundHashes();
    
    printf("Testing protobufs\n");
    HashFile.exportHashListToRemoteSystem(transfer);
    
    HashFile2.importHashListFromRemoteSystem(transfer);
    Hashes = HashFile2.exportUncrackedHashList();
    
    printf("Exported hashes: \n");
    for (i = 0; i < Hashes.size(); i++) {
        for (int j = 0; j < Hashes[i].size(); j++) {
            printf("%02x", Hashes[i][j]);
        }
        printf("\n");
    }

    for (i = 0; i < HashFile2.getTotalHashCount(); i += 2) {
        HashFile2.reportFoundPassword(Hashes[i], FoundPassword);
    }

    Hashes = HashFile2.exportUncrackedHashList();
    
    printf("Exported hashes: \n");
    for (i = 0; i < Hashes.size(); i++) {
        for (int j = 0; j < Hashes[i].size(); j++) {
            printf("%02x", Hashes[i][j]);
        }
        printf("\n");
    }
    
    
}

#endif
