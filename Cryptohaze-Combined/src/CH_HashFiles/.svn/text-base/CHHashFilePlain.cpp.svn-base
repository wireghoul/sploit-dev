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

//#define TRACE_PRINTF 1
#include "CH_HashFiles/CHHashFilePlain.h"
#include "MFN_Common/MFNDebugging.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

CHHashFilePlain::CHHashFilePlain(int newHashLengthBytes) : CHHashFile() {
    trace_printf("CHHashFilePlain::CHHashFilePlain()\n");
    this->HashLengthBytes = newHashLengthBytes;
    this->Protobuf.Clear();
    this->HashExportCacheValid = 0;
}

int CHHashFilePlain::outputNewFoundHashesToFile() {
    trace_printf("CHHashFilePlain::outputNewFoundHashesToFile()\n");
    std::vector<HashPlain>::iterator currentHash;

    // Lock is held by the password reporting function
    
    // Ensure the output file is opened for access before trying to write to it.
    if (this->OutputFile) {
        for (currentHash = this->Hashes.begin();
                currentHash < this->Hashes.end(); currentHash++) {
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

void CHHashFilePlain::parseFileLine(std::string fileLine, size_t lineNumber) {
    trace_printf("CHHashFilePlain::parseFileLine()\n");
    std::string userData, hashData;
    HashPlain HashVectorEntry;
    size_t found;
    
    HashVectorEntry.passwordFound = 0;
    HashVectorEntry.passwordOutputToFile = 0;
    HashVectorEntry.passwordPrinted = 0;
    HashVectorEntry.hash.clear();
    HashVectorEntry.userData.clear();

    // Look for the separator character.  If found, there's a username to
    // split out.
    found = fileLine.find(this->InputDelineator, 0);
        
    if (found != std::string::npos) {
        // Username found - split it out.
        userData = fileLine.substr(0, found);
        hashData = fileLine.substr(found + 1, fileLine.length());
    } else {
        // No username - simply copy the hash.
        hashData = fileLine;
    }
        
    // If the line is not empty and is not the right length, throw error.
    if ((hashData.length() > 0) && (hashData.length() != (this->HashLengthBytes * 2)))
    {
        printf("Hash on line %u: Incorrect length (%d, want %d)\n",
                (unsigned int) lineNumber, (int)hashData.length(),
                (int)(this->HashLengthBytes * 2));
        return;
    }
        
    // If it's a valid line, do the work.
    if (hashData.length() > 0) {
        // Convert the hash to binary.
        HashVectorEntry.hash = this->convertAsciiToBinary(hashData);
        HashVectorEntry.userData = userData;
        HashVectorEntry.originalHash = fileLine;
        if (HashVectorEntry.hash.size() == 0) {
            printf("Hash on line %u: Invalid hash\n",
                    (unsigned int) lineNumber);
            return;
        }
        this->Hashes.push_back(HashVectorEntry);
    }
}

void CHHashFilePlain::performPostLoadOperations() {
    trace_printf("CHHashFilePlain::performPostLoadOperations()\n");
    // Sort the hashes and remove duplicates.
    this->sortHashes();
    
    // Set total hashes and hashes remaining to size of hash vector.
    this->TotalHashes = this->Hashes.size();
    this->TotalHashesRemaining = this->TotalHashes;
}


std::vector<std::vector<uint8_t> > CHHashFilePlain::exportUncrackedHashList() {
    trace_printf("CHHashFilePlain::exportUncrackedHashList()\n");
    std::vector<HashPlain>::iterator currentHash;
    
    this->HashFileMutex.lock();
    
    // Check to see if the cache is valid.  If so, we can just return that.
    // Otherwise, need to generate it.
    
    if (!this->HashExportCacheValid) {
        // Clear the cache and regenerate it.
        this->HashExportCache.clear();
        // Loop through all current hashes.
        for (currentHash = this->Hashes.begin(); currentHash < this->Hashes.end(); currentHash++) {
            // If it's already found, continue.
            if (currentHash->passwordFound) {
                continue;
            }
            // If not, add it to the current return vector.
            this->HashExportCache.push_back(currentHash->hash);
        }
    }

    this->HashFileMutex.unlock();
    
    // No need to sort/unique - this has already been done with the main hash
    // list.  Just return the now-cached data.
    
    return this->HashExportCache;
}

int CHHashFilePlain::reportFoundPassword(std::vector<uint8_t> foundHash, std::vector<uint8_t> foundPassword) {
    return this->reportFoundPassword(foundHash, foundPassword, this->defaultHashAlgorithm);
}

int CHHashFilePlain::reportFoundPassword(std::vector<uint8_t> foundHash, 
        std::vector<uint8_t> foundPassword, uint8_t foundAlgorithmType) {
    trace_printf("CHHashFilePlain::reportFoundPassword()\n");
    std::vector<HashPlain>::iterator currentHash;
    char hashesAdded = 0;

    this->HashFileMutex.lock();
    // Loop through all hashes.
    for (currentHash = this->Hashes.begin(); currentHash < this->Hashes.end(); currentHash++) {
        // Skip if already found.
        if (currentHash->passwordFound) {
            continue;
        }
        // If lengths do not match, wtf?
        if (currentHash->hash.size() != foundHash.size()) {
            continue;
        }
        // Compare hashes.  If there's a match, add the password.
        // This handles multiple usernames having the same hash properly.
        if (memcmp(&currentHash->hash[0], &foundHash[0], currentHash->hash.size()) == 0) {
            if (!currentHash->passwordFound) {
                currentHash->password = foundPassword;
                currentHash->passwordFound = 1;
                currentHash->algorithmType = foundAlgorithmType;
                hashesAdded++;
                this->TotalHashesFound++;
                // Clear the caches.
                this->HashExportCacheValid = 0;
                this->clearProtobufCache();
           }
        }
    }
    
    this->outputNewFoundHashesToFile();
    this->HashFileMutex.unlock();
    return hashesAdded;
}

void CHHashFilePlain::printAllFoundHashes() {
    trace_printf("CHHashFilePlain::printAllFoundHashes()\n");
    std::vector<HashPlain>::iterator currentHash;
    
    this->HashFileMutex.lock();
    // Lock is held by the password reporting function
    
    for (currentHash = this->Hashes.begin();
            currentHash < this->Hashes.end(); currentHash++) {
        if (currentHash->passwordFound) {
            printf("%s\n", this->formatHashToPrint(*currentHash).c_str());
        }
    }
    this->HashFileMutex.unlock();
}

std::string CHHashFilePlain::formatHashToPrint(const HashPlain &hash) {
    trace_printf("CHHashFilePlain::formatHashToPrint()\n");
    // Easier to sprintf into a buffer than muck with streams for formatting.
    char stringBuffer[1024];
    
    memset(stringBuffer, 0, sizeof(stringBuffer));

    // If the printAlgorithm flag is specified, output the algorithm.
        if (this->printAlgorithm) {
            sprintf(stringBuffer, "%s", 
                getHashFunctionByDefinedByte(hash.algorithmType).c_str());
            sprintf(stringBuffer, "%s%c", stringBuffer, this->OutputSeparator);
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


void CHHashFilePlain::printNewFoundHashes() {
    trace_printf("CHHashFilePlain::printNewFoundHashes()\n");
    std::vector<HashPlain>::iterator currentHash;

    this->HashFileMutex.lock();
    
    // Loop through all hashes.
    for (currentHash = this->Hashes.begin(); currentHash < this->Hashes.end(); currentHash++) {
        // Skip if already found.
        if (currentHash->passwordFound && !currentHash->passwordPrinted) {
            printf("%s\n", this->formatHashToPrint(*currentHash).c_str());
            currentHash->passwordPrinted = 1;
        }
    }
    this->HashFileMutex.unlock();
}


int CHHashFilePlain::outputUnfoundHashesToFile(std::string filename) {
    trace_printf("CHHashFilePlain::outputUnfoundHashesToFile()\n");
    std::ofstream hashFile;
    std::vector<HashPlain>::iterator it;

    this->HashFileMutex.lock();
    
    hashFile.open(filename.c_str(), std::ios_base::out);
    //Hope you didn't need the previous contents of this file for anything.
    
    if (hashFile.good()) {
        for (it = this->Hashes.begin(); it < this->Hashes.end(); it++) {
            if (!it->passwordFound)
            {
                hashFile << it->originalHash << std::endl;
            } 
        }            
    }
    this->HashFileMutex.unlock();
    if (hashFile.good())
        return 1;
    else
        return 0;
}

void CHHashFilePlain::importHashListFromRemoteSystem(std::string & remoteData) {
    trace_printf("CHHashFilePlain::importHashListFromRemoteSystem()\n");
    // I hope your CHHashFilePlain was empty.
    // For cleanliness, I will clean this now.
    this->HashFileMutex.lock();
    
    this->Protobuf.Clear();
    this->Hashes.clear();
    this->HashLengthBytes = 0;
    this->HashExportCache.clear();
    this->HashExportCacheValid = 0;

    std::string hashBuffer;
    //Unpack protobuf
    this->Protobuf.ParseFromString(remoteData);
    //Get number of hashes in protobuf
    this->TotalHashes = this->Protobuf.hash_value_size();
    this->HashLengthBytes = this->Protobuf.hash_length_bytes();
    //Make individual HashPlain packages for each hash
    for(uint64_t i=0; i<this->TotalHashes; i++)
    {
        CHHashFilePlain::HashPlain newHashPlain;
        hashBuffer = std::string(this->Protobuf.hash_value(i));
        newHashPlain.hash = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        newHashPlain.password = std::vector<uint8_t>();
        newHashPlain.passwordFound = 0;
        newHashPlain.passwordOutputToFile = 0;
        newHashPlain.passwordPrinted = 0;
        
        this->Hashes.push_back(newHashPlain);
    }

    this->Protobuf.Clear();
    this->TotalHashesRemaining = this->TotalHashes;
    this->HashFileMutex.unlock();
}

void CHHashFilePlain::createHashListExportProtobuf() {
    trace_printf("CHHashFilePlain::createHashListExportProtobuf()\n");
    this->hashExportProtobufCache.clear();
    
    // Cache is not valid - create a new protobuf to export.
    this->Protobuf.Clear();
    std::string hashBuffer;
    
    std::vector<CHHashFilePlain::HashPlain>::iterator i;
    for(i=this->Hashes.begin();i<this->Hashes.end();i++)
    {
        // Only export not-found hashes.
        if (!i->passwordFound) {
            hashBuffer = std::string(i->hash.begin(), i->hash.end());
            this->Protobuf.add_hash_value(hashBuffer); 
        }
    }
    this->Protobuf.set_hash_length_bytes(this->HashLengthBytes);
    
    this->Protobuf.SerializeToString(&this->hashExportProtobufCache);
    
    this->protobufExportCachesValid = 1;    
}



bool CHHashFilePlain::plainHashSortPredicate(const HashPlain &d1, const HashPlain &d2) {
    for (size_t i = 0; i < d1.hash.size(); i++) {
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


bool CHHashFilePlain::plainHashUniquePredicate(const HashPlain &d1, const HashPlain &d2) {
    // Hashes are only equal if the hash is equal and the user data is equal.
    if (memcmp(&d1.hash[0], &d2.hash[0], d1.hash.size()) == 0) {
        if (d1.userData == d2.userData) {
            return 1;
        }
        return 0;
    }
    return 0;
}

void CHHashFilePlain::sortHashes() {
    trace_printf("CHHashFilePlain::sortHashes()\n");
    // Sort hashes and remove duplicates.
    std::sort(this->Hashes.begin(), this->Hashes.end(), CHHashFilePlain::plainHashSortPredicate);
    this->Hashes.erase(
        std::unique(this->Hashes.begin(), this->Hashes.end(), CHHashFilePlain::plainHashUniquePredicate ),
        this->Hashes.end() );
}


//#define UNIT_TEST 1

#if UNIT_TEST
#include <string.h>

char foundPasswordString[] = "Password";
static const std::string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

int main(int argc, char *argv[]) {
    
    CHHashFilePlain HashFile(16);
    CHHashFilePlain HashFile2(16);
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
