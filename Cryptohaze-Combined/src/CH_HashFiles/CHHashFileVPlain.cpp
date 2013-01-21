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

#include "CH_HashFiles/CHHashFileVPlain.h"

#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>

#define HEX( x ) std::setw(2) << std::setfill('0') << std::hex << (int)( x )


CHHashFileVPlain::CHHashFileVPlain(int newHashLengthBytes) : CHHashFileV() {
    this->HashLengthBytes = newHashLengthBytes;
    this->Protobuf.Clear();
    this->ExportProtobufCache.clear();
}

int CHHashFileVPlain::OutputFoundHashesToFile() {
    std::vector<HashPlain>::iterator currentHash;

    // Lock is held by the password reporting function
    
    // Ensure the output file is opened for access before trying to write to it.
    if (this->OutputFile) {
        for (currentHash = this->Hashes.begin();
                currentHash < this->Hashes.end(); currentHash++) {
            // Skip if already reported.
            if (currentHash->passwordFound && !currentHash->passwordOutputToFile) {
                // If the printAlgorithm flag is specified, output the algorithm.
                if (currentHash->userData.length()) {
                    fprintf(this->OutputFile, "%s%c", currentHash->userData.c_str(), this->OutputSeparator);
                }
                if (this->printAlgorithm) {
                    fprintf(this->OutputFile, "%s", 
                        getHashFunctionByDefinedByte(currentHash->algorithmType).c_str());
                    fprintf(this->OutputFile, "%c", this->OutputSeparator);
                }
                for (size_t i = 0; i < currentHash->hash.size(); i++) {
                    fprintf(this->OutputFile, "%02x", currentHash->hash[i]);
                }
                fprintf(this->OutputFile, "%c", this->OutputSeparator);
                for (size_t i = 0; i < currentHash->password.size(); i++) {
                    fprintf(this->OutputFile, "%c", currentHash->password[i]);
                }
                if (this->AddHexOutput) {
                    fprintf(this->OutputFile, "%c", this->OutputSeparator);
                    fprintf(this->OutputFile, "0x");
                    for (size_t i = 0; i < currentHash->password.size(); i++) {
                        fprintf(this->OutputFile, "%02x", currentHash->password[i]);
                    }
                }
                fprintf(this->OutputFile, "\n");
                // Mark hash as reported.
                currentHash->passwordOutputToFile = 1;
            }
        }
    }
    fflush(this->OutputFile);

    return 1;
}


int CHHashFileVPlain::OpenHashFile(std::string filename) {
    std::ifstream hashFile;
    std::string fileLine;
    std::string userData, hashData;
    HashPlain HashVectorEntry;
    
    std::string whitespaces (" \t\f\v\n\r");
    size_t found;
    
    HashVectorEntry.passwordFound = 0;
    HashVectorEntry.passwordOutputToFile = 0;
    HashVectorEntry.passwordPrinted = 0;

    this->HashFileMutex.lock();
    
    hashFile.open(filename.c_str(), std::ios_base::in);
    if (!hashFile.good())
    {
        
        std::cout << "ERROR: Cannot open hashfile " << filename <<"\n";
        exit(1);
    }
    
    while (std::getline(hashFile, fileLine)) {
        HashVectorEntry.hash.clear();
        HashVectorEntry.userData.clear();
        found=fileLine.find_last_not_of(whitespaces);
        if (found!=std::string::npos)
            fileLine.erase(found+1);
        else
            fileLine.clear();
        
        //printf("Hash length: %d\n", (int)fileLine.length());
        
        // Look for the separator character.  If found, there's a username to
        // split out.
        found = fileLine.find(this->InputDelineator, 0);
        
        userData.clear();
        hashData.clear();
        
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
            std::cout << "Hash in line "<< this->Hashes.size() <<" incorrect length!\n";
            exit(1);
        }
        
        // If it's a valid line, do the work.
        if (hashData.length() > 0) {
            // Convert the hash to binary.
            HashVectorEntry.hash = this->convertAsciiToBinary(hashData);
            HashVectorEntry.userData = userData;
            if (HashVectorEntry.hash.size() == 0) {
                std::cout << "Hash in line "<< this->Hashes.size() <<" invalid hash!\n";
                exit(1);
            }
            this->Hashes.push_back(HashVectorEntry);
        }
    }
    
    this->SortHashes();
    
    // Set total hashes and hashes remaining to size of hash vector.
    this->TotalHashes = this->Hashes.size();
    this->TotalHashesRemaining = this->TotalHashes;
    
    this->ExportProtobufCache.clear();

    hashFile.close();
    
    this->HashFileMutex.unlock();

    return 1;
}

std::vector<std::vector<uint8_t> > CHHashFileVPlain::ExportUncrackedHashList() {
    std::vector<std::vector<uint8_t> > returnVector;
    std::vector<HashPlain>::iterator currentHash;
    
    this->HashFileMutex.lock();
    
    // Loop through all current hashes.
    for (currentHash = this->Hashes.begin(); currentHash < this->Hashes.end(); currentHash++) {
        // If it's already found, continue.
        if (currentHash->passwordFound) {
            continue;
        }
        // If not, add it to the current return vector.
        returnVector.push_back(currentHash->hash);
    }
    
    this->HashFileMutex.unlock();
    
    std::sort(returnVector.begin(), returnVector.end());
    returnVector.erase(
        std::unique(returnVector.begin(), returnVector.end()),
        returnVector.end() );
    return returnVector;
}

int CHHashFileVPlain::ReportFoundPassword(std::vector<uint8_t> foundHash, std::vector<uint8_t> foundPassword) {
    return this->ReportFoundPassword(foundHash, foundPassword, this->defaultHashAlgorithm);
}

int CHHashFileVPlain::ReportFoundPassword(std::vector<uint8_t> foundHash, 
        std::vector<uint8_t> foundPassword, uint8_t foundAlgorithmType) {
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
                this->ExportProtobufCache.clear();
           }
        }
    }
    this->OutputFoundHashesToFile();
    this->ExportProtobufCache.clear();
    this->HashFileMutex.unlock();
    return hashesAdded;
}

void CHHashFileVPlain::PrintAllFoundHashes() {
    std::vector<HashPlain>::iterator currentHash;
    
    this->HashFileMutex.lock();
    
    // Loop through all hashes.
    for (currentHash = this->Hashes.begin(); currentHash < this->Hashes.end(); currentHash++) {
        // Skip if already found.
        if (currentHash->passwordFound) {
            this->PrintHash(*currentHash);
        }
    }
    this->HashFileMutex.unlock();
}


void CHHashFileVPlain::PrintNewFoundHashes() {
    std::vector<HashPlain>::iterator currentHash;

    this->HashFileMutex.lock();
    
    // Loop through all hashes.
    for (currentHash = this->Hashes.begin(); currentHash < this->Hashes.end(); currentHash++) {
        // Skip if already found.
        if (currentHash->passwordFound && !currentHash->passwordPrinted) {
            this->PrintHash(*currentHash);
            currentHash->passwordPrinted = 1;
        }
    }
    this->HashFileMutex.unlock();
}


int CHHashFileVPlain::OutputUnfoundHashesToFile(std::string filename) {
    std::ofstream hashFile;
    std::vector<HashPlain>::iterator it;

    this->HashFileMutex.lock();
    
    hashFile.open(filename.c_str(), std::ios_base::out);
    //Hope you didn't need the previous contents of this file for anything.
    
    if (hashFile.good()) {
        for (it = this->Hashes.begin(); it < this->Hashes.end(); it++) {
            if (!it->passwordFound)
            {
                for (uint32_t j=0; j < this->HashLengthBytes; j++)
                {
                    hashFile<<HEX(it->hash[j]);
                }
                hashFile<<std::endl; 
            } 
        }            
    }
    this->HashFileMutex.unlock();
    if (hashFile.good())
        return 1;
    else
        return 0;
}

void CHHashFileVPlain::ImportHashListFromRemoteSystem(std::string & remoteData) {
   // I hope your CHHashFileVPlain was empty.
    // For cleanliness, I will clean this now.
    this->HashFileMutex.lock();
    
    this->Protobuf.Clear();
    this->Hashes.clear();
    this->HashLengthBytes = 0;
    this->ExportProtobufCache.clear();

    std::string hashBuffer;
    //Unpack protobuf
    this->Protobuf.ParseFromString(remoteData);
    //Get number of hashes in protobuf
    this->TotalHashes = this->Protobuf.hash_value_size();
    this->HashLengthBytes = this->Protobuf.hash_length_bytes();
    //Make individual HashPlain packages for each hash
    for(uint64_t i=0; i<this->TotalHashes; i++)
    {
        CHHashFileVPlain::HashPlain newHashPlain;
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

void CHHashFileVPlain::ExportHashListToRemoteSystem(std::string *exportData) {

    this->HashFileMutex.lock();
    // If the cache is valid, simply return it.
    if (this->ExportProtobufCache.size()) {
        *exportData = this->ExportProtobufCache;
        this->HashFileMutex.unlock();
        return;
    }
    
    // Cache is not valid - create a new protobuf to export.
    this->Protobuf.Clear();
    std::string hashBuffer;
    
    std::vector<CHHashFileVPlain::HashPlain>::iterator i;
    for(i=this->Hashes.begin();i<this->Hashes.end();i++)
    {
        hashBuffer = std::string(i->hash.begin(), i->hash.end());
        this->Protobuf.add_hash_value(hashBuffer); 
    }
    this->Protobuf.set_hash_length_bytes(this->HashLengthBytes);
    
    //Danger: Please be sure to have some storage allocated to this pointer.
    //I shouldn't have to say this, but I will anyway.
    this->Protobuf.SerializeToString(exportData);
    
    // Store the cached result.
    this->ExportProtobufCache = *exportData;
    this->HashFileMutex.unlock();
}



bool CHHashFileVPlain::PlainHashSortPredicate(const HashPlain &d1, const HashPlain &d2) {
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


bool CHHashFileVPlain::PlainHashUniquePredicate(const HashPlain &d1, const HashPlain &d2) {
    // Hashes are only equal if the hash is equal and the user data is equal.
    if (memcmp(&d1.hash[0], &d2.hash[0], d1.hash.size()) == 0) {
        if (d1.userData == d2.userData) {
            return 1;
        }
        return 0;
    }
    return 0;
}

void CHHashFileVPlain::SortHashes() {
    // Sort hashes and remove duplicates.
    std::sort(this->Hashes.begin(), this->Hashes.end(), CHHashFileVPlain::PlainHashSortPredicate);
    this->Hashes.erase(
        std::unique(this->Hashes.begin(), this->Hashes.end(), CHHashFileVPlain::PlainHashUniquePredicate ),
        this->Hashes.end() );
}

void CHHashFileVPlain::PrintHash(HashPlain &Hash) {
    size_t position;
    
    if (this->UseJohnOutputStyle) {
        // Print John fields (username/password) only
        if (Hash.userData.size()) {
            std::cout << Hash.userData << this->OutputSeparator;
        }
        for (position = 0; position < Hash.password.size(); position++) {
                std::cout << (char)Hash.password[position];
        }
    } else {
        // Print all fields
        if (Hash.userData.size()) {
            std::cout << Hash.userData << this->OutputSeparator;
        }
        if (this->printAlgorithm) {
            std::cout << 
                getHashFunctionByDefinedByte(Hash.algorithmType) << 
                this->OutputSeparator;
        }
        for (position = 0; position < Hash.hash.size(); position++) {
            std::cout << HEX(Hash.hash[position]);
        }
        std::cout<<this->OutputSeparator;
        for (position = 0; position < Hash.password.size(); position++) {
                std::cout << (char)Hash.password[position];
        }
        if (this->AddHexOutput) {
            std::cout << this->OutputSeparator << "0x";
            for (position = 0; position < Hash.password.size(); position++) {
                    std::cout << HEX(Hash.password[position]);

            }
        }
    }
    std::cout << std::endl;
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
    
    CHHashFileVPlain HashFile(16);
    HashFile.testPHPPassHash();
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
    
    
    /*
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
    
    HashFile.OpenHashFile(argv[1]);
    printf("Loaded hash file.\n");
    Hashes = HashFile.ExportUncrackedHashList();
    
    printf("Exported hashes: \n");
    for (i = 0; i < Hashes.size(); i++) {
        for (int j = 0; j < Hashes[i].size(); j++) {
            printf("%02x", Hashes[i][j]);
        }
        printf("\n");
    }
    
    // Report every other hash as found.
    for (i = 0; i < Hashes.size(); i += 1) {
        HashFile.ReportFoundPassword(Hashes[i], FoundPassword);
    }
    HashFile.SetAddHexOutput(true);
    HashFile.SetUseJohnOutputStyle(true);
    //HashFile.SetOutputSeparator('-');
    
    HashFile.PrintAllFoundHashes();
    */
}

#endif
