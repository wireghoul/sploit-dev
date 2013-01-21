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

#include "CH_HashFiles/CHHashFileVSHA.h"

void CHHashFileVSHA::PrintHashToHandle(HashPlain &Hash, FILE *stream) {
    std::vector<uint8_t> base64Hash = this->base64Encode(Hash.hash);
    // Add a null terminator to the base64 encoded string so it is a valid cstr.
    base64Hash.push_back(0);

    if (this->UseJohnOutputStyle) {
        // Print John fields (username/password) only
        if (Hash.userData.size()) {
            fprintf(stream, "%s%c", Hash.userData.c_str(), this->OutputSeparator);
        }
        for (int pos = 0; pos < Hash.password.size(); pos++) {
            fprintf(stream, "%c", (char)Hash.password[pos]);
        }
        fprintf(stream, "\n");
    } else {
        // Print all fields
        if (Hash.userData.size()) {
            fprintf(stream, "%s%c", Hash.userData.c_str(), this->OutputSeparator);
        }
        if (this->printAlgorithm) {
            fprintf(stream, "%s%c",
                    getHashFunctionByDefinedByte(Hash.algorithmType).c_str(),
                    this->OutputSeparator);
        }
        fprintf(stream, "{SHA}%s%c", (char *)&base64Hash[0], this->OutputSeparator);
        for (int pos = 0; pos < Hash.password.size(); pos++) {
            fprintf(stream, "%c", (char)Hash.password[pos]);
        }
        if (this->AddHexOutput) {
            fprintf(stream, "%c0x", this->OutputSeparator);
            for (int pos = 0; pos < Hash.password.size(); pos++) {
                fprintf(stream, "%02x", (char)Hash.password[pos]);
            }
        }
        fprintf(stream, "\n");
    }
}

void CHHashFileVSHA::PrintHash(HashPlain &Hash) {
    this->PrintHashToHandle(Hash, stdout);
}

// SHA hashes are always 20 bytes long
CHHashFileVSHA::CHHashFileVSHA() : CHHashFileVPlain(20){ }

int CHHashFileVSHA::OpenHashFile(std::string filename) {
    std::ifstream hashFile;
    std::string fileLine;
    std::string userData, hashData;
    std::vector<uint8_t> hashVector;
    HashPlain HashVectorEntry;
    uint32_t fileLineCount = 0;
    
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
        // Be nice to users and 1-index the lines
        fileLineCount++;
        HashVectorEntry.hash.clear();
        HashVectorEntry.userData.clear();
        found=fileLine.find_last_not_of(whitespaces);
        if (found!=std::string::npos)
            fileLine.erase(found+1);
        else
            fileLine.clear();
        
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
        if ((hashData.length() > 0) && (hashData.length() != 33))
        {
            printf("Incorrect length: Line %d\n", fileLineCount);
            continue;
        }
        
        // Check for the '{SHA}' prefix - if not found, continue.
        if ((hashData[0] != '{') || (hashData[1] != 'S') || (hashData[2] != 'H')
                || (hashData[3] != 'A') || (hashData[4] != '}')) {
            printf("Prefix not found: Line %d\n", fileLineCount);
            continue;
        }
        
        // If it's a valid line, do the work.
        if (hashData.length() > 0) {
            // Load the base64 part of the hash - past the {SHA} prefix.
            hashVector = std::vector<uint8_t>(hashData.begin() + 5,
                    hashData.end());
            HashVectorEntry.userData = userData;
            HashVectorEntry.hash = this->base64Decode(hashVector);
            if (HashVectorEntry.hash.size() != 20) {
                printf("Hash data length error: Line %d\n", fileLineCount);
                continue;
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

int CHHashFileVSHA::OutputFoundHashesToFile() {
    std::vector<HashPlain>::iterator currentHash;

    // Lock is held by the password reporting function
    
    // Ensure the output file is opened for access before trying to write to it.
    if (this->OutputFile) {
        for (currentHash = this->Hashes.begin();
                currentHash < this->Hashes.end(); currentHash++) {
            // Skip if already reported.
            if (currentHash->passwordFound && !currentHash->passwordOutputToFile) {
                this->PrintHashToHandle(*currentHash, this->OutputFile);
                // Mark hash as reported.
                currentHash->passwordOutputToFile = 1;
            }
        }
    }
    fflush(this->OutputFile);

    return 1;
}

int CHHashFileVSHA::OutputUnfoundHashesToFile(std::string filename) {
    FILE *unfoundFile;
    std::vector<HashPlain>::iterator currentHash;
    std::vector<uint8_t> base64Hash;

    this->HashFileMutex.lock();
    // Overwrite old data
    unfoundFile = fopen(filename.c_str(), "w");
    // Return false if unable to open for writing
    if (!unfoundFile) {
        this->HashFileMutex.unlock();
        return 0;
    }

    for (currentHash = this->Hashes.begin();
            currentHash < this->Hashes.end(); currentHash++) {
        // Only print if not found.
        if (!currentHash->passwordFound) {
            base64Hash = this->base64Encode(currentHash->hash);
            base64Hash.push_back(0);
            if (currentHash->userData.size()) {
                // Use the input separator, as that's what it came in with.
                fprintf(unfoundFile, "%s%c", currentHash->userData.c_str(),
                        this->InputDelineator);
            }
            fprintf(unfoundFile, "{SHA}%s\n", (char *)&base64Hash[0]);
        }
    }

    fclose(unfoundFile);
    
    this->HashFileMutex.unlock();
    return 1;
}

#define UNIT_TEST_SHA 1
#if UNIT_TEST_SHA
#include <string.h>

static char foundPasswordStringSHA[] = "SHAPassword";

int main(int argc, char *argv[]) {
    
    CHHashFileVSHA HashFile;
    std::vector<std::vector<uint8_t> > Hashes;
    std::vector<uint8_t> FoundPassword;
    uint32_t i;
    
    if (argc != 2) {
        printf("program hashfile\n");
        exit(1);
    }

    for (i = 0; i < strlen(foundPasswordStringSHA); i++) {
        FoundPassword.push_back(foundPasswordStringSHA[i]);
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
    for (i = 0; i < Hashes.size(); i += 2) {
        HashFile.ReportFoundPassword(Hashes[i], FoundPassword);
    }
    HashFile.SetAddHexOutput(true);
    //HashFile.SetUseJohnOutputStyle(true);
    //HashFile.SetOutputSeparator('-');
    
    HashFile.PrintAllFoundHashes();
    
    HashFile.OutputUnfoundHashesToFile("/tmp/notfound.hash");
}

#endif
