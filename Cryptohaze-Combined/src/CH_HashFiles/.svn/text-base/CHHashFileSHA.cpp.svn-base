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

#include "CH_HashFiles/CHHashFileSHA.h"


// SHA hashes are always 20 bytes long
CHHashFileSHA::CHHashFileSHA() : CHHashFilePlain(20){ }

void CHHashFileSHA::parseFileLine(std::string fileLine, size_t lineNumber) {
    std::string userData, hashData;
    std::vector<uint8_t> hashVector;
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
        printf("Hash on line %u: Incorrect length (%d, want 33)\n",
                (unsigned int) lineNumber, (int)hashData.length());
        return;
    }

    // Check for the '{SHA}' prefix - if not found, continue.
    if ((hashData[0] != '{') || (hashData[1] != 'S') || (hashData[2] != 'H')
            || (hashData[3] != 'A') || (hashData[4] != '}')) {
        printf("Prefix not found in line %u\n",
                (unsigned int) lineNumber);
        return;
    }

    // If it's a valid line, do the work.
    if (hashData.length() > 0) {
        // Load the base64 part of the hash - past the {SHA} prefix.
        hashVector = std::vector<uint8_t>(hashData.begin() + 5,
                hashData.end());
        HashVectorEntry.userData = userData;
        HashVectorEntry.hash = this->base64Decode(hashVector);
        if (HashVectorEntry.hash.size() != 20) {
            printf("Hash data length error: Line %u\n", (unsigned int) lineNumber);
            return;
        }
        HashVectorEntry.originalHash = fileLine;
        this->Hashes.push_back(HashVectorEntry);
    }
}


#define UNIT_TEST_SHA 1
#if UNIT_TEST_SHA
#include <string.h>

static char foundPasswordStringSHA[] = "SHAPassword";

int main(int argc, char *argv[]) {
    
    CHHashFileSHA HashFile;
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
    //HashFile.SetUseJohnOutputStyle(true);
    //HashFile.SetOutputSeparator('-');
    
    HashFile.printAllFoundHashes();
    
    HashFile.outputUnfoundHashesToFile("/tmp/notfound.hash");
}

#endif
