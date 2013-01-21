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

#include "CH_HashFiles/CHHashFileSSHA.h"


// SHA hashes are always 20 bytes long
CHHashFileSSHA::CHHashFileSSHA() : CHHashFileSalted(20, 16, 0, 0, ':') {}

void CHHashFileSSHA::parseFileLine(std::string fileLine, size_t lineNumber) {
    std::string userData, hashData;
    std::vector<uint8_t> rawVector, hashVector;
    HashSalted HashVectorEntry;
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

    // Check for the '{SSHA}' prefix - if not found, continue.
    if ((hashData[0] != '{') || (hashData[1] != 'S') || (hashData[2] != 'S')
                || (hashData[3] != 'H') || (hashData[4] != 'A')
                || (hashData[5] != '}')) {
        printf("Prefix not found in line %u\n",
                (unsigned int) lineNumber);
        return;
    }

    // If it's a valid line, do the work.
    if (hashData.length() > 0) {
        // Load the base64 part of the hash - past the {SSHA} prefix.
        hashVector = std::vector<uint8_t>(hashData.begin() + 6,
                hashData.end());
        HashVectorEntry.userData = userData;
        rawVector = this->base64Decode(hashVector);
        if (rawVector.size() <= 20) {
            printf("Hash data length error: Line %u\n", (unsigned int) lineNumber);
            return;
        }
        // The first 20 bytes are the SHA1 hash
        HashVectorEntry.hash.assign(rawVector.begin(),
                rawVector.begin() + 20);
        // The remaining bytes are the salt.
        HashVectorEntry.originalSalt.assign(rawVector.begin() + 20,
                rawVector.end());
        HashVectorEntry.salt = HashVectorEntry.originalSalt;
        HashVectorEntry.originalHash = fileLine;
        this->SaltedHashes.push_back(HashVectorEntry);
    }
}


//#define UNIT_TEST_SSHA 1
#if UNIT_TEST_SSHA
#include <string.h>

static char foundPasswordStringSSHA[] = "SSHAPassword";

int main(int argc, char *argv[]) {
    
    CHHashFileSSHA HashFile;
    std::vector<std::vector<uint8_t> > Hashes;
    std::vector<uint8_t> FoundPassword;
    uint32_t i;
    CHHashFileSaltedDataBlob SaltExport;

    if (argc != 2) {
        printf("program hashfile\n");
        exit(1);
    }

    for (i = 0; i < strlen(foundPasswordStringSSHA); i++) {
        FoundPassword.push_back(foundPasswordStringSSHA[i]);
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

    SaltExport = HashFile.exportUniqueSaltedData();
    printf("Exported salts:\n");
    for (i = 0; i < SaltExport.SaltData.size(); i++) {
        for (int j = 0; j < SaltExport.SaltData[i].size(); j++) {
            printf("%02x", SaltExport.SaltData[i][j]);
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
