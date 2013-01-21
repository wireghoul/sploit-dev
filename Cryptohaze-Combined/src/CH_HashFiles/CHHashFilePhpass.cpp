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

#include "CH_HashFiles/CHHashFilePhpass.h"

//#define CHHASHFILEVPHPASS_DEBUG 1
#if CHHASHFILEVPHPASS_DEBUG
#define phpass_printf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define phpass_printf(fmt, ...) do {} while (0)
#endif

static std::string PhpassBase64 = "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

CHHashFilePhpass::CHHashFilePhpass() : CHHashFileSalted(16, 0, 0, 0, ':') {}

void CHHashFilePhpass::parseFileLine(std::string fileLine, size_t lineNumber) {
    std::string userData, hashData;

    std::vector<uint8_t> hashBase64Data;
    std::vector<uint8_t> hashDecodedData;

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

    
    if ((fileLine.substr(0, 3) != std::string(PHPBB_MAGIC_BYTES)) &&
        (fileLine.substr(0, 3) != std::string(PHPASS_MAGIC_BYTES))) {
        printf("Error: Hash on line %u missing '$H$'/'$P$' prefix\n",
                (unsigned int) lineNumber);
        return;
    }
        
    phpass_printf("Loaded hash %s\n", fileLine.c_str());
    

    // If it's a valid line, do the work.
    if (hashData.length() > 0) {
        phpass_printf("Loaded hash %s\n", fileLine.c_str());
        // Get the iteration count value
        phpass_printf("Got iteration character %c\n", fileLine[3]);
        HashVectorEntry.iterationCount = 1 << PhpassBase64.find(fileLine[3]);
        phpass_printf("Iteration count: %d\n", HashVectorEntry.iterations);
        
        // Copy the salt into the proper location
        for (size_t i = 4; i < 12; i++) {
            HashVectorEntry.salt.push_back(fileLine[i]);
        }
        phpass_printf("Salt: ");
        for (size_t i = 0; i < HashVectorEntry.salt.size(); i++) {
            phpass_printf("%c", HashVectorEntry.salt[i]);
        }
        phpass_printf("\n");
        
        // Copy the base64 encoded data into a buffer for decoding
        hashBase64Data.clear();
        for (size_t i = 12; i < fileLine.size(); i++) {
            hashBase64Data.push_back(fileLine[i]);
        }
        phpass_printf("Base64 encoded data: ");
        for (size_t i = 0; i < hashBase64Data.size(); i++) {
            phpass_printf("%c", hashBase64Data[i]);
        }
        phpass_printf("\n");
        
        
        hashDecodedData = this->phpHash64Decode(hashBase64Data, PhpassBase64);
        phpass_printf("Decoded data: ");
        for (int i = 0; i < hashDecodedData.size(); i++) {
            phpass_printf("%02x", hashDecodedData[i]);
        }
        phpass_printf("\n");
        phpass_printf("Decoded data length: %d\n", (int)hashDecodedData.size());
        
        HashVectorEntry.hash = hashDecodedData;
        
        HashVectorEntry.originalHash = fileLine;
        this->SaltedHashes.push_back(HashVectorEntry);
    }
}


//#define UNIT_TEST_PHPASS 1
#if UNIT_TEST_PHPASS
#include <string.h>

static char foundPasswordStringPhpass[] = "PhpassPassword";

int main(int argc, char *argv[]) {
    
    CHHashFilePhpass HashFile;
    std::vector<std::vector<uint8_t> > Hashes;
    std::vector<uint8_t> FoundPassword;
    uint32_t i;
    CHHashFileSaltedDataBlob SaltExport;

    if (argc != 2) {
        printf("program hashfile\n");
        exit(1);
    }

    for (i = 0; i < strlen(foundPasswordStringPhpass); i++) {
        FoundPassword.push_back(foundPasswordStringPhpass[i]);
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
