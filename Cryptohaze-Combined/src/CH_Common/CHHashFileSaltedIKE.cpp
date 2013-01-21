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

#include "CH_Common/CHHashFileSaltedIKE.h"

#include <stdlib.h>
#include <string.h>

// Converts a vector of hex bytes to binary
std::vector<uint8_t> convertAsciiVectorToBinary(std::vector<char> inputVector) {
  char convertSpace[3];
  uint32_t result;
  int i;
  std::vector<uint8_t> returnVector;

  //TODO: Fix this code to not suck and use scanf
  // Loop until either maxLength is hit, or strlen(intput) / 2 is hit.
  for (i = 0; (i < inputVector.size() / 2); i++) {
    convertSpace[0] = inputVector[2 * i];
    convertSpace[1] = inputVector[2 * i + 1];
    convertSpace[2] = 0;
    sscanf(convertSpace, "%2x", &result);
    // Do this to prevent scanf from overwriting memory with a 4 byte value...
    returnVector.push_back((uint8_t)result & 0xff);
  }
  return returnVector;
}


// Salt length is a max.  Can be longer than actual salt.
CHHashFileSaltedIKE::CHHashFileSaltedIKE() {
    this->hashLength = 0;
    this->saltLength = 0;
    this->outputFoundHashesToFile = 0;
}

CHHashFileSaltedIKE::~CHHashFileSaltedIKE() {

}

int CHHashFileSaltedIKE::OpenHashFile(std::string filename) {

    FILE *hashfile;
    char buffer[4096];
    uint32_t i;
    uint32_t currentPos;
    SaltedHashIKE HashEntry;
    uint32_t currentLine = 0;

    // Buffers for reading in values.
    std::vector<char> readValueHex;
    std::vector<uint8_t> readValueBinary;

    printf("Opening hash file %s\n", filename.c_str());

    hashfile = fopen(filename.c_str(), "r");
    if (!hashfile) {
        printf("Cannot open hash file %s.  Exiting.\n", filename.c_str());
        exit(1);
    }


    while (!feof(hashfile)) {
        currentLine++;

        HashEntry.hash_r.clear();
        HashEntry.hash_r_data.clear();
        HashEntry.skeyid_data.clear();
        HashEntry.password.clear();
        HashEntry.passwordFound = 0;
        HashEntry.passwordOutputToFile = 0;
        HashEntry.passwordReported = 0;

        memset(buffer, 0, 4096);

       // If fgets returns NULL, there's been an error or eof.  Continue.
        if (!fgets(buffer, 4096, hashfile)) {
            continue;
        }
 
        // Read from the file in the following colon separated format:
        // g_xr, g_xi, cky_r, cky_i, sai_b, idir_b, ni_b, nr_b, hash_r
        // hash_r_data = g_xr | g_xi | cky_r | cky_i | sai_b | idir_b
        // skeyid_data = ni_b | nr_b

        // ==== Read g_xr ====
        currentPos = 0;
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        // Add a position for the colon
        currentPos++;

        readValueBinary = convertAsciiVectorToBinary(readValueHex);
        // Append to hash_r_data vector
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read g_xi ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiVectorToBinary(readValueHex);
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read cky_r ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiVectorToBinary(readValueHex);
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read cky_i ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiVectorToBinary(readValueHex);
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read sai_b ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiVectorToBinary(readValueHex);
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read idir_b ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiVectorToBinary(readValueHex);
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // Done with hash_r_data.  Now for skeyid_data.

        // ==== Read ni_b ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiVectorToBinary(readValueHex);
        HashEntry.skeyid_data.insert(HashEntry.skeyid_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read nr_b ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiVectorToBinary(readValueHex);
        HashEntry.skeyid_data.insert(HashEntry.skeyid_data.end(), readValueBinary.begin(), readValueBinary.end());

        // Finally, read hash_r
        // ==== Read hash_r ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiVectorToBinary(readValueHex);
        HashEntry.hash_r.insert(HashEntry.hash_r.end(), readValueBinary.begin(), readValueBinary.end());

        //printf("Got hash_r_data length: %d\n", (int)HashEntry.hash_r_data.size());
        //printf("Got skeyid_data length: %d\n", (int)HashEntry.skeyid_data.size());
        //printf("Got hash_r length: %d\n", (int)HashEntry.hash_r.size());
        this->HashList.push_back(HashEntry);

    }
    this->TotalHashes = this->HashList.size();
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = this->TotalHashes;
    //this->SortHashList();
    return 1;
}

#define UNIT_TEST 1
#if UNIT_TEST

int main() {
    printf("oh hai\n");


    CHHashFileSaltedIKE HashFileIKE;

    HashFileIKE.OpenHashFile("../Downloads/psk-md5.txt");
}

#endif