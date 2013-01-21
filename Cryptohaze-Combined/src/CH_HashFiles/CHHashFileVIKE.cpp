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

#include "CH_HashFiles/CHHashFileVIKE.h"

#include <stdlib.h>
#include <string.h>

CHHashFileVIKE::CHHashFileVIKE() {
    this->HashLengthBytes = 0;
    this->IKEHashes.clear();
    this->SaltLength;
}

// Private member functions - stubs for now.

int CHHashFileVIKE::OutputFoundHashesToFile() {
    return 0;
}

void CHHashFileVIKE::SortHashes() {
    // Sort hashes and remove duplicates.
    std::sort(this->IKEHashes.begin(), this->IKEHashes.end(), CHHashFileVIKE::IKEHashSortPredicate);
    this->IKEHashes.erase(
        std::unique(this->IKEHashes.begin(), this->IKEHashes.end(), CHHashFileVIKE::IKEHashUniquePredicate ),
        this->IKEHashes.end() );
}

bool CHHashFileVIKE::IKEHashSortPredicate(const SaltedHashIKE &d1, const SaltedHashIKE &d2) {
    int i;
    for (i = 0; i < d1.hash_r.size(); i++) {
        if (d1.hash_r[i] == d2.hash_r[i]) {
            continue;
        } else if (d1.hash_r[i] > d2.hash_r[i]) {
            return 0;
        } else if (d1.hash_r[i] < d2.hash_r[i]) {
            return 1;
        }
    }
    // Exactly equal = return 0.
    return 0;
}

bool CHHashFileVIKE::IKEHashUniquePredicate(const SaltedHashIKE &d1, const SaltedHashIKE &d2) {
    if (memcmp(&d1.hash_r[0], &d2.hash_r[0], d1.hash_r.size()) == 0) {
        return 1;
    }
    return 0;
}

int CHHashFileVIKE::OpenHashFile(std::string filename) {
    FILE *hashfile;
    char buffer[4096];
    uint32_t currentPos;
    SaltedHashIKE HashEntry;
    uint32_t currentLine = 0;

    // Buffers for reading in values.
    std::vector<char> readValueHex;
    std::vector<uint8_t> readValueBinary;

    //printf("Opening hash file %s\n", filename.c_str());

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

        readValueBinary = convertAsciiToBinary(readValueHex);
        // Append to hash_r_data vector
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read g_xi ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiToBinary(readValueHex);
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read cky_r ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiToBinary(readValueHex);
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read cky_i ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiToBinary(readValueHex);
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read sai_b ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiToBinary(readValueHex);
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read idir_b ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiToBinary(readValueHex);
        HashEntry.hash_r_data.insert(HashEntry.hash_r_data.end(), readValueBinary.begin(), readValueBinary.end());

        // Done with hash_r_data.  Now for skeyid_data.

        // ==== Read ni_b ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiToBinary(readValueHex);
        HashEntry.skeyid_data.insert(HashEntry.skeyid_data.end(), readValueBinary.begin(), readValueBinary.end());

        // ==== Read nr_b ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiToBinary(readValueHex);
        HashEntry.skeyid_data.insert(HashEntry.skeyid_data.end(), readValueBinary.begin(), readValueBinary.end());

        // Finally, read hash_r
        // ==== Read hash_r ====
        readValueHex.clear();
        while ((buffer[currentPos] != ':') && (buffer[currentPos] != '\n')) {
            readValueHex.push_back(buffer[currentPos]);
            currentPos++;
        }
        currentPos++;
        readValueBinary = convertAsciiToBinary(readValueHex);
        HashEntry.hash_r.insert(HashEntry.hash_r.end(), readValueBinary.begin(), readValueBinary.end());

        //printf("Got hash_r_data length: %d\n", (int)HashEntry.hash_r_data.size());
        //printf("Got skeyid_data length: %d\n", (int)HashEntry.skeyid_data.size());
        //printf("Got hash_r length: %d\n", (int)HashEntry.hash_r.size());
        this->IKEHashes.push_back(HashEntry);

    }
    this->TotalHashes = this->IKEHashes.size();
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = this->TotalHashes;
    this->SortHashes();
    return 1;
}

std::vector<std::vector<uint8_t> > CHHashFileVIKE::ExportUncrackedHashList() {
    std::vector<std::vector<uint8_t> > returnValue;
    
    return returnValue;
}


int CHHashFileVIKE::ReportFoundPassword(std::vector<uint8_t> hash, std::vector<uint8_t> password) {
    return 0;
}

void CHHashFileVIKE::PrintAllFoundHashes() {
    
}

void CHHashFileVIKE::PrintNewFoundHashes() {
    
}

int CHHashFileVIKE::OutputUnfoundHashesToFile(std::string filename) {
    return 0;
}

void CHHashFileVIKE::ImportHashListFromRemoteSystem(std::string & remoteData) {
    
}
void CHHashFileVIKE::ExportHashListToRemoteSystem(std::string * exportData) {
    
}

std::vector<CHHashFileVIKE_IKEHashData> CHHashFileVIKE::ExportUncrackedIKEHashes() {
    std::vector<CHHashFileVIKE_IKEHashData> returnVector; 
    CHHashFileVIKE_IKEHashData dataItem;
    std::vector<SaltedHashIKE>::iterator HashListIterator;
   
    for (HashListIterator = this->IKEHashes.begin();
            HashListIterator < this->IKEHashes.end(); HashListIterator++) {
        dataItem.hash_r = HashListIterator->hash_r;
        dataItem.hash_r_data = HashListIterator->hash_r_data;
        dataItem.skeyid_data = HashListIterator->skeyid_data;
        returnVector.push_back(dataItem);
    }    
    return returnVector;
}


#define UNIT_TEST 1
#ifdef UNIT_TEST
#include <stdlib.h>

int main(int argc, char *argv[]) {
    printf("oh hai\n");
    
    std::vector<CHHashFileVIKE_IKEHashData> hashes;
    
    if (argc != 2) {
        printf("Usage: %s [ike psk file]\n", argv[0]);
        exit(1);
    }

    CHHashFileVIKE HashFileIKE;

    HashFileIKE.OpenHashFile(argv[1]);
    
    hashes = HashFileIKE.ExportUncrackedIKEHashes();
    
    printf("size of items:\n");
    printf("hash_r: %d\n", hashes[0].hash_r.size());
    printf("hash_r_data: %d\n", hashes[0].hash_r_data.size());
    printf("skeyid_data: %d\n", hashes[0].skeyid_data.size());
    
}

#endif