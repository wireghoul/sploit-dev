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

// TODO: Consider race conditions in collecting salt data vs other data.
// Perhaps emit all of it in a structure?

//#define TRACE_PRINTF 1

#include "CH_HashFiles/CHHashFileVPhpass.h"

#include "MFN_Common/MFNDebugging.h"
#include <string.h>

//#define CHHASHFILEVPHPASS_DEBUG 1
#if CHHASHFILEVPHPASS_DEBUG
#define phpass_printf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define phpass_printf(fmt, ...) do {} while (0)
#endif

// Phpass uses a weird base64 encoding.
static std::string PhpassBase64 = "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

CHHashFileVPhpass::CHHashFileVPhpass() {
    // Ensure the structures are clear.
    this->Hashes.clear();
    this->UniqueSalts.clear();
    this->UniqueSaltValues.clear();
    this->UniqueSaltIterationCounts.clear();
    this->UniqueSaltsValid = 0;

    this->TotalHashes = 0;
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = 0;
}


int CHHashFileVPhpass::OutputFoundHashesToFile() {
    
}

void CHHashFileVPhpass::SortHashes() {
    trace_printf("CHHashFileVPhpass::SortHashes()\n");
    // Sort hashes and remove duplicates.
    std::sort(this->Hashes.begin(), this->Hashes.end(),
            CHHashFileVPhpass::PhpassHashSortPredicate);
    this->Hashes.erase(
        std::unique(this->Hashes.begin(), this->Hashes.end(),
            CHHashFileVPhpass::PhpassHashUniquePredicate),
        this->Hashes.end());
}

void CHHashFileVPhpass::ExtractUncrackedSalts() {
    trace_printf("CHHashFileVPhpass::ExtractUncrackedSalts()\n");
    HashPhpassSalt PhpassSalt;
    // This function must be protected by a mutex outside!
    
    // Clear out the old salts.
    this->UniqueSalts.clear();
    
    // Loop through the hashes, copying unfound salts into the new structure.
    std::vector<HashPhpass>::iterator HashesIt;
    for (HashesIt = this->Hashes.begin(); HashesIt < this->Hashes.end(); 
            HashesIt++) {
        
        if (!HashesIt->passwordFound) {
            PhpassSalt.salt = HashesIt->salt;
            PhpassSalt.iterations = HashesIt->iterations;
            this->UniqueSalts.push_back(PhpassSalt);
        }
    }
    
    // Sort and unique the salts.
    std::sort(this->UniqueSalts.begin(), this->UniqueSalts.end(),
            CHHashFileVPhpass::PhpassSaltSortPredicate);
    this->UniqueSalts.erase(
        std::unique(this->UniqueSalts.begin(), this->UniqueSalts.end(),
            CHHashFileVPhpass::PhpassSaltUniquePredicate),
        this->UniqueSalts.end());
    
    // Extract out the salt values and the iteration count values.
    std::vector<HashPhpassSalt>::iterator SaltsIt;

    this->UniqueSaltValues.clear();
    this->UniqueSaltIterationCounts.clear();
    
    this->UniqueSaltValues.reserve(this->UniqueSalts.size());
    this->UniqueSaltIterationCounts.reserve(this->UniqueSalts.size());

    for (SaltsIt = this->UniqueSalts.begin(); SaltsIt < this->UniqueSalts.end(); 
            SaltsIt++) {
        this->UniqueSaltValues.push_back(SaltsIt->salt);
        this->UniqueSaltIterationCounts.push_back(SaltsIt->iterations);
    }
    
    this->UniqueSaltsValid = 1;
}

/**
 * Returns true if d1.hash less than d2.hash.  Only compares the hash values -
 * the salts are handled separately after extraction.
 * 
 * @param d1
 * @param d2
 * @return 
 */
bool CHHashFileVPhpass::PhpassHashSortPredicate(const HashPhpass &d1, const HashPhpass &d2) {
    // Get the minimum length of either hash.
    int length = (d1.hash.size() < d2.hash.size()) ? d1.hash.size() : d2.hash.size();

    // If d1.hash < d2.hash, return true, else return false.
    if (memcmp(&d1.hash[0], &d2.hash[0], length) < 0) {
        return 1;
    }
    return 0;
}
    
bool CHHashFileVPhpass::PhpassHashUniquePredicate(const HashPhpass &d1, const HashPhpass &d2) {
    // Ensure that the hash, salt, and iteration count are all identical for it
    // to get removed.
    
    // If d1.hash != d2.hash, return false.
    if (d1.hash != d2.hash) {
        return 0;
    }
    // Check the salts
    if (d1.salt != d2.salt) {
        return 0;
    }
    // And iterations
    if (d1.iterations != d2.iterations) {
        return 0;
    }
    
    // All checks pass - hashes are identical.
    return 1;
}

bool CHHashFileVPhpass::PhpassSaltSortPredicate(const HashPhpassSalt &d1, const HashPhpassSalt &d2) {
    
    // If the iteration count of d1 is not the same, sort by this first.
    if (d1.iterations < d2.iterations) {
        return 1;
    } else if (d1.iterations > d2.iterations) {
        return 0;
    }
    
    // Iteration count is the same - continue testing.
    
    // Get the minimum length of either hash.
    int length = (d1.salt.size() < d2.salt.size()) ? d1.salt.size() : d2.salt.size();

    // If d1.hash < d2.hash, return true, else return false.
    if (memcmp(&d1.salt[0], &d2.salt[0], length) < 0) {
        return 1;
    }
    return 0;
}
    
bool CHHashFileVPhpass::PhpassSaltUniquePredicate(const HashPhpassSalt &d1, const HashPhpassSalt &d2) {
    // If salt or iterations do not match, salts are not identical.
    if (d1.salt != d2.salt) {
        return 0;
    }
    if (d1.iterations != d2.iterations) {
        return 0;
    }
    // Salts must be identical.
    return 1;
}

int CHHashFileVPhpass::OpenHashFile(std::string filename) {
    std::ifstream hashFile;
    std::string fileLine;
    std::string hashValue;
    std::string saltValue;
    HashPhpass HashVectorEntry;
    size_t separatorPos;
    uint64_t fileLineCount = 0;
    std::vector<uint8_t> hashBase64Data;
    std::vector<uint8_t> hashDecodedData;
    
    std::string whitespaces (" \t\f\v\n\r");
    size_t found;
    
    HashVectorEntry.passwordFound = 0;
    HashVectorEntry.passwordOutputToFile = 0;
    HashVectorEntry.passwordPrinted = 0;
    
    hashBase64Data.reserve(128);
    
    hashFile.open(filename.c_str(), std::ios_base::in);
    if (!hashFile.good())
    {
        
        std::cout << "ERROR: Cannot open hashfile " << filename <<"\n";
        exit(1);
    }
    
    while (std::getline(hashFile, fileLine)) {
        HashVectorEntry.hash.clear();
        HashVectorEntry.salt.clear();
        HashVectorEntry.originalHashString.clear();
        
        found=fileLine.find_last_not_of(whitespaces);
        if (found!=std::string::npos)
            fileLine.erase(found+1);
        else
            fileLine.clear();
        phpass_printf("Line length: %d\n", (int)fileLine.length());
        
        // If the line length is 0, continue - blank line that we can ignore.
        if (fileLine.length() == 0) {
            continue;
        }
        
        if ((fileLine.substr(0, 3) != std::string(PHPBB_MAGIC_BYTES)) &&
            (fileLine.substr(0, 3) != std::string(PHPASS_MAGIC_BYTES))) {
            printf("Error: Hash on line %lu missing '$H$'/'$P$' prefix\n", fileLineCount);
        }
        
        phpass_printf("Loaded hash %s\n", fileLine.c_str());
        
        // Copy the hash into the printout line.
        HashVectorEntry.originalHashString = fileLine;
        
        // Get the iteration count value
        phpass_printf("Got iteration character %c\n", fileLine[3]);
        HashVectorEntry.iterations = 1 << PhpassBase64.find(fileLine[3]);
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
        
        this->Hashes.push_back(HashVectorEntry);

        fileLineCount++;
    }
    
    this->SortHashes();
    
    // Set total hashes and hashes remaining to size of hash vector.
    this->TotalHashes = this->Hashes.size();
    this->TotalHashesRemaining = this->TotalHashes;
    
    hashFile.close();
    
    this->ExtractUncrackedSalts();
    
    // If NO hashes are loaded, something is probably very wrong.
    if (this->TotalHashes == 0) {
        printf("No hashes loaded!\n");
        exit(1);
    }
    
    return 1;
}

std::vector<std::vector<uint8_t> > CHHashFileVPhpass::ExportUncrackedHashList() {
    trace_printf("CHHashFileVPhpass::ExportUncrackedHashList()\n");
    std::vector<std::vector<uint8_t> > ReturnHashes;
    this->HashFileMutex.lock();
    
    // Loop through the hashes, copying unfound hashes into the new structure.
    std::vector<HashPhpass>::iterator HashesIt;
    
    for (HashesIt = this->Hashes.begin(); HashesIt < this->Hashes.end(); 
            HashesIt++) {
        
        if (!HashesIt->passwordFound) {
            ReturnHashes.push_back(HashesIt->hash);
        }
    }

    this->HashFileMutex.unlock();
    return ReturnHashes;
}

CHHashFileVSaltedDataBlob CHHashFileVPhpass::ExportUniqueSaltedData() {
    trace_printf("CHHashFileVPhpass::ExportUniqueSaltedData()\n");
    
    CHHashFileVSaltedDataBlob returnBlob;
    
    this->HashFileMutex.lock();
    // If the salt cache is not valid, update it.
    if (!this->UniqueSaltsValid) {
        // Update the list of uncracked salts.
        this->ExtractUncrackedSalts();
    }
    
    this->HashFileMutex.unlock();
    
    returnBlob.SaltData = this->UniqueSaltValues;
    returnBlob.iterationCount = this->UniqueSaltIterationCounts;
    
    return returnBlob;
}

int CHHashFileVPhpass::ReportFoundPassword(std::vector<uint8_t> hash, std::vector<uint8_t> password) {
    trace_printf("CHHashFileVPhpass::ReportFoundPassword()\n");
    
    uint64_t i;
    int passwordsFound = 0;

    this->HashFileMutex.lock();

    for (i = 0; i < this->TotalHashes; i++) {
        if (hash == this->Hashes[i].hash) {
            // Only do this if the password is not already reported.
            if (!this->Hashes[i].passwordFound) {
                this->Hashes[i].password = password;
                this->Hashes[i].passwordFound = 1;
                this->TotalHashesFound++;
                this->TotalHashesRemaining--;
                // Output to a file if needed.
                passwordsFound++;
            }
        }
    }
    
    // Dump passwords to a file if they're found, and clear the cache.
    if (passwordsFound) {
        this->OutputFoundHashesToFile();
        this->clearCaches();
    }
    this->HashFileMutex.unlock();
    return passwordsFound;
}

void CHHashFileVPhpass::PrintAllFoundHashes() {
    trace_printf("CHHashFileVPhpass::PrintAllFoundHashes()\n");
    
    uint64_t i;
    int j;
    std::string pbuf;
    
    this->HashFileMutex.lock();
    for (i = 0; i < this->TotalHashes; i++) {
        if (this->Hashes[i].passwordFound) {
            pbuf = std::string(this->Hashes[i].password.begin(),
                    this->Hashes[i].password.end());
            printf("%s:%s", this->Hashes[i].originalHashString.c_str(),
                    pbuf.c_str());
            if (this->AddHexOutput) {
                printf(":0x");
                for (j = 0; j < Hashes[i].password.size(); j++) {
                    printf("%02x", this->Hashes[i].password[j]);
                } 
            }
            printf("\n");
        }
    }
    this->HashFileMutex.unlock();
}

void CHHashFileVPhpass::PrintNewFoundHashes() {
    trace_printf("CHHashFileVPhpass::PrintAllFoundHashes()\n");
    
    uint64_t i;
    int j;
    std::string pbuf;
    
    this->HashFileMutex.lock();
    for (i = 0; i < this->TotalHashes; i++) {
        if (this->Hashes[i].passwordFound && !this->Hashes[i].passwordPrinted) {
            pbuf = std::string(this->Hashes[i].password.begin(),
                    this->Hashes[i].password.end());
            printf("%s:%s", this->Hashes[i].originalHashString.c_str(),
                    pbuf.c_str());
            if (this->AddHexOutput) {
                printf(":0x");
                for (j = 0; j < Hashes[i].password.size(); j++) {
                    printf("%02x", this->Hashes[i].password[j]);
                } 
            }
            printf("\n");
            this->Hashes[i].passwordPrinted = 1;
        }
    }
    this->HashFileMutex.unlock();
}

int CHHashFileVPhpass::OutputUnfoundHashesToFile(std::string filename) {
    
}

void CHHashFileVPhpass::ImportHashListFromRemoteSystem(std::string & remoteData) {
    
}

void CHHashFileVPhpass::ExportHashListToRemoteSystem(std::string * exportData) {
    
}

#define UNIT_TEST 0

#if UNIT_TEST

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    
    CHHashFileVPhpass HashFile;
    
    std::vector<std::vector<uint8_t> > HashData;
    std::vector<std::vector<uint8_t> > SaltData;
    std::vector<std::vector<uint8_t> > OtherData1;
    
    if (argc != 2) {
        printf("Call it with the file name!\n");
        exit(1);
    }
    
    
    HashFile.OpenHashFile(argv[1]);
    
    printf("Loaded %d hashes\n", (int)HashFile.GetTotalHashCount());
    
    
    
    
}

#endif
