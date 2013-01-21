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

#include "CH_HashFiles/CHHashFileSalted.h"
#include "CH_Common/CHHashImplementation.h"
#include "CH_Common/CHHashImplementationMD5.h"
//#define TRACE_PRINTF 1
#include "MFN_Common/MFNDebugging.h"

// For supa-verbose printouts.
//#define SALTEDFILE_PRINTF 1

#if SALTEDFILE_PRINTF
#define saltedfile_printf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define saltedfile_printf(fmt, ...) do {} while (0)
#endif


CHHashFileSalted::CHHashFileSalted(int newHashLengthBytes, 
        int newMaxSaltLengthBytes, char newSaltIsFirst, char newSaltIsLiteral, 
        char newSeperatorSymbol = ':') : CHHashFile() {
    trace_printf("CHHashFileSalted::CHHashFileSalted()\n");
    
    // Ensure the structures are clear.
    this->SaltedHashes.clear();
    this->UniqueSalts.clear();

    this->TotalHashes = 0;
    this->TotalHashesFound = 0;
    this->TotalHashesRemaining = 0;
    
    // Copy parameters into the internal state
    this->HashLengthBytes = newHashLengthBytes;
    this->MaxSaltLengthBytes = newMaxSaltLengthBytes;
    this->SaltIsFirst = newSaltIsFirst;
    this->SaltIsLiteral = newSaltIsLiteral;
    this->SeperatorSymbol = newSeperatorSymbol;
    
    this->HashExportCacheValid = 0;
    this->UniqueSaltsValid = 0;
    
    this->clearProtobufCache();
}


std::string CHHashFileSalted::formatHashToPrint(const HashSalted &saltedHash) {
    char hashBuffer[1024];
    memset(hashBuffer, 0, sizeof(hashBuffer));

    // Put the original hash in the beginning of the buffer with the separator.
    sprintf(hashBuffer, "%s%c", saltedHash.originalHash.c_str(),
            this->OutputSeparator);
    
    if (saltedHash.passwordFound) {
        for (size_t i = 0; i < saltedHash.password.size(); i++) {
            sprintf(hashBuffer, "%s%c", hashBuffer, saltedHash.password[i]);
        }

        if (this->AddHexOutput) {
            sprintf(hashBuffer, "%s%c0x", hashBuffer, this->OutputSeparator);
            for (size_t i = 0; i < saltedHash.password.size(); i++) {
                sprintf(hashBuffer, "%s%02x", hashBuffer, saltedHash.password[i]);
            }
        }
    }

    return std::string(hashBuffer);
}

void CHHashFileSalted::parseFileLine(std::string fileLine, size_t lineNumber) {
    trace_printf("CHHashFileSalted::parseFileLine()\n");
    std::string hashValue;
    std::string saltValue;
    HashSalted HashVectorEntry;
    size_t separatorPos;
    
    std::string whitespaces (" \t\f\v\n\r");
    size_t found;

    HashVectorEntry.passwordFound = 0;
    HashVectorEntry.passwordOutputToFile = 0;
    HashVectorEntry.passwordPrinted = 0;
    HashVectorEntry.hash.clear();
    HashVectorEntry.salt.clear();
    HashVectorEntry.originalSalt.clear();
    

    found=fileLine.find_last_not_of(whitespaces);
    if (found!=std::string::npos) {
        fileLine.erase(found+1);
    } else {
        fileLine.clear();
    }
    saltedfile_printf("Hash length: %d\n", (int)fileLine.length());

    // If the line length is 0, continue - blank line that we can ignore.
    if (fileLine.length() == 0) {
        return;
    }
    
    // Store the original hash for later printing.
    HashVectorEntry.originalHash = fileLine;

    // Determine the location of the separator symbol in the line.
    separatorPos = fileLine.find_first_of(this->SeperatorSymbol, 0);
    if (separatorPos == std::string::npos) {
        // Separator not found - abort!
        printf("Warning: Separator character '%c' not found on line %u!\n",
                this->SeperatorSymbol, (unsigned int)lineNumber);
        return;
    } else {
        saltedfile_printf("Found split at %d\n", (int)separatorPos);
    }

    if (this->SaltIsFirst) {
        // Salt is the first part of the line.

        // Check hash length - don't forget the length of the separator.
        if ((fileLine.length() - (separatorPos + 1)) != (this->HashLengthBytes * 2)) {
            printf("Warning: Hash on line %u is not correct length!\n",
                    (unsigned int)lineNumber);
            return;
        }

        // Copy the salt into the salt string.
        saltValue = fileLine.substr(0, separatorPos);
        // Copy the hash into the hash string - from the separator to the end of the line.
        hashValue = fileLine.substr(separatorPos + 1, std::string::npos);
        saltedfile_printf("Salt:Hash format\n");
        saltedfile_printf("Salt: %s\n", saltValue.c_str());
        saltedfile_printf("Hash: %s\n", hashValue.c_str());
    } else {
        // Hash is the first part of the line.

        // Check the hash length to ensure it is correct.
        if (separatorPos != (this->HashLengthBytes * 2)) {
            printf("Warning: Hash on line %lu is not correct length!\n",
                    (unsigned int)lineNumber);
            return;;
        }
        // Copy the hash into the hash string.
        hashValue = fileLine.substr(0, separatorPos);
        // Copy the salt into the salt string - from the separator to the end of the line.
        saltValue = fileLine.substr(separatorPos + 1, std::string::npos);
        saltedfile_printf("Hash:Salt format\n");
        saltedfile_printf("Hash: %s\n", hashValue.c_str());
        saltedfile_printf("Salt: %s\n", saltValue.c_str());
    }

    // Deal with the hash: It should be ASCII-hex, so convert it.
    HashVectorEntry.hash = this->convertAsciiToBinary(hashValue);

    // Deal with the salt properly.
    if (this->SaltIsLiteral) {
        // Salt is literal - copy it into the salt vector with a std::copy operation.
        HashVectorEntry.salt.reserve(saltValue.length());
        std::copy(saltValue.begin(), saltValue.end(), std::back_inserter(HashVectorEntry.salt));
    } else {
        // Salt is ascii-hex - convert it from a string to a vector.
        HashVectorEntry.salt = this->convertAsciiToBinary(saltValue);
    }
    
    saltedfile_printf("Loaded hash value: 0x");
    for (int i = 0; i < HashVectorEntry.hash.size(); i++) {
        saltedfile_printf("%02x", HashVectorEntry.hash[i]);
    }
    saltedfile_printf("\n");
    saltedfile_printf("Loaded salt value: 0x");
    for (int i = 0; i < HashVectorEntry.salt.size(); i++) {
        saltedfile_printf("%02x", HashVectorEntry.salt[i]);
    }
    saltedfile_printf("\n");
    
    // Store the original salt for output if it's been processed.
    HashVectorEntry.originalSalt = HashVectorEntry.salt;

    // If the salt is to be hashed, do it here.
    if (this->saltPrehashAlgorithm == CH_HASHFILE_MD5_ASCII) {
        HashVectorEntry.salt = this->HashImplementationMD5.
            hashDataAsciiVector(HashVectorEntry.salt);

        saltedfile_printf("Hashed salt with MD5 function\n");
        saltedfile_printf("Salt: ");
        for (int i = 0; i < HashVectorEntry.originalSalt.size(); i++) {
            saltedfile_printf("%c", HashVectorEntry.originalSalt[i]);
        }
        saltedfile_printf("\nHex : ");
        for (int i = 0; i < HashVectorEntry.originalSalt.size(); i++) {
            saltedfile_printf("%02x", HashVectorEntry.originalSalt[i]);
        }
        saltedfile_printf("\nHashed: ");
        for (int i = 0; i < HashVectorEntry.salt.size(); i++) {
            saltedfile_printf("%c", HashVectorEntry.salt[i]);
        }
        saltedfile_printf("\n");
    }
        
    this->SaltedHashes.push_back(HashVectorEntry);

    
}

void CHHashFileSalted::performPostLoadOperations() {
    this->sortHashes();
    
    // Set total hashes and hashes remaining to size of hash vector.
    this->TotalHashes = this->SaltedHashes.size();
    this->TotalHashesRemaining = this->TotalHashes;
    
    this->extractUncrackedSalts();
}

int CHHashFileSalted::outputNewFoundHashesToFile() {
    trace_printf("CHHashFileSalted::outputNewFoundHashesToFile()\n");
    
    std::vector<HashSalted>::iterator currentHash;
    
    // Ensure the output file is opened for access before trying to write to it.
    if (this->OutputFile) {
        for (currentHash = this->SaltedHashes.begin();
                currentHash < this->SaltedHashes.end(); currentHash++) {
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

void CHHashFileSalted::sortHashes() {
    trace_printf("CHHashFileSalted::SortHashes()\n");
    // Sort hashes and remove duplicates.
    std::sort(this->SaltedHashes.begin(), this->SaltedHashes.end(),
            CHHashFileSalted::saltedHashSortPredicate);
    this->SaltedHashes.erase(
        std::unique(this->SaltedHashes.begin(), this->SaltedHashes.end(),
            CHHashFileSalted::saltedHashUniquePredicate),
            this->SaltedHashes.end());
}

void CHHashFileSalted::extractUncrackedSalts() {
    trace_printf("CHHashFileSalted::ExtractUncrackedSalts()\n");
    // This function must be protected by a mutex outside!
    SaltData saltElement;
    // Determine if other data is used.
    uint8_t usesOtherData[5];
    uint8_t usesIterations = 0;;
    
    memset(usesOtherData, 0, sizeof(usesOtherData));
    
    // Clear out the old salts.
    this->UniqueSalts.clear();
    
    // Loop through the hashes, copying unfound salts into the new structure.
    std::vector<HashSalted>::iterator HashesIt;
    for (HashesIt = this->SaltedHashes.begin();
            HashesIt < this->SaltedHashes.end(); 
            HashesIt++) {
        
        if (!HashesIt->passwordFound) {
            saltElement.iterationCount = HashesIt->iterationCount;
            if (saltElement.iterationCount) {
                usesIterations = 1;
            }
            
            saltElement.salt = HashesIt->salt;
            
            saltElement.otherData1 = HashesIt->otherData1;
            if (saltElement.otherData1.size()) {
                usesOtherData[0] = 1;
            }
            saltElement.otherData2 = HashesIt->otherData2;
            if (saltElement.otherData2.size()) {
                usesOtherData[1] = 1;
            }
            saltElement.otherData3 = HashesIt->otherData3;
            if (saltElement.otherData3.size()) {
                usesOtherData[2] = 1;
            }
            saltElement.otherData4 = HashesIt->otherData4;
            if (saltElement.otherData4.size()) {
                usesOtherData[3] = 1;
            }
            saltElement.otherData5 = HashesIt->otherData5;
            if (saltElement.otherData5.size()) {
                usesOtherData[4] = 1;
            }
            
            this->UniqueSalts.push_back(saltElement);
        }
    }
    
    // Sort and unique the salts.
    std::sort(this->UniqueSalts.begin(), this->UniqueSalts.end(),
            CHHashFileSalted::saltSortPredicate);
    this->UniqueSalts.erase(
        std::unique(this->UniqueSalts.begin(), this->UniqueSalts.end(),
            CHHashFileSalted::saltUniquePredicate),
        this->UniqueSalts.end());
    
    // Unique salts are sorted out.  Build the cache.
    this->UniqueSaltDataExportCache.SaltData.clear();
    this->UniqueSaltDataExportCache.iterationCount.clear();
    this->UniqueSaltDataExportCache.OtherData1.clear();
    this->UniqueSaltDataExportCache.OtherData2.clear();
    this->UniqueSaltDataExportCache.OtherData3.clear();
    this->UniqueSaltDataExportCache.OtherData4.clear();
    this->UniqueSaltDataExportCache.OtherData5.clear();

    this->UniqueSaltDataExportCache.SaltData.reserve(this->UniqueSalts.size());
    
    // If features are used, reserve data for them.
    if (usesIterations) {
        this->UniqueSaltDataExportCache.iterationCount.reserve(
            this->UniqueSalts.size());
    }
    if (usesOtherData[0]) {
        this->UniqueSaltDataExportCache.OtherData1.reserve(
            this->UniqueSalts.size());
    }
    if (usesOtherData[1]) {
        this->UniqueSaltDataExportCache.OtherData2.reserve(
            this->UniqueSalts.size());
    }
    if (usesOtherData[2]) {
        this->UniqueSaltDataExportCache.OtherData3.reserve(
            this->UniqueSalts.size());
    }
    if (usesOtherData[3]) {
        this->UniqueSaltDataExportCache.OtherData4.reserve(
            this->UniqueSalts.size());
    }
    if (usesOtherData[4]) {
        this->UniqueSaltDataExportCache.OtherData5.reserve(
            this->UniqueSalts.size());
    }
    
    std::vector<SaltData>::iterator SaltsIt;
    for (SaltsIt = this->UniqueSalts.begin();
            SaltsIt < this->UniqueSalts.end(); 
            SaltsIt++) {

        this->UniqueSaltDataExportCache.SaltData.push_back(SaltsIt->salt);
        if (usesIterations) {
            this->UniqueSaltDataExportCache.iterationCount.push_back(
                SaltsIt->iterationCount);
        }
        if (usesOtherData[0]) {
            this->UniqueSaltDataExportCache.OtherData1.push_back(
                SaltsIt->otherData1);
        }
        if (usesOtherData[1]) {
            this->UniqueSaltDataExportCache.OtherData2.push_back(
                SaltsIt->otherData2);
        }
        if (usesOtherData[2]) {
            this->UniqueSaltDataExportCache.OtherData3.push_back(
                SaltsIt->otherData3);
        }
        if (usesOtherData[3]) {
            this->UniqueSaltDataExportCache.OtherData4.push_back(
                SaltsIt->otherData4);
        }
        if (usesOtherData[4]) {
            this->UniqueSaltDataExportCache.OtherData5.push_back(
                SaltsIt->otherData5);
        }
    }    
    
    this->UniqueSaltDataExportCacheValid = 1;
    this->UniqueSaltsValid = 1;
}

CHHashFileSaltedDataBlob CHHashFileSalted::exportUniqueSaltedData() {
    trace_printf("CHHashFileSalted::ExportUniqueSaltedData()\n");

    CHHashFileSaltedDataBlob returnBlob;

    this->HashFileMutex.lock();
    // If the salt cache is not valid, update it.
    if (!this->UniqueSaltDataExportCacheValid) {
        // Update the list of uncracked salts.
        this->extractUncrackedSalts();
    }

    // Return a copy of the internal buffer.
    returnBlob = this->UniqueSaltDataExportCache;
    
    this->HashFileMutex.unlock();

    return returnBlob;
}

// Sort by the hashes ONLY.  No point in sorting other stuff right now.
bool CHHashFileSalted::saltedHashSortPredicate(const HashSalted &d1, const HashSalted &d2) {
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
    return 0;
}

// Compare everything - we don't want to delete hashes that aren't identical
// in every way.
bool CHHashFileSalted::saltedHashUniquePredicate(const HashSalted &d1, const HashSalted &d2) {
    // Compare all elements.  Not-equal means a 0 (false) return.
    // I'll break from my usual style for this case.
    if (d1.hash != d2.hash) return 0;
    if (d1.salt != d2.salt) return 0;
    if (d1.iterationCount != d2.iterationCount) return 0;
    if (d1.otherData1 != d2.otherData1) return 0;
    if (d1.otherData2 != d2.otherData2) return 0;
    if (d1.otherData3 != d2.otherData3) return 0;
    if (d1.otherData4 != d2.otherData4) return 0;
    if (d1.otherData5 != d2.otherData5) return 0;
    if (d1.userData != d2.userData) return 0;
    
    // All checks passed - hashes are equal.
    return 1;
}

// Sort by salt length, then iteration count, then salt value, then other stuff.
bool CHHashFileSalted::saltSortPredicate(const SaltData& d1,
        const SaltData& d2) {

    // d1 smaller salt than d2, return d1 < 2
    if (d1.salt.size() < d2.salt.size()) {
        return 1;
    }
    // d1 smaller iteration count than d2, return d1 < d2
    if (d1.iterationCount < d2.iterationCount) {
        return 1;
    }
    
    // Check salt
    if (memcmp(&d1.salt[0], &d2.salt[0], d1.salt.size()) < 0) {
        return 1;
    }
    
    // Check other data.  If lengths match, memcmp them.
    if (d1.otherData1.size() < d2.otherData1.size()) {
        return 1;
    }
    if (memcmp(&d1.otherData1[0], &d2.otherData1[0], d1.otherData1.size()) < 0) {
        return 1;
    }

    if (d1.otherData2.size() < d2.otherData2.size()) {
        return 1;
    }
    if (memcmp(&d1.otherData2[0], &d2.otherData2[0], d1.otherData2.size()) < 0) {
        return 1;
    }
    
    if (d1.otherData3.size() < d2.otherData3.size()) {
        return 1;
    }
    if (memcmp(&d1.otherData3[0], &d2.otherData3[0], d1.otherData3.size()) < 0) {
        return 1;
    }

    if (d1.otherData4.size() < d2.otherData4.size()) {
        return 1;
    }
    if (memcmp(&d1.otherData4[0], &d2.otherData4[0], d1.otherData4.size()) < 0) {
        return 1;
    }

    if (d1.otherData5.size() < d2.otherData5.size()) {
        return 1;
    }
    if (memcmp(&d1.otherData5[0], &d2.otherData5[0], d1.otherData5.size()) < 0) {
        return 1;
    }

    return 0;
}

// Compare everything - we don't want to delete hashes that aren't identical
// in every way.
bool CHHashFileSalted::saltUniquePredicate(const SaltData& d1,
        const SaltData& d2) {
    // Compare all elements.  Not-equal means a 0 (false) return.
    // I'll break from my usual style for this case.
    if (d1.salt != d2.salt) return 0;
    if (d1.iterationCount != d2.iterationCount) return 0;
    if (d1.otherData1 != d2.otherData1) return 0;
    if (d1.otherData2 != d2.otherData2) return 0;
    if (d1.otherData3 != d2.otherData3) return 0;
    if (d1.otherData4 != d2.otherData4) return 0;
    if (d1.otherData5 != d2.otherData5) return 0;
    
    // All checks passed - hashes are equal.
    return 1;
}


std::vector<std::vector<uint8_t> > CHHashFileSalted::exportUncrackedHashList() {
    trace_printf("CHHashFileSalted::exportUncrackedHashList()\n");
    std::vector<HashSalted>::iterator currentHash;
    
    this->HashFileMutex.lock();
    
    // Check to see if the cache is valid.  If so, we can just return that.
    // Otherwise, need to generate it.
    
    if (!this->HashExportCacheValid) {
        // Clear the cache and regenerate it.
        this->HashExportCache.clear();
        // Loop through all current hashes.
        for (currentHash = this->SaltedHashes.begin();
                currentHash < this->SaltedHashes.end(); currentHash++) {
            // If it's already found, continue.
            if (currentHash->passwordFound) {
                continue;
            }
            // If not, add it to the current return vector.
            this->HashExportCache.push_back(currentHash->hash);
        }
    }

    this->HashExportCacheValid = 1;
    
    this->HashFileMutex.unlock();
    
    // No need to sort/unique - this has already been done with the main hash
    // list.  Just return the now-cached data.
    
    return this->HashExportCache;
}

int CHHashFileSalted::reportFoundPassword(std::vector<uint8_t> hash, std::vector<uint8_t> password) {
    trace_printf("CHHashFileSalted::ReportFoundPassword()\n");
    
    uint64_t i;
    int passwordsFound = 0;

    this->HashFileMutex.lock();

    for (i = 0; i < this->TotalHashes; i++) {
        if (memcmp(&hash[0], &this->SaltedHashes[i].hash[0], this->HashLengthBytes) == 0) {
            // Only do this if the password is not already reported.
            if (!this->SaltedHashes[i].passwordFound) {
                this->SaltedHashes[i].password = password;
                this->SaltedHashes[i].passwordFound = 1;
                this->TotalHashesFound++;
                this->TotalHashesRemaining--;
                // Output to a file if needed.
                passwordsFound++;
            }
        }
    }
    
    // Dump passwords to a file if they're found, and clear the cache.
    if (passwordsFound) {
        this->outputNewFoundHashesToFile();
        this->clearProtobufCache();
        this->HashExportCacheValid = 0;
        this->UniqueSaltDataExportCacheValid = 0;
        this->UniqueSaltsValid = 0;
        this->HashExportCache.clear();
    }
    this->HashFileMutex.unlock();
    return passwordsFound;
}

void CHHashFileSalted::printAllFoundHashes() {
    trace_printf("CHHashFileSalted::PrintAllFoundHashes()\n");
    
    uint64_t i;
    int j;
    // So we can build them ahead of time and paste them together.
    // Two character arrays are here in case we need to make strings
    // in interesting formats.
    
    
    // These are here to turn vector<uint8_t>s into something we can
    // print to screen without so much manipulation. God bless <<
    // operators.
    
    std::string sbuf, hbuf, pbuf;
    this->HashFileMutex.lock();
    for (i = 0; i < this->TotalHashes; i++) {
        if (this->SaltedHashes[i].passwordFound) {
            printf("%s\n", this->formatHashToPrint(
                    this->SaltedHashes[i]).c_str());
        }
    }
    this->HashFileMutex.unlock();
}

void CHHashFileSalted::printNewFoundHashes() {
    trace_printf("CHHashFileSalted::PrintNewFoundHashes()\n");
    int j;
    // TODO: C++-ify this.
    this->HashFileMutex.lock();
    for (uint64_t i=0; i< this->TotalHashes; i++)
    {
        if ((this->SaltedHashes[i].passwordFound) &&
                (!this->SaltedHashes[i].passwordPrinted)) {
            printf("%s\n", this->formatHashToPrint(
                    this->SaltedHashes[i]).c_str());
            this->SaltedHashes[i].passwordPrinted = 1;
        
        }   
    }
    this->HashFileMutex.unlock();
}

int CHHashFileSalted::outputUnfoundHashesToFile(std::string filename) {
    trace_printf("CHHashFileSalted::OutputUnfoundHashesToFile()\n");
    std::vector<HashSalted>::iterator currentHash;
    char hashBuffer[1024];
    char saltBuffer[1024];
    FILE *UnfoundHashes;
    
    this->HashFileMutex.lock();
    // If we can't open it... oh well?
    UnfoundHashes = fopen(filename.c_str(), "w");

    // Ensure the output file is opened for access before trying to write to it.
    if (UnfoundHashes) {
        for (currentHash = this->SaltedHashes.begin();
                currentHash < this->SaltedHashes.end(); currentHash++) {
            // Skip if already reported.
            if (!currentHash->passwordFound) {
                fprintf(UnfoundHashes, "%s\n",
                        currentHash->originalHash.c_str()); 
            }
        }
    }
    fflush(UnfoundHashes);
    fclose(UnfoundHashes);
    
    this->HashFileMutex.unlock();
    return 1;
    
}

void CHHashFileSalted::importHashListFromRemoteSystem(std::string & remoteData)  {
    trace_printf("CHHashFileSalted::ImportHashListFromRemoteSystem()\n");
    std::string hashBuffer;
    CHHashFileSalted::HashSalted hashSalted;

    this->HashFileMutex.lock();
    this->SaltedHashProtobuf.Clear();
    this->SaltedHashProtobuf.ParseFromString(remoteData);
    this->SaltedHashes.clear();
    

    //Unpack the CHHashFileSalted general member variables other than Hashes.
    this->HashLengthBytes = this->SaltedHashProtobuf.hash_length_bytes();
    this->MaxSaltLengthBytes = this->SaltedHashProtobuf.salt_length_bytes();
    
    // Clear common things.
    hashSalted.password = std::vector<uint8_t>();
    hashSalted.passwordFound = 0;
    hashSalted.passwordOutputToFile = 0;
    hashSalted.passwordPrinted = 0;

    //Then unpack individual HashSalted structs.
    this->TotalHashes = this->SaltedHashProtobuf.salted_hash_value_size();
    for(int i=0;i<this->TotalHashes;i++)
    {
        this->SaltedHashInnerProtobuf.Clear();
        hashSalted.hash.clear();
        hashSalted.salt.clear();
        hashSalted.iterationCount = 0;
        hashSalted.originalHash.clear();
        hashSalted.originalSalt.clear();
        hashSalted.otherData1.clear();
        hashSalted.otherData2.clear();
        hashSalted.otherData3.clear();
        hashSalted.otherData4.clear();
        hashSalted.otherData5.clear();
        
        this->SaltedHashInnerProtobuf = this->SaltedHashProtobuf.salted_hash_value(i);
        
        // Only copy fields needed.
        if (this->SaltedHashInnerProtobuf.has_hash()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.hash());
            hashSalted.hash = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }
        if (this->SaltedHashInnerProtobuf.has_salt()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.salt());
            hashSalted.saltLength = hashBuffer.length();
            hashSalted.salt = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }
        if (this->SaltedHashInnerProtobuf.has_iteration_count()) {
            hashSalted.iterationCount = this->SaltedHashInnerProtobuf.iteration_count();
        }
        if (this->SaltedHashInnerProtobuf.has_other_data_1()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.other_data_1());
            hashSalted.otherData1 = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }
        if (this->SaltedHashInnerProtobuf.has_other_data_2()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.other_data_2());
            hashSalted.otherData2 = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }
        if (this->SaltedHashInnerProtobuf.has_other_data_3()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.other_data_3());
            hashSalted.otherData3 = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }
        if (this->SaltedHashInnerProtobuf.has_other_data_4()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.other_data_4());
            hashSalted.otherData4 = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }
        if (this->SaltedHashInnerProtobuf.has_other_data_5()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.other_data_5());
            hashSalted.otherData5 = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }

        // Hash the salt if needed.
        if (this->saltPrehashAlgorithm == CH_HASHFILE_MD5_ASCII) {
            hashSalted.originalSalt = hashSalted.salt;
            hashSalted.salt = this->HashImplementationMD5.
                hashDataAsciiVector(hashSalted.salt);
        }

        this->SaltedHashes.push_back(hashSalted);
    }
    
    // All data is in SaltedHashes.  Sort and unique
    this->sortHashes();
    
    this->SaltedHashProtobuf.Clear();
    this->SaltedHashInnerProtobuf.Clear();
    
    this->clearProtobufCache();
    this->HashExportCacheValid = 0;
    this->UniqueSaltDataExportCacheValid = 0;
    this->UniqueSaltsValid = 0;

    this->TotalHashesRemaining = this->TotalHashes;

    this->HashFileMutex.unlock();
}

void CHHashFileSalted::createHashListExportProtobuf() {
    trace_printf("CHHashFileSalted::createHashListExportProtobuf()\n");

    std::string hashBuffer;
    std::vector<CHHashFileSalted::HashSalted>::iterator Hash;
    ::MFNHashFileSaltedProtobuf_SaltedHash *uniqueHashProtobuf;

    this->hashExportProtobufCache.clear();
    
    this->SaltedHashProtobuf.Clear();
    this->SaltedHashInnerProtobuf.Clear();
    
    // Set global values
    this->SaltedHashProtobuf.set_hash_length_bytes(this->HashLengthBytes);
    this->SaltedHashProtobuf.set_salt_length_bytes(this->MaxSaltLengthBytes);

    // Copy all not-found hashes into the protobuf.
    for (Hash = this->SaltedHashes.begin();
            Hash < this->SaltedHashes.end(); Hash++) {
        // Skip found passwords - no need to waste network bandwidth.
        if (Hash->passwordFound) {
            continue;
        }
        
        uniqueHashProtobuf = this->SaltedHashProtobuf.add_salted_hash_value();
        
        // Copy the hash into the inner protobuf.
        hashBuffer = std::string(Hash->hash.begin(), Hash->hash.end());
        uniqueHashProtobuf->set_hash(hashBuffer);
        
        // Only insert the other elements if they are present.
        if (Hash->salt.size()) {
            hashBuffer = std::string(Hash->salt.begin(), Hash->salt.end());
            uniqueHashProtobuf->set_salt(hashBuffer);
        }
        
        if (Hash->iterationCount) {
            uniqueHashProtobuf->set_iteration_count(Hash->iterationCount);
        }
        
        if (Hash->otherData1.size()) {
            hashBuffer = std::string(Hash->otherData1.begin(),
                    Hash->otherData1.end());
            uniqueHashProtobuf->set_other_data_1(hashBuffer);
        }
        if (Hash->otherData2.size()) {
            hashBuffer = std::string(Hash->otherData1.begin(),
                    Hash->otherData2.end());
            uniqueHashProtobuf->set_other_data_1(hashBuffer);
        }
        if (Hash->otherData3.size()) {
            hashBuffer = std::string(Hash->otherData1.begin(),
                    Hash->otherData3.end());
            uniqueHashProtobuf->set_other_data_1(hashBuffer);
        }
        if (Hash->otherData4.size()) {
            hashBuffer = std::string(Hash->otherData1.begin(),
                    Hash->otherData4.end());
            uniqueHashProtobuf->set_other_data_1(hashBuffer);
        }
        if (Hash->otherData5.size()) {
            hashBuffer = std::string(Hash->otherData1.begin(),
                    Hash->otherData5.end());
            uniqueHashProtobuf->set_other_data_1(hashBuffer);
        }
    }
    
    this->SaltedHashProtobuf.SerializeToString(&this->hashExportProtobufCache);
    
    this->protobufExportCachesValid = 1;
}


void CHHashFileSalted::importUniqueSaltsFromRemoteSystem(std::string & remoteData) {
    trace_printf("CHHashFileSalted::importUniqueSaltsFromRemoteSystem()\n");
    std::string hashBuffer;
    std::vector<uint8_t> salt;

    this->HashFileMutex.lock();
    // Clear the protobuf and load it with data.
    this->SaltedHashProtobuf.Clear();
    this->SaltedHashProtobuf.ParseFromString(remoteData);

    this->UniqueSalts.clear();
    
    for(int i = 0; i < this->SaltedHashProtobuf.salted_hash_value_size(); i++) {
        SaltData newSaltData;
        newSaltData.iterationCount = 0;
        
        this->SaltedHashInnerProtobuf.Clear();
        this->SaltedHashInnerProtobuf = this->SaltedHashProtobuf.salted_hash_value(i);
        
        if (this->SaltedHashInnerProtobuf.has_salt()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.salt());
            salt = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());

            if (this->saltPrehashAlgorithm == CH_HASHFILE_MD5_ASCII) {
                salt = this->HashImplementationMD5.
                    hashDataAsciiVector(salt);
            }
            newSaltData.salt = salt;
        }
        
        if (this->SaltedHashInnerProtobuf.has_iteration_count()) {
            newSaltData.iterationCount = this->SaltedHashInnerProtobuf.iteration_count();
        }
        
        if (this->SaltedHashInnerProtobuf.has_other_data_1()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.other_data_1());
            newSaltData.otherData1 = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }
        if (this->SaltedHashInnerProtobuf.has_other_data_2()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.other_data_2());
            newSaltData.otherData2 = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }
        if (this->SaltedHashInnerProtobuf.has_other_data_3()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.other_data_3());
            newSaltData.otherData3 = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }
        if (this->SaltedHashInnerProtobuf.has_other_data_4()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.other_data_4());
            newSaltData.otherData4 = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }
        if (this->SaltedHashInnerProtobuf.has_other_data_5()) {
            hashBuffer = std::string(this->SaltedHashInnerProtobuf.other_data_5());
            newSaltData.otherData5 = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        }
        
        this->UniqueSalts.push_back(newSaltData);
    }
    this->SaltedHashProtobuf.Clear();

    // Sort and unique the salts.
    std::sort(this->UniqueSalts.begin(), this->UniqueSalts.end(),
            CHHashFileSalted::saltSortPredicate);
    this->UniqueSalts.erase(
        std::unique(this->UniqueSalts.begin(), this->UniqueSalts.end(),
            CHHashFileSalted::saltUniquePredicate),
        this->UniqueSalts.end());

    /**
     * Now that there is a sorted list of salts, go through and mark any hash
     * structure without a salt as found, so it is not exported to future
     * requests for the hashes.
     */
    std::vector<CHHashFileSalted::HashSalted>::iterator i;
    for (i = this->SaltedHashes.begin(); i < this->SaltedHashes.end(); i++) {
        // If the password has NOT been found, look for it.
        if (!i->passwordFound) {
            // Binary search the sorted list.  If not found, set the password to found.
            SaltData newSaltData;
            newSaltData.salt = i->salt;
            newSaltData.iterationCount = i->iterationCount;
            newSaltData.otherData1 = i->otherData1;
            newSaltData.otherData2 = i->otherData2;
            newSaltData.otherData3 = i->otherData3;
            newSaltData.otherData4 = i->otherData4;
            newSaltData.otherData5 = i->otherData5;
            
            if (!std::binary_search(this->UniqueSalts.begin(),
                    this->UniqueSalts.end(), newSaltData,
                    CHHashFileSalted::saltSortPredicate)) {
                i->passwordFound = 1;
            }
        }
    }
    
    this->UniqueSaltsValid = 1;
    this->extractUncrackedSalts();
}

void CHHashFileSalted::createUniqueSaltsExportProtobuf() {
    trace_printf("CHHashFileSalted::createUniqueSaltsExportProtobuf()\n");
    this->SaltedHashProtobuf.Clear();
    
    MFNHashFileSaltedProtobuf_SaltedHash * uniqueHashProtobuf;
    
    std::string hashBuffer;
    
    // Here we have to pack a nested message: a bunch of SaltedHash, each of
    // which contains just the salt.
    std::vector<CHHashFileSalted::HashSalted>::iterator Hash;
    
    // Now pack individual HashSalted structs with the ORIGINAL hashes.
    // They will be rehashed on the other end.
    for(Hash = this->SaltedHashes.begin(); Hash < this->SaltedHashes.end(); Hash++) {
        if (!Hash->passwordFound) {
            uniqueHashProtobuf = this->SaltedHashProtobuf.add_salted_hash_value();
            
            if (Hash->salt.size()) {
                hashBuffer = std::string(Hash->salt.begin(), Hash->salt.end());
                uniqueHashProtobuf->set_salt(hashBuffer);
            }

            if (Hash->iterationCount) {
                uniqueHashProtobuf->set_iteration_count(Hash->iterationCount);
            }

            if (Hash->otherData1.size()) {
                hashBuffer = std::string(Hash->otherData1.begin(),
                        Hash->otherData1.end());
                uniqueHashProtobuf->set_other_data_1(hashBuffer);
            }
            if (Hash->otherData2.size()) {
                hashBuffer = std::string(Hash->otherData1.begin(),
                        Hash->otherData2.end());
                uniqueHashProtobuf->set_other_data_1(hashBuffer);
            }
            if (Hash->otherData3.size()) {
                hashBuffer = std::string(Hash->otherData1.begin(),
                        Hash->otherData3.end());
                uniqueHashProtobuf->set_other_data_1(hashBuffer);
            }
            if (Hash->otherData4.size()) {
                hashBuffer = std::string(Hash->otherData1.begin(),
                        Hash->otherData4.end());
                uniqueHashProtobuf->set_other_data_1(hashBuffer);
            }
            if (Hash->otherData5.size()) {
                hashBuffer = std::string(Hash->otherData1.begin(),
                        Hash->otherData5.end());
                uniqueHashProtobuf->set_other_data_1(hashBuffer);
            }
        }
    }
    
    this->SaltedHashProtobuf.SerializeToString(&this->saltExportProtobufCache);
}


//#define UNIT_TEST 1

#if UNIT_TEST

#include <stdlib.h>
#include <stdio.h>

char foundPasswordString[] = "Password";

int main(int argc, char* argv[]) {
    
    //CHHashFileSalted HashFile(16, 0, CHHASHFILESALTED_SALT_IS_FIRST, CHHASHFILESALTED_HEX_SALT);
    CHHashFileSalted HashFile(16, 0, CHHASHFILESALTED_HASH_IS_FIRST, CHHASHFILESALTED_LITERAL_SALT);
    CHHashFileSalted HashFile2(16, 0, CHHASHFILESALTED_HASH_IS_FIRST, CHHASHFILESALTED_LITERAL_SALT);

    std::vector<std::vector<uint8_t> > Hashes;
    std::vector<uint8_t> FoundPassword;
    CHHashFileSaltedDataBlob SaltExport;
    int i;

    if (argc != 2) {
        printf("Call it with the file name!\n");
        exit(1);
    }

    for (i = 0; i < strlen(foundPasswordString); i++) {
        FoundPassword.push_back(foundPasswordString[i]);
    }

    
    HashFile.openHashFile(argv[1]);
    
    printf("Loaded %d hashes\n", (int)HashFile.getTotalHashCount());
    
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
            printf("%c", SaltExport.SaltData[i][j]);
        }
        printf("\n");
    }
    
    
    // Report every other hash as found.
    for (i = 0; i < Hashes.size(); i += 2) {
        HashFile.reportFoundPassword(Hashes[i], FoundPassword);
    }
    
    HashFile.setAddHexOutput(true);
    //HashFile.setUseJohnOutputStyle(true);
    //HashFile.SetOutputSeparator('-');
    
    HashFile.printAllFoundHashes();
    
    
    
    std::string hashExport;
    HashFile.exportHashListToRemoteSystem(hashExport);
    
    HashFile2.importHashListFromRemoteSystem(hashExport);

    Hashes = HashFile2.exportUncrackedHashList();
    
    printf("Exported hashes 2: \n");
    for (i = 0; i < Hashes.size(); i++) {
        for (int j = 0; j < Hashes[i].size(); j++) {
            printf("%02x", Hashes[i][j]);
        }
        printf("\n");
    }

    SaltExport = HashFile2.exportUniqueSaltedData();
    printf("Exported salts2:\n");
    for (i = 0; i < SaltExport.SaltData.size(); i++) {
        for (int j = 0; j < SaltExport.SaltData[i].size(); j++) {
            printf("%c", SaltExport.SaltData[i][j]);
        }
        printf("\n");
    }
    
    // Report every other hash as found.
    for (i = 0; i < Hashes.size(); i += 2) {
        HashFile2.reportFoundPassword(Hashes[i], FoundPassword);
    }
    
    HashFile2.printAllFoundHashes();
    
    while (HashFile.getUncrackedHashCount()) {
        Hashes = HashFile.exportUncrackedHashList();
        HashFile.reportFoundPassword(Hashes[0], FoundPassword);
        HashFile.exportHashListToRemoteSystem(hashExport);
        HashFile2.importHashListFromRemoteSystem(hashExport);
        SaltExport = HashFile2.exportUniqueSaltedData();
        printf("Hashfile2: %d salts\n", SaltExport.SaltData.size());
        if (SaltExport.SaltData.size() < 10) {
            printf("Exported salts2:\n");
            for (i = 0; i < SaltExport.SaltData.size(); i++) {
                for (int j = 0; j < SaltExport.SaltData[i].size(); j++) {
                    printf("%c", SaltExport.SaltData[i][j]);
                }
                printf("\n");
            }
        }
        
    }
}

#endif
