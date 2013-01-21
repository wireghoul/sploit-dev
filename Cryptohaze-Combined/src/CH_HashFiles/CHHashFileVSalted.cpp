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

#include "CH_HashFiles/CHHashFileVSalted.h"
#include "CH_Common/CHHashImplementation.h"
#include "CH_Common/CHHashImplementationMD5.h"
//#define TRACE_PRINTF 1
#include "MFN_Common/MFNDebugging.h"

// For supa-verbose printouts.
//#define CHHASHFILEVSALTED_DEBUG 1

CHHashFileVSalted::CHHashFileVSalted(int newHashLengthBytes, 
        int newMaxSaltLengthBytes, char newSaltIsFirst, char newSaltIsLiteral, 
        char newSeperatorSymbol = ':') : CHHashFileV() {
    trace_printf("CHHashFileVSalted::CHHashFileVSalted()\n");
    
    // Ensure the structures are clear.
    this->Hashes.clear();
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
    
    this->clearCaches();
}


int CHHashFileVSalted::OpenHashFile(std::string filename) {
    trace_printf("CHHashFileVSalted::OpenHashFile()\n");
    std::ifstream hashFile;
    std::string fileLine;
    std::string hashValue;
    std::string saltValue;
    HashSalted HashVectorEntry;
    size_t separatorPos;
    uint64_t fileLineCount = 0;
    CHHashImplementation *HashFunction = NULL;
    
    std::string whitespaces (" \t\f\v\n\r");
    size_t found;

    this->HashFileMutex.lock();
    this->clearCaches();

    HashVectorEntry.passwordFound = 0;
    HashVectorEntry.passwordOutputToFile = 0;
    HashVectorEntry.passwordPrinted = 0;
    
    // If the salt is to be hashed, set up the class to do it.
    if (this->saltPrehashAlgorithm == CH_HASHFILE_MD5_ASCII) {
        HashFunction = new CHHashImplementationMD5();
    }
    
    hashFile.open(filename.c_str(), std::ios_base::in);
    if (!hashFile.good())
    {
        
        std::cout << "ERROR: Cannot open hashfile " << filename <<"\n";
        exit(1);
    }
    
    while (std::getline(hashFile, fileLine)) {
        HashVectorEntry.hash.clear();
        HashVectorEntry.salt.clear();
        HashVectorEntry.originalSalt.clear();
        
        found=fileLine.find_last_not_of(whitespaces);
        if (found!=std::string::npos)
            fileLine.erase(found+1);
        else
            fileLine.clear();  
#if CHHASHFILEVSALTED_DEBUG
        printf("Hash length: %d\n", (int)fileLine.length());
#endif
        
        // If the line length is 0, continue - blank line that we can ignore.
        if (fileLine.length() == 0) {
            continue;
        }

        // Determine the location of the separator symbol in the line.
        separatorPos = fileLine.find_first_of(this->SeperatorSymbol, 0);
        if (separatorPos == std::string::npos) {
            // Separator not found - abort!
            printf("Warning: Separator character '%c' not found on line %lu!\n", this->SeperatorSymbol, fileLineCount);
            continue;
        } else {
#if CHHASHFILEVSALTED_DEBUG
            printf("Found split at %d\n", (int)separatorPos);
#endif
        }

        if (this->SaltIsFirst) {
            // Salt is the first part of the line.
            
            // Check hash length - don't forget the length of the separator.
            if ((fileLine.length() - (separatorPos + 1)) != (this->HashLengthBytes * 2)) {
                printf("Warning: Hash on line %lu is not correct length!\n", fileLineCount);
                continue;
            }
            
            // Copy the salt into the salt string.
            saltValue = fileLine.substr(0, separatorPos);
            // Copy the hash into the hash string - from the separator to the end of the line.
            hashValue = fileLine.substr(separatorPos + 1, std::string::npos);
#if CHHASHFILEVSALTED_DEBUG
            printf("Salt:Hash format\n");
            printf("Salt: %s\n", saltValue.c_str());
            printf("Hash: %s\n", hashValue.c_str());
#endif
        } else {
            // Hash is the first part of the line.
            
            // Check the hash length to ensure it is correct.
            if (separatorPos != (this->HashLengthBytes * 2)) {
                printf("Warning: Hash on line %lu is not correct length!\n", fileLineCount);
                continue;
            }
            // Copy the hash into the hash string.
            hashValue = fileLine.substr(0, separatorPos);
            // Copy the salt into the salt string - from the separator to the end of the line.
            saltValue = fileLine.substr(separatorPos + 1, std::string::npos);
#if CHHASHFILEVSALTED_DEBUG
            printf("Hash:Salt format\n");
            printf("Hash: %s\n", hashValue.c_str());
            printf("Salt: %s\n", saltValue.c_str());
#endif
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
#if CHHASHFILEVSALTED_DEBUG
        printf("Loaded hash value: 0x");
        for (int i = 0; i < HashVectorEntry.hash.size(); i++) {
            printf("%02x", HashVectorEntry.hash[i]);
        }
        printf("\n");
        printf("Loaded salt value: 0x");
        for (int i = 0; i < HashVectorEntry.salt.size(); i++) {
            printf("%02x", HashVectorEntry.salt[i]);
        }
        printf("\n");
        
#endif
        // Store the original salt for output if it's been processed.
        HashVectorEntry.originalSalt = HashVectorEntry.salt;
        
        // If the salt is to be hashed, do it here.
        if (this->saltPrehashAlgorithm == CH_HASHFILE_MD5_ASCII) {
            HashVectorEntry.salt = HashFunction->
                hashDataAsciiVector(HashVectorEntry.salt);
#if CHHASHFILEVSALTED_DEBUG
            printf("Hashed salt with MD5 function\n");
            printf("Salt: ");
            for (int i = 0; i < HashVectorEntry.originalSalt.size(); i++) {
                printf("%c", HashVectorEntry.originalSalt[i]);
            }
            printf("\nHex : ");
            for (int i = 0; i < HashVectorEntry.originalSalt.size(); i++) {
                printf("%02x", HashVectorEntry.originalSalt[i]);
            }
            printf("\nHashed: ");
            for (int i = 0; i < HashVectorEntry.salt.size(); i++) {
                printf("%c", HashVectorEntry.salt[i]);
            }
            printf("\n");
#endif
        }
        
        this->Hashes.push_back(HashVectorEntry);

        fileLineCount++;
    }
    
    this->SortHashes();
    
    // Set total hashes and hashes remaining to size of hash vector.
    this->TotalHashes = this->Hashes.size();
    this->TotalHashesRemaining = this->TotalHashes;
    
    hashFile.close();
    
    this->ExtractUncrackedSalts();
    
    // Clean up the hash operator if it's in use.
    if (HashFunction) {
        delete HashFunction;
    }
    
    // If NO hashes are loaded, something is probably very wrong.
    if (this->TotalHashes == 0) {
        printf("ERROR: No hashes loaded!\n");
        exit(1);
    }
    
    this->HashFileMutex.unlock();
    
    return 1;
}


int CHHashFileVSalted::OutputFoundHashesToFile() {
    trace_printf("CHHashFileVSalted::OutputFoundHashesToFile()\n");
    
    std::vector<HashSalted>::iterator currentHash;
    char hashBuffer[1024];
    char saltBuffer[1024];
    
    // Ensure the output file is opened for access before trying to write to it.
    if (this->OutputFile) {
        for (currentHash = this->Hashes.begin();
                currentHash < this->Hashes.end(); currentHash++) {
            // Skip if already reported.
            if (currentHash->passwordFound && !currentHash->passwordOutputToFile) {
                memset(hashBuffer, 0, sizeof(hashBuffer));
                memset(saltBuffer, 0, sizeof(saltBuffer));
                // Write the hash
                for (size_t i = 0; i < currentHash->hash.size(); i++) {
                    sprintf(hashBuffer, "%s%02x", hashBuffer, currentHash->hash[i]);
                }
                // Write the salt
                for (size_t i = 0; i < currentHash->originalSalt.size(); i++) {
                    if (this->SaltIsLiteral) {
                        // Print the character
                        sprintf(saltBuffer, "%s%c", saltBuffer, currentHash->originalSalt[i]);
                    } else {
                        // Print the hex
                        sprintf(saltBuffer, "%s%02x", saltBuffer, (uint8_t)currentHash->originalSalt[i]);
                    }
                }
                
                // Write the hash/salt in the proper order
                if (this->SaltIsFirst) {
                    fprintf(this->OutputFile, "%s%c%s%c", saltBuffer, 
                        this->OutputSeparator, hashBuffer, this->OutputSeparator);
                } else {
                    fprintf(this->OutputFile, "%s%c%s%c", hashBuffer, 
                        this->OutputSeparator, saltBuffer, this->OutputSeparator);
                }
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

void CHHashFileVSalted::SortHashes() {
    trace_printf("CHHashFileVSalted::SortHashes()\n");
    // Sort hashes and remove duplicates.
    std::sort(this->Hashes.begin(), this->Hashes.end(), CHHashFileVSalted::SaltedHashSortPredicate);
    this->Hashes.erase(
        std::unique(this->Hashes.begin(), this->Hashes.end(), CHHashFileVSalted::SaltedHashUniquePredicate),
        this->Hashes.end());
}

void CHHashFileVSalted::ExtractUncrackedSalts() {
    trace_printf("CHHashFileVSalted::ExtractUncrackedSalts()\n");
    // This function must be protected by a mutex outside!
    
    // Clear out the old salts.
    this->UniqueSalts.clear();
    
    // Loop through the hashes, copying unfound salts into the new structure.
    std::vector<HashSalted>::iterator HashesIt;
    for (HashesIt = this->Hashes.begin(); HashesIt < this->Hashes.end(); 
            HashesIt++) {
        
        if (!HashesIt->passwordFound) {
            this->UniqueSalts.push_back(HashesIt->salt);
        }
    }
    
    // Sort and unique the salts.
    std::sort(this->UniqueSalts.begin(), this->UniqueSalts.end());
    this->UniqueSalts.erase(
        std::unique(this->UniqueSalts.begin(), this->UniqueSalts.end()),
        this->UniqueSalts.end());
    this->UniqueSaltsValid = 1;
}

CHHashFileVSaltedDataBlob CHHashFileVSalted::ExportUniqueSaltedData() {
    trace_printf("CHHashFileVSalted::ExportUniqueSaltedData()\n");

    CHHashFileVSaltedDataBlob returnBlob;

    this->HashFileMutex.lock();
    // If the salt cache is not valid, update it.
    if (!this->UniqueSaltsValid) {
        // Update the list of uncracked salts.
        this->ExtractUncrackedSalts();
    }
    
    this->HashFileMutex.unlock();
    // Return a copy of the internal buffer.
    returnBlob.SaltData = this->UniqueSalts;

    return returnBlob;
}

bool CHHashFileVSalted::SaltedHashSortPredicate(const HashSalted &d1, const HashSalted &d2) {
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
    // Exactly equal hashes - check the salt.
    // Sort by length first.
    if (d1.salt.size() < d2.salt.size()) {
        return 1;
    } else if (d1.salt.size() > d2.salt.size()) {
        return 0;
    }
    // Salts are the same length.  Compare.
    for (i = 0; i < d1.salt.size(); i++) {
        if (d1.salt[i] == d2.salt[i]) {
            continue;
        } else if (d1.salt[i] > d2.salt[i]) {
            return 0;
        } else if (d1.salt[i] < d2.salt[i]) {
            return 1;
        }
    }
    return 0;
}

bool CHHashFileVSalted::SaltedHashUniquePredicate(const HashSalted &d1, const HashSalted &d2) {
    if ((memcmp(&d1.hash[0], &d2.hash[0], d1.hash.size()) == 0) &&
            (d1.salt.size() == d2.salt.size()) &&
            (memcmp(&d1.salt[0], &d2.salt[0], d1.salt.size()) == 0)){
        return 1;
    }
    return 0;
}

std::vector<std::vector<uint8_t> > CHHashFileVSalted::ExportUncrackedHashList() {
    trace_printf("CHHashFileVSalted::ExportUncrackedHashList()\n");
    std::vector<std::vector<uint8_t> > ReturnHashes;
    this->HashFileMutex.lock();
    
    // Loop through the hashes, copying unfound hashes into the new structure.
    std::vector<HashSalted>::iterator HashesIt;
    
    for (HashesIt = this->Hashes.begin(); HashesIt < this->Hashes.end(); 
            HashesIt++) {
        
        if (!HashesIt->passwordFound) {
            ReturnHashes.push_back(HashesIt->hash);
        }
    }

    this->HashFileMutex.unlock();
    return ReturnHashes;
}

int CHHashFileVSalted::ReportFoundPassword(std::vector<uint8_t> hash, std::vector<uint8_t> password) {
    trace_printf("CHHashFileVSalted::ReportFoundPassword()\n");
    
    uint64_t i;
    int passwordsFound = 0;

    this->HashFileMutex.lock();

    for (i = 0; i < this->TotalHashes; i++) {
        if (memcmp(&hash[0], &this->Hashes[i].hash[0], this->HashLengthBytes) == 0) {
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

void CHHashFileVSalted::PrintAllFoundHashes() {
    trace_printf("CHHashFileVSalted::PrintAllFoundHashes()\n");
    
    uint64_t i;
    int j;
    // So we can build them ahead of time and paste them together.
    // Two character arrays are here in case we need to make strings
    // in interesting formats.
    
    char saltBuffer[1024];
    char hashBuffer[1024];
    
    // These are here to turn vector<uint8_t>s into something we can
    // print to screen without so much manipulation. God bless <<
    // operators.
    
    std::string sbuf, hbuf, pbuf;
    this->HashFileMutex.lock();
    for (i = 0; i < this->TotalHashes; i++) {
        if (this->Hashes[i].passwordFound) {
            memset(saltBuffer, 0, 1024);
            memset(hashBuffer, 0, 1024);
            sbuf.empty();
            hbuf.empty();
            if (this->SaltIsLiteral) { 
                // Copy the literal salt into the buffer.
                sbuf = std::string(this->Hashes[i].originalSalt.begin(), this->Hashes[i].originalSalt.end());
            } else {
                // Print the salt in hex.
                for (j = 0; j < this->Hashes[i].originalSalt.size(); j++) {
                    sprintf(saltBuffer, "%s%02X", saltBuffer,
                            this->Hashes[i].salt[j]);
                } 
                // Shouldn't we trim this to be MaxSaltLengthBytes bytes long?
                sbuf = std::string(saltBuffer);
            }
            
            
            // Build the hash
            for (j = 0; j < this->HashLengthBytes; j++) {
                sprintf(hashBuffer, "%s%02X", hashBuffer, this->Hashes[i].hash[j]);
            }
            
            hbuf = std::string(hashBuffer);
            pbuf = std::string(this->Hashes[i].password.begin(), this->Hashes[i].password.end());
            // And print whatever order we need.
            if (this->SaltIsFirst)
                std::cout<<sbuf<<':'<<hbuf<<':'<<pbuf;
            else 
                std::cout<<hbuf<<':'<<sbuf<<':'<<pbuf;
            
            
            if (this->AddHexOutput) {
                std::cout<<":0x";
                for (j = 0; j < Hashes[i].password.size(); j++) 
                        std::cout<<std::ios_base::hex<<this->Hashes[i].password[j];
            }
            printf("\n");
        }
    }
    this->HashFileMutex.unlock();
}

void CHHashFileVSalted::PrintNewFoundHashes() {
    trace_printf("CHHashFileVSalted::PrintNewFoundHashes()\n");
    int j;
    // TODO: C++-ify this.
    this->HashFileMutex.lock();
    for (uint64_t i=0; i< this->TotalHashes; i++)
    {
        if ((this->Hashes[i].passwordFound) && (!this->Hashes[i].passwordPrinted)) {
            for (j=0; j< this->HashLengthBytes; j++)
                printf("%02X", this->Hashes[i].hash[j]);
            printf(":");
            for (j=0; j< this->Hashes[i].password.size(); j++)
                printf("%c", this->Hashes[i].password[j]);
            if (this->AddHexOutput)
            {
                printf(":0x");
                for (j = 0; j < this->Hashes[i].password.size(); j++)
                     printf("%02x", this->Hashes[i].password[j]);
                
            }
            printf("\n");
            this->Hashes[i].passwordPrinted = 1;
        
        }   
    }
    this->HashFileMutex.unlock();
}

int CHHashFileVSalted::OutputUnfoundHashesToFile(std::string filename) {
    trace_printf("CHHashFileVSalted::OutputUnfoundHashesToFile()\n");
    std::vector<HashSalted>::iterator currentHash;
    char hashBuffer[1024];
    char saltBuffer[1024];
    FILE *UnfoundHashes;
    
    this->HashFileMutex.lock();
    // If we can't open it... oh well?
    UnfoundHashes = fopen(filename.c_str(), "w");

    // Ensure the output file is opened for access before trying to write to it.
    if (UnfoundHashes) {
        for (currentHash = this->Hashes.begin();
                currentHash < this->Hashes.end(); currentHash++) {
            // Skip if already reported.
            if (!currentHash->passwordFound) {
                memset(hashBuffer, 0, sizeof(hashBuffer));
                memset(saltBuffer, 0, sizeof(saltBuffer));
                // Write the hash
                for (size_t i = 0; i < currentHash->hash.size(); i++) {
                    sprintf(hashBuffer, "%s%02x", hashBuffer, currentHash->hash[i]);
                }
                // Write the salt
                for (size_t i = 0; i < currentHash->originalSalt.size(); i++) {
                    if (this->SaltIsLiteral) {
                        // Print the character
                        sprintf(saltBuffer, "%s%c", saltBuffer, currentHash->originalSalt[i]);
                    } else {
                        // Print the hex
                        sprintf(saltBuffer, "%s%02x", saltBuffer, (uint8_t)currentHash->originalSalt[i]);
                    }
                }
                
                // Write the hash/salt in the proper order
                if (this->SaltIsFirst) {
                    fprintf(UnfoundHashes, "%s%c%s\n", saltBuffer, 
                        this->OutputSeparator, hashBuffer);
                } else {
                    fprintf(UnfoundHashes, "%s%c%s\n", hashBuffer, 
                        this->OutputSeparator, saltBuffer);
                }
            }
        }
    }
    fflush(UnfoundHashes);
    fclose(UnfoundHashes);
    
    this->HashFileMutex.unlock();
    return 1;
    
}

void CHHashFileVSalted::ImportHashListFromRemoteSystem(std::string & remoteData)  {
    trace_printf("CHHashFileVSalted::ImportHashListFromRemoteSystem()\n");
    std::string hashBuffer;
    CHHashImplementation *HashFunction = NULL;

    this->HashFileMutex.lock();
    this->HashesProtobuf.Clear();
    this->HashesProtobuf.ParseFromString(remoteData);

    // If the salt is to be hashed, set up the class to do it.
    if (this->saltPrehashAlgorithm == CH_HASHFILE_MD5_ASCII) {
        HashFunction = new CHHashImplementationMD5();
    }
    
    //Unpack the CHHashFileVSalted general member variables other than Hashes.
    this->HashLengthBytes = this->HashesProtobuf.hash_length_bytes();
    this->MaxSaltLengthBytes = this->HashesProtobuf.salt_length_bytes();
    
    //Then unpack individual HashSalted structs.
    this->TotalHashes = this->HashesProtobuf.salted_hash_value_size();
    for(int i=0;i<this->TotalHashes;i++)
    {
        CHHashFileVSalted::HashSalted hashSalted;
        this->SaltedHashProtobuf.Clear();
        this->SaltedHashProtobuf = this->HashesProtobuf.salted_hash_value(i);
        hashBuffer = std::string(this->SaltedHashProtobuf.hash());
        hashSalted.hash = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        hashBuffer = std::string(this->SaltedHashProtobuf.salt());
        hashSalted.saltLength = hashBuffer.length();
        hashSalted.salt = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());

        // Hash the salt if needed.
        if (this->saltPrehashAlgorithm == CH_HASHFILE_MD5_ASCII) {
            hashSalted.salt = HashFunction->
                hashDataAsciiVector(hashSalted.salt);
        }

        hashSalted.password = std::vector<uint8_t>();
        hashSalted.passwordFound = 0;
        hashSalted.passwordOutputToFile = 0;
        hashSalted.passwordPrinted = 0;
        this->Hashes.push_back(hashSalted);
    }
    this->HashesProtobuf.Clear();
    this->SaltedHashProtobuf.Clear();
    
    this->clearCaches();

    if (HashFunction) {
        delete HashFunction;
    }
    this->TotalHashesRemaining = this->TotalHashes;

    this->HashFileMutex.unlock();
}

void CHHashFileVSalted::ExportHashListToRemoteSystem(std::string * exportData) {
    trace_printf("CHHashFileVSalted::ExportHashListToRemoteSystem()\n");
   
    this->HashesProtobuf.Clear();
    MFNHashFileSaltedProtobuf_SaltedHash * newSaltedHashProtobuf;
    
    std::string hashBuffer;
    
    this->HashFileMutex.lock();
    
    // If the cache is valid, simply return it.
    if (this->FullHashExportProtobufCache.size()) {
        *exportData = this->FullHashExportProtobufCache;
        this->HashFileMutex.unlock();
        return;
    }

    
    // Here we have to pack a nested message: a bunch of SaltedHash, each of
    // which contains a salt and a hash.
    std::vector<CHHashFileVSalted::HashSalted>::iterator i;
    
    // Now pack individual HashSalted structs 
    for(i=this->Hashes.begin();i<this->Hashes.end();i++)
    {
        if (!i->passwordFound) {
            newSaltedHashProtobuf = this->HashesProtobuf.add_salted_hash_value();
            hashBuffer = std::string(i->hash.begin(), i->hash.end());
            newSaltedHashProtobuf->set_hash(hashBuffer);
            // Pack the original salt - not the hashed one if present.
            hashBuffer = std::string(i->originalSalt.begin(), i->originalSalt.end());
            newSaltedHashProtobuf->set_salt(hashBuffer);
        }
    }
    // Pack CHHashFileVSalted general member variables
    this->HashesProtobuf.set_hash_length_bytes(this->HashLengthBytes);
    this->HashesProtobuf.set_salt_length_bytes(this->MaxSaltLengthBytes);
    
    //Danger: Please be sure to have some storage allocated to this pointer.
    //I shouldn't have to say this, but I will anyway.
    this->HashesProtobuf.SerializeToString(exportData);
    
    // Store the result for future use.
    this->FullHashExportProtobufCache = *exportData;
    
    this->HashFileMutex.unlock();
}


void CHHashFileVSalted::ImportUniqueSaltsFromRemoteSystem(std::string & remoteData) {
    trace_printf("CHHashFileVSalted::ImportUniqueSaltsFromRemoteSystem()\n");
    std::string hashBuffer;
    std::vector<uint8_t> salt;
    CHHashImplementation *HashFunction = NULL;

    this->HashFileMutex.lock();
    this->HashesProtobuf.Clear();
    this->HashesProtobuf.ParseFromString(remoteData);

    // If the salt is to be hashed, set up the class to do it.
    if (this->saltPrehashAlgorithm == CH_HASHFILE_MD5_ASCII) {
        HashFunction = new CHHashImplementationMD5();
    }

    this->clearCaches();
    this->UniqueSalts.clear();
    
    for(int i = 0; i < this->HashesProtobuf.salted_hash_value_size(); i++) {
        this->SaltedHashProtobuf.Clear();
        this->SaltedHashProtobuf = this->HashesProtobuf.salted_hash_value(i);
        
        hashBuffer = std::string(this->SaltedHashProtobuf.salt());
        salt = std::vector<uint8_t>(hashBuffer.begin(), hashBuffer.end());
        /*
        printf("Got salt: ");
        for (int j = 0; j < salt.size(); j++) {
            printf("%c", salt[j]);
        }
        printf("   ");
        for (int j = 0; j < salt.size(); j++) {
            printf("%02x", (uint8_t)salt[j]);
        }
        printf("\n");
        */
        if (this->saltPrehashAlgorithm == CH_HASHFILE_MD5_ASCII) {
            salt = HashFunction->
                hashDataAsciiVector(salt);
        }
        /*
        printf("Pushing salt: ");
        for (int j = 0; j < salt.size(); j++) {
            printf("%c", salt[j]);
        }
        printf("\n");
        */
        
        this->UniqueSalts.push_back(salt);
    }
    this->HashesProtobuf.Clear();
    this->SaltedHashProtobuf.Clear();

    // Sort and unique the salts.
    std::sort(this->UniqueSalts.begin(), this->UniqueSalts.end());
    this->UniqueSalts.erase(
        std::unique(this->UniqueSalts.begin(), this->UniqueSalts.end()),
        this->UniqueSalts.end());
    this->UniqueSaltsValid = 1;

    /**
     * Now that there is a sorted list of salts, go through and mark any hash
     * structure without a salt as found, so it is not exported to future
     * requests for the hashes.
     */
    std::vector<CHHashFileVSalted::HashSalted>::iterator i;
    for (i = this->Hashes.begin(); i < this->Hashes.end(); i++) {
        // If the password has NOT been found, look for it.
        if (!i->passwordFound) {
            // Binary search the sorted list.  If not found, set the password to found.
            if (!std::binary_search(this->UniqueSalts.begin(), this->UniqueSalts.end(), i->salt)) {
                i->passwordFound = 1;
            }
        }
    }
    
    this->HashFileMutex.unlock();

    // Clean up the hash operator if it's in use.
    if (HashFunction) {
        delete HashFunction;
    }
}

void CHHashFileVSalted::ExportUniqueSaltsToRemoteSystem(std::string * exportData) {
    trace_printf("CHHashFileVSalted::ExportUniqueSaltsToRemoteSystem()\n");
    this->HashesProtobuf.Clear();
    MFNHashFileSaltedProtobuf_SaltedHash * newSaltedHashProtobuf;
    
    std::string hashBuffer;
    
    this->HashFileMutex.lock();
    
    // If the cache is valid, simply return it.
    if (this->SaltsExportProtobufCache.size()) {
        *exportData = this->SaltsExportProtobufCache;
        this->HashFileMutex.unlock();
        return;
    }
    
    // Ensure we have a unique and current list of salts.
    this->ExtractUncrackedSalts();
    
    // Here we have to pack a nested message: a bunch of SaltedHash, each of
    // which contains just the salt.
    std::vector<CHHashFileVSalted::HashSalted>::iterator i;
    
    // Now pack individual HashSalted structs with the ORIGINAL hashes.
    // They will be rehashed on the other end.
    for(i=this->Hashes.begin();i<this->Hashes.end();i++)
    {
        if (!i->passwordFound) {
            newSaltedHashProtobuf = this->HashesProtobuf.add_salted_hash_value();
            hashBuffer = std::string(i->originalSalt.begin(), i->originalSalt.end());
            newSaltedHashProtobuf->set_salt(hashBuffer);
        }
    }
    
    //Danger: Please be sure to have some storage allocated to this pointer.
    //I shouldn't have to say this, but I will anyway.
    this->HashesProtobuf.SerializeToString(exportData);
    
    // Store the result for future use.
    this->SaltsExportProtobufCache = *exportData;
    
    this->HashFileMutex.unlock();
}


#define UNIT_TEST 0

#if UNIT_TEST

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    
    std::cout<<"foo"<<std::endl;
    
    //CHHashFileVSalted HashFile(16, 0, CHHASHFILESALTED_SALT_IS_FIRST, CHHASHFILESALTED_HEX_SALT);
    CHHashFileVSalted HashFile(16, 0, CHHASHFILESALTED_HASH_IS_FIRST, CHHASHFILESALTED_LITERAL_SALT);
    
    if (argc != 2) {
        printf("Call it with the file name!\n");
        exit(1);
    }
    
    
    HashFile.OpenHashFile(argv[1]);
    
    std::cout<<(int)HashFile.GetTotalHashCount();
}

#endif
