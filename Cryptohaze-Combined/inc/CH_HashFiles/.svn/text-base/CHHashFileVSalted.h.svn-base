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

/**
 * @section DESCRIPTION
 * CHHashFileVSalted is an implementation of the CHHashFileV class for 
 * salted hash types with a simple salt separation.  It requires a separating
 * character that can be specified if not the default.
 * 
 * This class will separate out the salts, and sort/unique them to avoid doing
 * more work than needed - if a salt is present twice, this can be tried once
 * for the actual hash algorithm.  Also, the salts will be selected from the
 * passwords that are not cracked, because there's no point in trying a salt
 * that has already been cracked!  That just wastes time...
 * 
 * As a result of this, this hash class should also be able to export just the
 * unique salt list.  Remote hosts will use this to improve performance if
 * hashes have been cracked.
 * 
 */



#ifndef _CHHASHFILEVSALTED_H
#define _CHHASHFILEVSALTED_H

#include "CH_HashFiles/CHHashFileV.h"
#include <iostream>
#include <fstream>

// Some useful defines to make things easier to read.

// For newSaltIsLiteral value to the constructor
#define CHHASHFILESALTED_HEX_SALT 0
#define CHHASHFILESALTED_LITERAL_SALT 1

// For newSaltIsFirst value to the constructor
#define CHHASHFILESALTED_HASH_IS_FIRST 0
#define CHHASHFILESALTED_SALT_IS_FIRST 1




class CHHashFileVSalted : public CHHashFileV {
protected:

    /**
     * A structure to contain data for each hash.
     * 
     * This structure contains the various fields related to each hash.
     * 
     * hash: The target hash as a sequence of bytes.
     * salt: The salt to be sent to the devices - either the original salt, or
     *   the hashed version, depending on the algorithm.
     * originalSalt: If the salt is being hashed before being sent the device,
     *   this is the original - it is used to reduce network bandwidth when
     *   sending the salts to a remote device.  They can be hashed at the other
     *   endpoint if needed.
     * saltLength: The length of the salt, in bytes
     */    
    typedef struct HashSalted {
        std::vector<uint8_t> hash; /**< Hash in file order - as binary representation. */
        std::vector<uint8_t> salt; /**< Salt in file order - as binary representation. */
        std::vector<uint8_t> originalSalt; /**< Original salt, if salt is being hashed. */
        std::vector<uint8_t> password; /**< Password related to the hash, or null */
        std::string userData; /**< The username or other user data */
        char saltLength;
        char passwordPrinted; /**< True if the password has been printed to screen */
        char passwordFound; /**< True if the password is found. */
        char passwordOutputToFile; /**< True if the password has been placed in the output file. */
    } HashSalted;

    
    /**
     * A vector of all loaded hashes.
     * 
     * This is the main store of hashes.  It contains an entry for each line of
     * the hashfile loaded.
     */
    std::vector<HashSalted> Hashes;
    
    /**
     * A vector containing all the unique salts.  This will be updated at intervals
     * based on updates to hashes.  This should only contain the salts for
     * uncracked hashes.
     */
    std::vector<std::vector<uint8_t> > UniqueSalts;
    
    /**
     * True if the unique salt list is valid, else false.  This is used for
     * avoiding regenerating the list if nothing has changed.
     */
    uint8_t UniqueSaltsValid;

    /**
     * Caches for the full hash and unique salt export protobufs.  If they have
     * data, they are valid.
     * 
     * ALL FUNCTIONS THAT CHANGE THE INTERNAL STATE MUST CLEAR THIS.
     */
    std::string FullHashExportProtobufCache;
    std::string SaltsExportProtobufCache;
    
    
    /**
     * The current hash length in bytes.  Set by the constructor, and hashes
     * will be ignored if they do not match exactly.
     */
    uint32_t HashLengthBytes;
    
    /**
     * The max allowed salt length in bytes.  If 0, salt length is unlimited.
     * 
     * This may be set by the constructor.  If it is set and a salt of 
     * excessive length is found, it will be skipped.  It's probably a good
     * idea to leave this set to 0.
     */
    uint32_t MaxSaltLengthBytes;
    
    /**
     * Set if the salt value comes first in the password file.  Default is hash:salt
     */
    char SaltIsFirst;
    
    /**
     * Set if the salt should be read as a literal character string, not hex values.
     * 
     * If this is not set, the salt string is interpreted as a series of hex
     * values - a4b5c6 is a 3 byte salt.  If this is set, that sequence is 
     * read as a 6 byte salt.
     */
    char SaltIsLiteral;
    
    /**
     * The character separating hash and salt in the hash file.
     */
    char SeperatorSymbol;
    
    /**
     * Protocol buffer object used for serialization.
     */
    
    ::MFNHashFileSaltedProtobuf HashesProtobuf;
    ::MFNHashFileSaltedProtobuf_SaltedHash SaltedHashProtobuf;
    /**
     * Appends the found hashes to the specified output file.
     * 
     * This function adds new found hashes to the open output file.  It appends
     * to the end of the file, and syncs the file if possible on the OS.  If the
     * output file is not being used, this function returns 0.
     * 
     * @return True if the hashes were successfully written, else false.
     */
    virtual int OutputFoundHashesToFile();
    
    
    
    /**
     * Sorts and unique the hash list by hash value.
     * 
     * This function sorts the currently loaded hashes based on the value of
     * the hash.  It also removes duplicate hashes to reduce the workload.
     */
    virtual void SortHashes();
    
    /**
     * Extract the unique salts from uncracked hashes.
     * 
     * This function will read the Hashes vector and copy all the uncracked
     * hashes to the UniqueSalts vector, then sort the hashes by byte order and 
     * remove duplicates.
     */
    virtual void ExtractUncrackedSalts();
    
    /**
     * Sort predicate: returns true if d1.hash < d2.hash.
     * 
     * @param d1 First HashSalted struct
     * @param d2 Second HashSalted struct
     * @return true if d1.hash < d2.hash, else false.
     */
    static bool SaltedHashSortPredicate(const HashSalted &d1, const HashSalted &d2);
    
    /**
     * Unique predicate: returns true if d1.hash == d2.hash.
     * 
     * @param d1 First HashSalted struct
     * @param d2 Second HashSalted struct
     * @return true if d1.hash == d2.hash, else false.
     */
    
    static bool SaltedHashUniquePredicate(const HashSalted &d1, const HashSalted &d2);
    
    /**
     * Clear the various caches.  This should be called any time a hash is found
     * or added.
     */
    void clearCaches() {
        this->UniqueSaltsValid = 0;
        this->FullHashExportProtobufCache.clear();
        this->SaltsExportProtobufCache.clear();
    }
    
public:

    /**
     * Default constructor for CHHashFileVSalted.
     * 
     * Sets up the data needed for the class as described below.
     * newHashLengthBytes specifies the length of the actual hash in bytes - 
     * so 16 for MD5 hashes, 20 for SHA1, etc.  This does not specify the
     * length of the hash line in the file, as this may include a salt of
     * arbitrary length.
     * 
     * newSaltIsFirst is set if the line is salt:hash, else the line is
     * assumed to be hash:salt
     * 
     * newLiteralSalt is set if the salt is literal (to be copied byte for byte).
     * Else, the salt is assumed to be ascii-hex and is decoded accordingly.
     * 
     * newSeperatorSymbol is set to the symbol that separates the hash and salt.
     * 
     * 
     * @param newHashLengthBytes The length of the target hash type, in bytes.
     * @param newMaxSaltLengthBytes The max length of the salt, 0 for unlimited.
     * @param newSaltIsFirst True if salt is first on each line.
     * @param newLiteralSalt True if salt is literal, else is assumed hex.
     * @param newSeperatorSymbol Set to the symbol separating hash and salt.
     * 
     */
    CHHashFileVSalted(int newHashLengthBytes, int newMaxSaltLengthBytes, 
            char newSaltIsFirst, char newSaltIsLiteral, char newSeperatorSymbol);

    virtual int OpenHashFile(std::string filename);

    virtual std::vector<std::vector<uint8_t> > ExportUncrackedHashList();

    //virtual std::vector<std::vector<uint8_t> > ExportUniqueSalts();
    virtual CHHashFileVSaltedDataBlob ExportUniqueSaltedData();

    virtual int ReportFoundPassword(std::vector<uint8_t> hash, std::vector<uint8_t> password);

    virtual void PrintAllFoundHashes();

    virtual void PrintNewFoundHashes();

    virtual int OutputUnfoundHashesToFile(std::string filename);

    virtual void ImportHashListFromRemoteSystem(std::string & remoteData);
    virtual void ExportHashListToRemoteSystem(std::string * exportData);

    virtual void ImportUniqueSaltsFromRemoteSystem(std::string & remoteData);
    virtual void ExportUniqueSaltsToRemoteSystem(std::string * exportData);

    /**
     * Returns the current hash length in bytes.
     * 
     * @return Hash length in bytes.
     */
    virtual uint32_t GetHashLengthBytes() {
        return this->HashLengthBytes;
    }
    
};


#endif
