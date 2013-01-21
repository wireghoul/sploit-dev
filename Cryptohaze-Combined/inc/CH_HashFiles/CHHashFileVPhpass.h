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
 * This file implements the hash loading for phpass hashes - base64 encoded
 * hashes with the '$H$ or '$P' prefix.  The prefix is '$X$N', where N is the 
 * number of iterations, encoded in the funky base64 encoding they use.
 * 
 * Salt is 8 bytes.
 * 
 */



#ifndef _CHHASHFILEVPHPASS_H
#define _CHHASHFILEVPHPASS_H

#include "CH_HashFiles/CHHashFileV.h"
#include <iostream>
#include <fstream>

#define PHPBB_MAGIC_BYTES "$H$"
#define PHPASS_MAGIC_BYTES "$P$"

class CHHashFileVPhpass : public CHHashFileV {
protected:

    /**
     * hash: The target hash as a sequence of binary.
     * salt: The salt to be sent to the devices.
     * originalHashString: The base64 encoded mess for output purposes.
     * iterations: The iteration count (1 << iterationCount)
     */    
    typedef struct {
        std::vector<uint8_t> hash; /**< Hash in file order - as binary representation. */
        std::vector<uint8_t> salt; /**< Salt in file order - as binary representation. */
        std::vector<uint8_t> password; /**< Password related to the hash, or null */
        std::string originalHashString; /** <Raw hash value for output */
        uint32_t iterations; /**< Iteration count value */
        char passwordPrinted; /**< True if the password has been printed to screen */
        char passwordFound; /**< True if the password is found. */
        char passwordOutputToFile; /**< True if the password has been placed in the output file. */
    } HashPhpass;
    
    /**
     * Structure to contain the salt and iteration count.  These must be stored
     * together as they are both attributes of the hash that must be correct
     * to crack it.
     */
    typedef struct {
        std::vector<uint8_t> salt;
        uint32_t iterations;
    } HashPhpassSalt;

    
    /**
     * A vector of all loaded hashes.
     * 
     * This is the main store of hashes.  It contains an entry for each line of
     * the hashfile loaded.
     */
    std::vector<HashPhpass> Hashes;
    
    /**
     * A vector containing all the unique salts.  This will be updated at intervals
     * based on updates to hashes.  This should only contain the salts for
     * uncracked hashes.
     */
    std::vector<HashPhpassSalt> UniqueSalts;
    
    /**
     * Store the salt data and iteration count data in corresponding vectors.
     * The data fields line up - [0] in both correspond, [1] correspond, etc.
     */
    std::vector<std::vector<uint8_t> > UniqueSaltValues;
    std::vector<uint32_t> UniqueSaltIterationCounts;

    /**
     * Set to true if the unique salts extracted are valid, or false if they
     * need to be reextracted.
     */
    uint8_t UniqueSaltsValid;

    /**
     * Caches for the full hash and unique salt/iteration export protobufs.  If
     * they have data, they are valid.
     * 
     * ALL FUNCTIONS THAT CHANGE THE INTERNAL STATE MUST CLEAR THIS.
     */
    std::string FullHashExportProtobufCache;
    std::string SaltsExportProtobufCache;
    
    
    /**
     * Protocol buffer object used for serialization.
     */
    
    ::MFNHashFileSaltedProtobuf HashesProtobuf;
    ::MFNHashFileSaltedProtobuf_SaltedHash SaltedHashProtobuf;

    int OutputFoundHashesToFile();

    void SortHashes();

    void ExtractUncrackedSalts();
    
    /**
     * Sort predicate: returns true if d1.hash < d2.hash.  Sorts by iteration
     * count first, then by hash value - puts all the hashes with a given
     * iteration count together for performance reasons.
     * 
     * @param d1 First HashPhpass struct
     * @param d2 Second HashPhpass struct
     * @return true if d1.hash < d2.hash, else false.
     */
    static bool PhpassHashSortPredicate(const HashPhpass &d1, const HashPhpass &d2);
    static bool PhpassSaltSortPredicate(const HashPhpassSalt &d1, const HashPhpassSalt &d2);
    
    /**
     * Unique predicate: returns true if d1.hash == d2.hash.  Checks iteration
     * count and salt.
     * 
     * @param d1 First HashPhpass struct
     * @param d2 Second HashPhpass struct
     * @return true if d1.hash == d2.hash, else false.
     */
    
    static bool PhpassHashUniquePredicate(const HashPhpass &d1, const HashPhpass &d2);
    static bool PhpassSaltUniquePredicate(const HashPhpassSalt &d1, const HashPhpassSalt &d2);

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
     * Default constructor for CHHashFileVPhpass.  Everything is defined by the
     * hash format.
     */
    CHHashFileVPhpass();

    int OpenHashFile(std::string filename);

    std::vector<std::vector<uint8_t> > ExportUncrackedHashList();

    //std::vector<std::vector<uint8_t> > ExportUniqueSalts();
    //std::vector<std::vector<uint8_t> > GetOtherDataByIndex(int dataId);
    virtual CHHashFileVSaltedDataBlob ExportUniqueSaltedData();
    
    int ReportFoundPassword(std::vector<uint8_t> hash, std::vector<uint8_t> password);

    void PrintAllFoundHashes();

    void PrintNewFoundHashes();

    int OutputUnfoundHashesToFile(std::string filename);

    void ImportHashListFromRemoteSystem(std::string & remoteData);

    void ExportHashListToRemoteSystem(std::string * exportData);

    /**
     * phpBB hashes are always 16 bytes of hash in length.
     * 
     * @return Hash length in bytes.
     */
    uint32_t GetHashLengthBytes() {
        return 16;
    }
};


#endif
