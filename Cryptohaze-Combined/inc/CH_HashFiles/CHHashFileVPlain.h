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
 * CHHashFileVPlain is an implementation of the CHHashFileV class for 
 * plain (unsalted/simple) hash types such as MD5, NTLM, SHA1, etc.
 * 
 * This class deals with files that have one hash per line, in ASCII-hex 
 * notation, newline separated.  It is provided the length of the hash and 
 * will ignore hashes that are not the correct length in the file.
 * 
 * This class also will handle the format of "username:hash" in the input file.
 * In this case, username is stored with the hash, and can be output along with
 * the hash.  If multiple usernames have the same hash, the results are handled
 * sanely.
 */



#ifndef _CHHASHFILEVPLAIN_H
#define _CHHASHFILEVPLAIN_H

#include "CH_HashFiles/CHHashFileV.h"
#include <iostream>
#include <fstream>

class CHHashFileVPlain : public CHHashFileV {
protected:

    /**
     * A structure to contain data on each hash found.
     * 
     * This structure contains the various fields related to each hash.
     */
    typedef struct HashPlain {
        std::vector<uint8_t> hash; /**< Hash in file order */
        std::vector<uint8_t> password; /**< Password related to the hash, or null */
        char passwordPrinted; /**< True if the password has been printed to screen */
        char passwordFound; /**< True if the password is found. */
        char passwordOutputToFile; /**< True if the password has been placed in the output file. */
        uint8_t algorithmType; /**< The algorithm identifier for this hash type */
        std::string userData; /**< The username or other user data */
    } HashPlain;

    
    /**
     * A vector of all loaded hashes.
     * 
     * This is the main store of hashes.  It contains an entry for each line of
     * the hashfile loaded.
     */
    std::vector<HashPlain> Hashes;
    
    /**
     * The current hash length in bytes.
     */
    uint32_t HashLengthBytes;

    /**
     * Protocol buffer object used for serialization.
     */
    
     // We could make this truly static, but can we guarantee that
     // there's only one instance of this class using the Protobuf at a time?
     // Need a static lock for that. 
    
    ::MFNHashFilePlainProtobuf Protobuf;
    
    /**
     * Cache the exported protobuf - if it has not changed, no need to recreate
     * it.  This should help with a ton of network clients hammering in at
     * once.  If this is valid, it will have a non-zero length.
     * 
     * ALL FUNCTIONS THAT CHANGE THE INTERNAL STATE MUST CLEAR THIS.
     */
    std::string ExportProtobufCache;
   
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
     * Sort predicate: returns true if d1 < d2.
     * 
     * @param d1 First HashPlain struct
     * @param d2 Second HashPlain struct
     * @return true if d1.hash < d2.hash, else false.
     */
    static bool PlainHashSortPredicate(const HashPlain &d1, const HashPlain &d2);
    
    /**
     * Unique predicate: returns true if d1 == d2.
     * 
     * @param d1 First HashPlain struct
     * @param d2 Second HashPlain struct
     * @return true if d1.hash == d2.hash, else false.
     */
    
    static bool PlainHashUniquePredicate(const HashPlain &d1, const HashPlain &d2);
    
    /**
     * Print the passed in hash to stdout.
     * 
     * @param Hash HashPlain struct containing the hash to print.
     */
    virtual void PrintHash(HashPlain &Hash);
    
public:

    /**
     * Default constructor for CHHashFileVPlain.
     * 
     * Clears variables as needed.  All non-stl variables should be cleared.
     * 
     * @param newHashLengthBytes The length of the target hash type, in bytes.
     */
    CHHashFileVPlain(int newHashLengthBytes);

    /**
     * Attempts to open a hash file with the given filename.
     * 
     * This function will attempt to open and parse the given filename.  After
     * completion, the HashFile class will be fully set up and ready to go.
     * Returns true on success, false on failure.  If an error occurs, this 
     * function will printf details of it before returning, and therefore should
     * be called before any curses GUIs are brought online.
     * 
     * @param filename The hashfile path to open.
     * @return True on success, False on failure.
     */
    virtual int OpenHashFile(std::string filename);

    
    /**
     * Exports the currently uncracked hashes in a vector of vectors.
     * 
     * This function exports a vector of vectors containing the currently
     * uncracked hashes (those without passwords).  The outer vector contains
     * a number of inner vectors equal to the number of uncracked hashes, and 
     * each inner vector contains a single hash.  The return may or may not be
     * in sorted order.  Calling code should sort if required.
     * 
     * @return The vector of vectors of currently uncracked hashes.
     */
    virtual std::vector<std::vector<uint8_t> > ExportUncrackedHashList();


    /**
     * Reports a found password.
     * 
     * This function is used to report a found password.  The hash and found 
     * password are reported.  If they are successfully imported as a new 
     * password/hash combination, the function returns 1, else 0.  0 may mean
     * that the hash is not present in the list, or may mean that the password
     * has already been reported.
     * 
     * @param hash A vector containing the hash corresponding to the found password.
     * @param password The found password for the hash.
     * @return 1 if the password is newly found, else 0.
     */
    virtual int ReportFoundPassword(std::vector<uint8_t> hash, std::vector<uint8_t> password);
    virtual int ReportFoundPassword(std::vector<uint8_t> foundHash, std::vector<uint8_t> foundPassword, uint8_t foundAlgorithmType);

    /**
     * Prints a list of all found hashes.
     * 
     * This function prints out a list of all found hashes and their passwords,
     * along with the hex of the password if requested.  It uses printf, so
     * call it after any curses display has been torn down.
     */
    virtual void PrintAllFoundHashes();


    /**
     * Prints out newly found hashes - ones that haven't been printed yet.
     * 
     * This function prints out found hashes that have not been printed yet.
     * It is used for display hashes as we find them in the daemon mode.  This
     * function uses printf, so must not be called during curses display.
     */
    virtual void PrintNewFoundHashes();

    
    /**
     * Outputs hashes that were not found to the specified filename.
     * 
     * This function outputs all the hashes that have not been found to the
     * specified filename.  They will be written in the same format that the
     * file was read in - typically just "hash", one per line.  Returns true
     * if the file was written successfully, else false.
     * 
     * @param filename The filename to write the unfound hashes to.
     * @return True if successfully written, else false.
     */
    virtual int OutputUnfoundHashesToFile(std::string filename);


    /**
     * Imports a hash list from a remote system.
     * 
     * This function is related to the network operation, and is used to import
     * a list of hashes/salts/etc from the remote system in a hashfile specific
     * format.  The only requirement is that this properly read the data
     * exported by the corresponding ExportHashListToRemoteSystem function in 
     * each class.  Other details are totally up to the implementation.  This
     * function overwrites any existing data in the class with the new received
     * data.
     * 
     *
     */
    virtual void ImportHashListFromRemoteSystem(std::string & remoteData);
    /**
     * Exports a list of hashes to a remote system.
     * 
     * This function is related to network operation, and is used to export a 
     * list of hashes or other data to the remote system.  This can be in a 
     * hashfile specific format, and the only requirement is that the
     * corresponding ImportHashListFromRemoteSystem can read the output format.
     * This function may export the entire hash list, or it may only export
     * the uncracked hashes.  If it exports the entire hash list, it should
     * also export data as to whether the hash has been cracked or not.
     * 
     * 
     */
    virtual void ExportHashListToRemoteSystem(std::string * exportData);

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
