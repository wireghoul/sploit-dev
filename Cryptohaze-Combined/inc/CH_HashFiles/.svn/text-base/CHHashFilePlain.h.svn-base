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



#ifndef _CHHASHFILEPLAIN_H
#define _CHHASHFILEPLAIN_H

#include "CH_HashFiles/CHHashFile.h"

class CHHashFilePlain : public CHHashFile {
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
        std::string originalHash; /**< The hash as in the file, undecoded */
    } HashPlain;

    
    /**
     * A vector of all loaded hashes.
     * 
     * This is the main store of hashes.  It contains an entry for each line of
     * the hashfile loaded.
     */
    std::vector<HashPlain> Hashes;
    
    /**
     * Caching for the hashes to export.
     */
    std::vector<std::vector<uint8_t> > HashExportCache;
    uint8_t HashExportCacheValid;
    
    /**
     * Protocol buffer object used for serialization.
     */
    ::MFNHashFilePlainProtobuf Protobuf;
    
    virtual int outputNewFoundHashesToFile();
    
    /**
     * Sorts and unique the hash list by hash value.
     * 
     * This function sorts the currently loaded hashes based on the value of
     * the hash.  It also removes duplicate hashes to reduce the workload.
     */
    virtual void sortHashes();
    
    /**
     * Sort predicate: returns true if d1 < d2.
     * 
     * @param d1 First HashPlain struct
     * @param d2 Second HashPlain struct
     * @return true if d1.hash < d2.hash, else false.
     */
    static bool plainHashSortPredicate(const HashPlain &d1,
            const HashPlain &d2);
    
    /**
     * Unique predicate: returns true if d1 == d2.
     * 
     * @param d1 First HashPlain struct
     * @param d2 Second HashPlain struct
     * @return true if d1.hash == d2.hash, else false.
     */
    
    static bool plainHashUniquePredicate(const HashPlain &d1,
            const HashPlain &d2);
    
    virtual void performPostLoadOperations();

    virtual void parseFileLine(std::string fileLine, size_t lineNumber);

    /**
     * Return a string of the hash, formatted.  Will include password if
     * found.
     * 
     * @param hash The hash to print.
     * @return A string containing the formatted hash.
     */
    virtual std::string formatHashToPrint(const HashPlain &hash);
    
public:

    /**
     * Default constructor for CHHashFileVPlain.
     * 
     * Clears variables as needed.  All non-stl variables should be cleared.
     * 
     * @param newHashLengthBytes The length of the target hash type, in bytes.
     */
    CHHashFilePlain(int newHashLengthBytes);

    
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
    virtual std::vector<std::vector<uint8_t> > exportUncrackedHashList();


    virtual int reportFoundPassword(std::vector<uint8_t> hash,
            std::vector<uint8_t> password);
    virtual int reportFoundPassword(std::vector<uint8_t> foundHash,
            std::vector<uint8_t> foundPassword, uint8_t foundAlgorithmType);

    /**
     * Prints a list of all found hashes.
     * 
     * This function prints out a list of all found hashes and their passwords,
     * along with the hex of the password if requested.  It uses printf, so
     * call it after any curses display has been torn down.
     */
    virtual void printAllFoundHashes();


    /**
     * Prints out newly found hashes - ones that haven't been printed yet.
     * 
     * This function prints out found hashes that have not been printed yet.
     * It is used for display hashes as we find them in the daemon mode.  This
     * function uses printf, so must not be called during curses display.
     */
    virtual void printNewFoundHashes();

    
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
    virtual int outputUnfoundHashesToFile(std::string filename);


    virtual void importHashListFromRemoteSystem(std::string &remoteData);

    virtual void createHashListExportProtobuf();
};

#endif
