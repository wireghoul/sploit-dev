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

/* 
 * Base64 functions heavily based on base64.cpp and base64.h, code copyright
 * below.  No issues with GPL compatibility as far as I can tell.

   base64.cpp and base64.h

   Copyright (C) 2004-2008 René Nyffenegger

   This source code is provided 'as-is', without any express or implied
   warranty. In no event will the author be held liable for any damages
   arising from the use of this software.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

   1. The origin of this source code must not be misrepresented; you must not
      claim that you wrote the original source code. If you use this source code
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original source code.

   3. This notice may not be removed or altered from any source distribution.

   René Nyffenegger rene.nyffenegger@adp-gmbh.ch

*/

/**
 * @section DESCRIPTION
 * CHHashFileV is a base class for the vector hash file types.
 *
 * These are a replacement set of classes for the old CHHashType classes
 * that use vectors for passing data around instead of arrays of data.
 *
 * This should improve reliability and stability by reducing memory leaks.
 *
 * Also, this hash file type handles passing arbitrary data to the hash
 * functions - this will support cracking odd things such as file types,
 * WPA/WPA2 hashes, IKE hashes, etc.
 * 
 * Also, this no longer uses the network class.  Submitting hashes to the
 * network is handled by the upstream code if needed... I think.  This may
 * get revisited later if needed.
 */

#ifndef _CHHASHFILEV_H
#define _CHHASHFILEV_H

//#include "Multiforcer_Common/CHCommon.h"

// We need to include the mutex defines.
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "CH_HashFiles/CHHashFileVPlain.pb.h"
#include "CH_HashFiles/CHHashFileVSalted.pb.h"
#include "MFN_Common/MFNDefines.h"
#include <google/protobuf/descriptor.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <stdio.h>

/**
 * Structure to contain all of the salted data in one blob.  This contains the
 * salts, the iteration count, and other data, all in a structure for atomic
 * return.  This is designed to work around problems with race conditions in
 * data return.
 */
typedef struct {
    std::vector<std::vector<uint8_t> > SaltData;
    std::vector<uint32_t> iterationCount;
    std::vector<std::vector<uint8_t> > OtherData1;
    std::vector<std::vector<uint8_t> > OtherData2;
    std::vector<std::vector<uint8_t> > OtherData3;
    std::vector<std::vector<uint8_t> > OtherData4;
    std::vector<std::vector<uint8_t> > OtherData5;
} CHHashFileVSaltedDataBlob;

// Salt prehashing defines
#define CH_HASHFILE_NO_PREHASH 0
// MD5 ascii - like PHP returns.
#define CH_HASHFILE_MD5_ASCII 1

class CHHashFileV {
protected:
    /**
     * Mutex for hash file operations
     * 
     * This is a mutex used to protect all standard template operations.
     * As the STL is not threadsafe, this is used to enforce only one thread
     * at a time modifying things.  This should be locked before doing anything
     * with the STL types, and unlocked on exit.  In general, lock this at the
     * beginning of a function and unlock on all exit paths.
     */
    boost::mutex HashFileMutex;
    
    /**
     * Set if hex output is to be added to the output file.
     * 
     * If this is not set, the output file will have a standard "hash:password" 
     * output format (as relevant for the hash file).  If this is set, the 
     * output will be "hash:password:0x[password in hex]" - this is useful
     * for people who put spaces or other weird characters at the end of the 
     * password, as well as for non-ASCII hash output such as SL3.
     */
    char AddHexOutput;
    
    /**
     * If set, uses the "JTR style output" of username:password if applicable.
     */
    char UseJohnOutputStyle;
    
    /**
     * Path to the output filename for found hashes.
     * 
     * This contains the relative path for the file containing found hashes.
     * If it is null, the output file is not being used.
     */
    std::string OutputFilePath;
    
    /**
     * File for the hash output file.
     * 
     * This is the opened file for hash output.  If the file is not being used,
     * this is set to NULL.
     */
    FILE *OutputFile;
    
    /**
     * Character to separate output segments (hash/pass, user/hash/pass, etc).
     * 
     * This character will be inserted in the output stream to separate segments
     * of a hash output line.  This defaults to ':' but can be adjusted to other
     * characters (such as tab) if needed.
     */
    char OutputSeparator;
    
    /**
     * Character to separate input sections in hash types that support this.
     * 
     * For hash types that can have an optional input separation, this specifies
     * the character used.  This should normally be a character not present in
     * the usernames/hashes/etc.  The default is ':'.
     */
    char InputDelineator;

    /**
     * Total number of hashes currently loaded.
     */
    uint64_t TotalHashes;
    
    /**
     * Total number of hashes that have been found in the current set.
     */
    uint64_t TotalHashesFound;
    
    /**
     * Total number of hashes remaining to be found.
     */
    uint64_t TotalHashesRemaining;
    
    /**
     * The default hash algorithm for the file type, if there is one.  If the
     * kernel is for only one type, this being set will indicate the type in
     * the output, and no algorithm will be displayed.
     */
    uint8_t defaultHashAlgorithm;
    
    /**
     * This is set if the hash function (MD5/NTLM/SHA1/etc) should be printed
     * before each line of output to stdout/files.  This is most useful on a
     * multi-algorith kernel, or if you're running attacks against a number of
     * hash types to the same output file.
     */
    uint8_t printAlgorithm;
    
    /**
     * Some algorithms benefit from salts being prehashed in some way or
     * another.  If set, this will enable the pre-hashing of salts before they
     * are sent to the rest of the classes.  Consider network bandwidth
     * requirements when using this!
     */
    uint8_t saltPrehashAlgorithm;

    
    /**
     * Appends the found hashes to the specified output file.
     * 
     * This function adds new found hashes to the open output file.  It appends
     * to the end of the file, and syncs the file if possible on the OS.  If the
     * output file is not being used, this function returns 0.
     * 
     * @return True if the hashes were successfully written, else false.
     */
    virtual int OutputFoundHashesToFile() = 0;

    /**
     * Converts a string of ascii-hex into a vector of uint8_t values matching.
     *
     * Takes a std::string type and converts it into the ascii representation.
     * Will return this vector, or a null vector if there is an error.
     * If any non-[0-9,a-f] characters are present or the number of characters
     * is odd, it will error out.
     *
     * @param asciiHex The ascii string to convert
     * @return A vector consisting of the binary value of the string.
     */
    virtual std::vector<uint8_t> convertAsciiToBinary(std::string asciiHex);
    
    /**
     * Overloaded convertAsciiToBinary with a vector of char input vs string.
     * @param asciiHex Vector of chars to convert.
     * @return A vector consisting of the binary value of the string.
     */
    virtual std::vector<uint8_t> convertAsciiToBinary(std::vector<char> asciiHex);

    
    /**
     * Base64 encode a series of bytes with the given encoding string.
     * 
     * This function allows for base64 encoding of a string of bytes.  It is
     * based on René Nyffenegger's code available at
     * http://www.adp-gmbh.ch/cpp/common/base64.html but is modified to take a
     * vector of data instead of a string to better match what is going on with
     * my code.  The base64 charset is passed in as different hash functions use
     * different encodings of data for some weird reason.
     * 
     * Pass in the data you wish to encode, and the character set to encode to
     * as a string.  Magic should happen, and this is, in theory, usable in all
     * the various hash class functions.  Default is the "normal" base64
     * charset.
     * 
     * @param bytesToEncode The stream of bytes to encode.
     * @param base64Characters The base64 character set ordering to use.
     * @return A sequence of base64 encoded bytes with the specified charset.
     */
    virtual std::vector<uint8_t> base64Encode(
        std::vector<uint8_t> bytesToEncode,
        std::string base64Characters);

    virtual std::vector<uint8_t> base64Encode(
        std::vector<uint8_t> bytesToEncode);
    
    /**
     * Decode a base64 string into binary data with the specified character set.
     * 
     * Much as above, this is a modified version of the public code to use
     * vectors instead of char arrays.  It will work with whatever character set
     * you specify.  Default is the "normal" base64 charset.
     * 
     * @param bytesToDecode The base64 data to decode.
     * @param base64Characters The base64 character set ordering to use.
     * @return The decoded data.
     */
    virtual std::vector<uint8_t> base64Decode(
        std::vector<uint8_t> charactersToDecode,
        std::string base64Characters);

    virtual std::vector<uint8_t> base64Decode(
        std::vector<uint8_t> charactersToDecode);

    /**
     * This encodes a hash in the style of phpbb hashes, phppass, and possibly
     * others.  I don't really know what the algorithm is called.  Source based
     * on the phpass library.
     * 
     * @param bytesToEncode The binary data to encode.
     * @param base64Characters The base64 string in use.
     * @return The encoded string.
     */
    virtual std::vector<uint8_t> phpHash64Encode(
        std::vector<uint8_t> bytesToEncode,
        std::string base64Characters);
    virtual std::vector<uint8_t> phpHash64Encode(
        std::vector<uint8_t> bytesToEncode);

    /**
     * This decodes a phpbb style hash, phppass, and whatever else uses this
     * particular variety of "base64" encoding.
     * 
     * @param charactersToDecode The base64-ish encoded string.
     * @param base64Characters The base64 character set in use.
     * @return The decoded binary data.
     */
    virtual std::vector<uint8_t> phpHash64Decode(
        std::vector<uint8_t> charactersToDecode,
        std::string base64Characters);
    virtual std::vector<uint8_t> phpHash64Decode(
        std::vector<uint8_t> charactersToDecode);
    
public:

    /**
     * Default constructor for CHHashFileV.
     * 
     * Clears variables as needed.  All non-stl variables should be cleared.
     */
    CHHashFileV() {
        this->AddHexOutput = 0;
        this->UseJohnOutputStyle = 0;
        this->TotalHashes = 0;
        this->TotalHashesFound = 0;
        this->TotalHashesRemaining = 0;
        this->OutputFile = NULL;
        this->defaultHashAlgorithm = MFN_PASSWORD_NOT_FOUND;
        this->printAlgorithm = 0;
        this->OutputSeparator = ':';
        this->InputDelineator = ':';
        this->saltPrehashAlgorithm = CH_HASHFILE_NO_PREHASH;
    }
    
    /**
     * Sets the default hash algorithm.  If the hashfile is only dealing with a
     * single algorithm type, this will always be used (if the hashfile
     * implements it).  Otherwise, the call to set a password for a hash must
     * specify the algorithm type used (for multi-algorithm kernels).
     * 
     * @param newDefaultHashAlgorithm The algorithm ID used.
     */
    void setDefaultHashAlgorithn(uint8_t newDefaultHashAlgorithm) {
        this->defaultHashAlgorithm = newDefaultHashAlgorithm;
    }
    
    /**
     * Set the print algorithm status for the hashfile.  If this is set, the
     * output to stdout and the file will print the hash function in front of
     * the output.
     * 
     * @param newPrintAlgorithm False to disable, true to enable.
     */
    void setPrintAlgorithm(uint8_t newPrintAlgorithm) {
        this->printAlgorithm = newPrintAlgorithm;
    }

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
    virtual int OpenHashFile(std::string filename) = 0;

    
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
    virtual std::vector<std::vector<uint8_t> > ExportUncrackedHashList() = 0;


    /**
     * Reports a found password.
     * 
     * This function is used to report a found password.  The hash and found 
     * password are reported.  If they are successfully imported as a new 
     * password/hash combination, the function returns number of successful
     * additions, else 0.  0 may mean that the hash is not present in the list,
     * or may mean that the password has already been reported.
     * 
     * The hashType parameter is used to specify which hashing algorithm the
     * found password was.  This is used for multi-hash kernels to specify which
     * algorithm was used.
     * 
     * @param foundHash A vector containing the hash corresponding to the found password.
     * @param foundPassword The found password for the hash.
     * @param hashType The byte long hash type identifier associated with the hash.
     * @return Number of times password added to hash list.
     */
    virtual int ReportFoundPassword(std::vector<uint8_t> foundHash, 
        std::vector<uint8_t> foundPassword) = 0;
    virtual int ReportFoundPassword(std::vector<uint8_t> foundHash, 
        std::vector<uint8_t> foundPassword, uint8_t foundAlgorithmType) {
        // Default operation is to ignore it.
        return this->ReportFoundPassword(foundHash, foundPassword);
    };


    /**
     * Prints a list of all found hashes.
     * 
     * This function prints out a list of all found hashes and their passwords,
     * along with the hex of the password if requested.  It uses printf, so
     * call it after any curses display has been torn down.
     */
    virtual void PrintAllFoundHashes() = 0;


    /**
     * Prints out newly found hashes - ones that haven't been printed yet.
     * 
     * This function prints out found hashes that have not been printed yet.
     * It is used for display hashes as we find them in the daemon mode.  This
     * function uses printf, so must not be called during curses display.
     */
    virtual void PrintNewFoundHashes() = 0;

    /**
     * Sets the filename for output of found hashes.
     * 
     * This function sets the filename for output hashes and attempts to open
     * the file.  If it is successful, it returns true, else returns false.
     * Failures are silent - it is up to the calling code to detect that this 
     * function failed and report properly.
     * 
     * @param filename Output filename for hashes to be appended to.
     * @return True if file is opened successfully, else false.
     */
    virtual int SetFoundHashesOutputFilename(std::string filename) {
        this->OutputFilePath = filename;
        // Attempt to open the file and return the path.
        this->OutputFile = fopen(filename.c_str(), "a");
        if (this->OutputFile) {
            return 1;
        } else {
            return 0;
        }
    }

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
    virtual int OutputUnfoundHashesToFile(std::string filename) = 0;

    /**
     * Returns the total number of hashes loaded by the file.
     * 
     * @return The total number of hashes present in the hashfile.
     */
    virtual uint64_t GetTotalHashCount() {
        return this->TotalHashes;
    }

    /**
     * Returns the number of cracked hashes.
     * 
     * @return The number of cracked hashes in the current instance.
     */
    virtual uint64_t GetCrackedHashCount() {
        return this->TotalHashesFound;
    }
    
    /**
     * Returns the number of uncracked hashes remaining.
     * 
     * @return The number of uncracked hashes in the hash file.
     */
    virtual uint64_t GetUncrackedHashCount() {
        return this->TotalHashesRemaining;
    }

    /**
     * Returns the current hash length in bytes.
     * 
     * @return Hash length in bytes.
     */
    virtual uint32_t GetHashLengthBytes() {
        return 0;
    }
    
    /**
     * Enables hex output in the password output file.
     * 
     * This function allows enabling or disabling hex output in the password
     * file.  If the value passed in is true, an additional column of hex output
     * will be added to the output file.  If false, no hex will be added.
     * 
     * @param newAddHexOutput True to add hex output, false to disable.
     */
    virtual void SetAddHexOutput(char newAddHexOutput) {
        this->AddHexOutput = newAddHexOutput;
    }
    
    /**
     * Enables John style output of username:password if set.
     * 
     * @param newUseJohnOutputStyle True to use John style output, else false.
     */
    virtual void SetUseJohnOutputStyle(char newUseJohnOutputStyle) {
        this->UseJohnOutputStyle = newUseJohnOutputStyle;
    }

    /**
     * Imports a hash list from a remote system by passing a reference to a string.
     * 
     * This function is related to the network operation, and is used to import
     * a list of hashes/salts/etc from the remote system in a hashfile specific
     * format.  The only requirement is that this properly read the data
     * exported by the corresponding ExportHashListToRemoteSystem function in 
     * each class.  Other details are totally up to the implementation.  This
     * function overwrites any existing data in the class with the new received
     * data.
     */
    virtual void ImportHashListFromRemoteSystem(std::string & remoteData) = 0;
    
    /**
     * Exports a list of hashes to a remote system by passing a pointer to a string.
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
    virtual void ExportHashListToRemoteSystem(std::string * exportData) = 0;
    
    /**
     * Imports the remaining active salts from a remote system.
     * 
     * This function is related to network operation, and is used to import a
     * list of only salts from a remote system.  For salted hashes, by doing
     * this, unused hashes can be removed from the list (or the salts can be
     * stored separately to export).  In either case, it speeds cracking of
     * salted hash lists.
     * 
     * Default is to do nothing.
     * 
     * @param remoteData A string to import data from via protobuf format.
     */
    virtual void ImportUniqueSaltsFromRemoteSystem(std::string & remoteData) {};

    /**
     * Export the remaining active salts to a remote system.
     * 
     * This function exports the remaining uncracked salts over the network to
     * a remote system.  This pairs with ImportUniqueSaltsFromRemoteSystem to
     * update the remote system with only the uncracked salts.
     * 
     * Default is to do nothing.
     * 
     * @param exportData
     */
    virtual void ExportUniqueSaltsToRemoteSystem(std::string * exportData) {};
    
    /**
     * Exports all of the data needed for the salted hash types to use.
     * 
     * Instead of getting the other data needed by index, which adds the
     * possibility of race conditions, this now returns all of the needed data
     * (the salt, the iteration count, and other data if needed) as one blob
     * of data, handled atomically.  This should eliminate the possibility of
     * a race condition leading to mismatches in salt/iteration lineup or
     * things along those lines.
     * 
     * @return A structure containing a lot of vectors of salt related data.
     */
    virtual CHHashFileVSaltedDataBlob ExportUniqueSaltedData() {
        CHHashFileVSaltedDataBlob returnBlob;
        return returnBlob;
    }
    
    /**
     * Sets the output separator for sections of the hash output.
     * 
     * @param newOutputSeparator Character to separate the output with.
     */
    void SetOutputSeparator(char newOutputSeparator) {
        this->OutputSeparator = newOutputSeparator;
    }
    
    /**
     * Sets the input delineator for hash files.  This can be used to separate
     * username/hash to track data with the hash.
     * 
     * @param newInputDelineator Character to use as the input delineator.
     */
    void SetInputDelineator(char newInputDelineator) {
        this->InputDelineator = newInputDelineator;
    }
    
    void setSaltPrehashAlgorithm(int newSaltPrehashAlgorithm) {
        this->saltPrehashAlgorithm = newSaltPrehashAlgorithm;
    }
    
    /**
     * Test the phppass hashing algorithm.
     */
    void testPHPPassHash();

};

/**
 * This function exists as a complement to the byte hash defines in MFNDefines.h
 * 
 * Given a byte value for a hash function, this returns the string describing
 * it in short form (MD5, SHA1, SHA256, etc).  This is used by the various
 * hash file classes if there are multiple hash types present in the results.
 * 
 * @param hashFunctionValue The hash function byte from MFNDefines.h
 * @return A string consisting of the hash function type.
 */
std::string getHashFunctionByDefinedByte(uint8_t hashFunctionValue);

#endif
