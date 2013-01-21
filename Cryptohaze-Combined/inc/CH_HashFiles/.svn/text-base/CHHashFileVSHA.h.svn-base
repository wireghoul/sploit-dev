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
 * 
 * CHHashFileVSHA is an implementation of the CHHashFileVPlain class for the
 * LDAP {SHA} hashes.  These are simply a base64 encoded SHA1 hash with the
 * prefix '{SHA}' - decode the base64 and go!
 * 
 * The hashes are to be newline separated and may contain a username at the
 * beginning of the line, colon separated from the hash.
 */

#ifndef _CHHASHFILEVSHA_H
#define _CHHASHFILEVSHA_H

#include "CH_HashFiles/CHHashFileVPlain.h"

class CHHashFileVSHA : public CHHashFileVPlain {
protected:
   
    /**
     * Appends the found hashes to the specified output file.
     * 
     * This function adds new found hashes to the open output file.  It appends
     * to the end of the file, and syncs the file if possible on the OS.  If the
     * output file is not being used, this function returns 0.
     * 
     * Overrides the base function to print out the proper {SHA} formatting.
     * 
     * @return True if the hashes were successfully written, else false.
     */
    virtual int OutputFoundHashesToFile();
    
    /**
     * Print the passed in hash to stdout.
     * 
     * Overrides the default to print out the proper formatting.  This means the
     * base PrintAllFoundHashes/etc function work just fine.
     * 
     * This outputs to stdout.
     * 
     * @param Hash HashPlain struct containing the hash to print.
     */
    virtual void PrintHash(HashPlain &Hash);
    
    /**
     * Prints the password to the specified file stream.  This is used to allow
     * the same function to print to stdout and to a file for the output.
     */
    virtual void PrintHashToHandle(HashPlain &Hash, FILE *stream);
    
public:

    /**
     * Default constructor for CHHashFileVPlain.
     * 
     * Calls the plain constructor with the 20 byte length.
     */
    CHHashFileVSHA();

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

};


#endif