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
 * CHHashFileLM implements the slightly odd behavior for the LM hash type.
 * 
 * Each line is actually two plain hashes joined to each other, but
 * treated as a single hash for auth purposes.
 * 
 * The CHHashPlain class gets each half in the plain hash queue, which means
 * network support and everything else that needs to operate on these is
 * untouched.  The overrides are on the input and output side, which need to
 * join the hashes together into the full hash such that the data is output
 * in the expected form (up to 14 characters of password).
 * 
 * This involves overriding most of the input/output functions.
 */


#ifndef _CHHASHFILELM_H
#define _CHHASHFILELM_H

#include "CH_HashFiles/CHHashFilePlain.h"

class CHHashFileLM : public CHHashFilePlain {
protected:

    /**
     * This structure contains the full hash data - all 16 bytes together.
     * This is used for loading and storing passwords and reporting them back
     * to the user in a sane form.
     * 
     * The Part1Inserted, Part2IsInserted indicate the half that has been inserted
     * into the main output.
     * 
     * Part1/2IsNull are set if the part is null (no data).
     */
    typedef struct LMFullHashData {
        std::vector<uint8_t> hash;
        std::vector<uint8_t> password;
        char passwordPart1Inserted; // Part 1 is inserted
        char passwordPart2Inserted; // Part 2 is inserted
        char passwordPart1IsNull; // Part 1 is null
        char passwordPart2IsNull; // Part 2 is null
        char passwordPrinted; /**< True if the password has been printed to screen */
        char passwordFound; /**< True if the password is found. */
        char passwordOutputToFile; /**< True if the password has been placed in the output file. */
        std::string userData; /**< The username or other user data */
        std::string originalHash; /**< The hash as in the file, undecoded */
    } LMFullHashData;

    // List of the full hashes.
    std::vector<LMFullHashData> fullHashList;
    
    virtual void parseFileLine(std::string fileLine, size_t lineNumber);
    virtual int outputNewFoundHashesToFile();
    virtual void MergeHalfPartsIntoFullPasswords();
    virtual void SortHashes();
    static bool LMHashFullSortPredicate(const LMFullHashData &d1, const LMFullHashData &d2);
    static bool LMHashFullUniquePredicate(const LMFullHashData &d1, const LMFullHashData &d2);
    virtual std::string formatHashToPrint(const LMFullHashData &hash);
    
public:

    CHHashFileLM();
    virtual void printAllFoundHashes();
    virtual void printNewFoundHashes();
    virtual int outputUnfoundHashesToFile(std::string filename);
    
};


#endif