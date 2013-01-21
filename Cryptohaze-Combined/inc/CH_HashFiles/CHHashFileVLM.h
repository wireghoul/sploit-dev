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
 * CHHashFileVPlainLM is an implementation of the CHHashFileVPlain class for 
 * LM hashes.
 * 
 * This class deals with splitting LM hashes into two separate parts
 * and merging them together at the end, as well as dealing with NULL hashes
 * internally so they do not go out to the devices.
 * 
 * It extends CHHashFileVPlain as many of the functions are the same.
 */



#ifndef _CHHASHFILEVPLAINLM_H
#define _CHHASHFILEVPLAINLM_H

#include "CH_HashFiles/CHHashFileV.h"
#include "CH_HashFiles/CHHashFileVPlain.h"
#include <iostream>
#include <fstream>


class CHHashFileVPlainLM : public CHHashFileV {
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
        char passwordReported;
        char passwordPart1Inserted; // Part 1 is inserted
        char passwordPart2Inserted; // Part 2 is inserted
        char passwordPart1IsNull; // Part 1 is null
        char passwordPart2IsNull; // Part 2 is null
        char passwordFound;
        char passwordOutputToFile;
    } LMFullHashData;

    /**
     * This structure contains the half-hashes.  As the system is fed a list of
     * half-hashes to crack, this is what goes to the rest of the system.
     */
    typedef struct LMFragmentHashData {
        unsigned char halfHash[8];
        unsigned char password[8]; // Null termination space
        char passwordReported;
        char passwordFound;
        char passwordOutputToFile;
    } LMFragmentHashData;


    std::vector<LMFullHashData> fullHashList;
    std::vector<LMFragmentHashData> halfHashList;
    
    virtual int OutputFoundHashesToFile();

    /**
     * Merges the half hashes into the full hashes/passwords.
     */
    void MergeHalfPartsIntoFullPasswords();

    void SortHashes();

    /**
     * Sort predicate: returns true if d1 < d2.
     * 
     * @param d1 First LMFullHashData struct
     * @param d2 Second LMFullHashData struct
     * @return true if d1.hash < d2.hash, else false.
     */
    static bool LMHashFullSortPredicate(const LMFullHashData &d1, const LMFullHashData &d2);
    static bool LMHashHalfSortPredicate(const LMFragmentHashData &d1, const LMFragmentHashData &d2);
    
    /**
     * Unique predicate: returns true if d1 == d2.
     * 
     * @param d1 First HashPlain struct
     * @param d2 Second HashPlain struct
     * @return true if d1.hash == d2.hash, else false.
     */
    
    static bool LMHashFullUniquePredicate(const LMFullHashData &d1, const LMFullHashData &d2);
    static bool LMHashHalfUniquePredicate(const LMFragmentHashData &d1, const LMFragmentHashData &d2);

public:

    CHHashFileVPlainLM();

    virtual int OpenHashFile(std::string filename);

    virtual std::vector<std::vector<uint8_t> > ExportUncrackedHashList();

    virtual int ReportFoundPassword(std::vector<uint8_t> hash, std::vector<uint8_t> password);

    virtual void PrintAllFoundHashes();

    virtual void PrintNewFoundHashes();

    virtual int OutputUnfoundHashesToFile(std::string filename);
    
    virtual void ImportHashListFromRemoteSystem(std::string & remoteData);
    
    virtual void ExportHashListToRemoteSystem(std::string * exportData);

    // The LM half-hashes are always 8 bytes long.
    virtual uint32_t GetHashLengthBytes() {
        return 8;
    }
};


#endif
