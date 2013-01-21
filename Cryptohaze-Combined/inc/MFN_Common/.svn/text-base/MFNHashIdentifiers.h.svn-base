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

#ifndef __MFNHASHIDENTIFIERS_H__
#define __MFNHASHIDENTIFIERS_H__

/**
 * @section DESCRIPTION
 *
 * This header defines the various hash types supported by the Multiforcer.
 * 
 * This should remain largely unchanged to not break network support.
 * 
 * Add new hash types in the appropriate class, but do not modify existing values
 * without good reason!
 */


#include <stdint.h>
#include <vector>
#include <string>
#include "MFN_Common/MFNDefines.h"

/**
 * Structure to contain the hash identifiers for use.
 */
typedef struct MFNHashIdentifierData {
    // Hash ID - used to uniquely identify this hash type.
    uint32_t HashID;
    // Descriptive string for the hash: MD5, SHA1, NTLM, LM, etc.
    std::string HashDescriptor;
    // This is a text description of the hash, notes, etc.
    std::string HashDetails;
    // This is an algorithm for the hash to clarify.
    std::string HashAlgorithm;
    // Minimum and maximum supported lengths
    uint8_t MinSupportedLength;
    uint8_t MaxSupportedLength;
    // Network support functioning or not?
    uint8_t NetworkSupportEnabled;
    // Default workunit size, in bits.
    uint8_t DefaultWorkunitSize;
    // Maximum number of hashes supported by this algorithm.
    // 0: Unlimited
    uint32_t MaxHashCount;
    // Flags set if the hash type has support for this device.
    uint8_t HasOpenCLSupport;
    uint8_t HasCUDASupport;
    uint8_t HasCPUSupport;
    // Hash type class identifier
    uint32_t HashTypeIdentifier;
    uint32_t HashFileIdentifier;
    // Wordlist usage enabled
    uint8_t HasWordlistSupport;
} MFNHashIdentifierData;



class MFNHashIdentifiers {
private:
    // The current hash ID of the class: Default undefined
    uint32_t CurrentHashId;
    
    // The offset in the vector of the current hash ID
    uint32_t CurrentHashPosition;
    
    // A vector containing the hash identifier structs, filled out.
    std::vector<MFNHashIdentifierData> SupportedHashTypes;
public:

    MFNHashIdentifiers();
    
    /**
     * Returns the hash ID from the specified string.  Sets the internal
     * hash type to the found hash ID.
     * 
     * @param HashString Text string to match.
     * @return Hash ID value, or MFN_HASHTYPE_UNDEFINED if not found.
     */
    uint32_t GetHashIdFromString(std::string HashString);
    
    /**
     * Sets the hash ID to the new value.
     * @param newHashId The new hash ID to use.
     */
    void SetHashId(uint32_t newHashId);
    
    /**
     * Returns the hash name string given the hash ID.
     */
    std::string GetHashStringFromID(uint32_t);

    // Returns 0 if hash is not set.
    uint8_t GetMinSupportedLength() {
        return this->SupportedHashTypes[this->CurrentHashPosition].MinSupportedLength;
    }
    uint8_t GetMaxSupportedLength() {
        return this->SupportedHashTypes[this->CurrentHashPosition].MaxSupportedLength;
    }
    uint8_t GetIsNetworkSupported() {
        return this->SupportedHashTypes[this->CurrentHashPosition].NetworkSupportEnabled;
    }
    uint8_t GetDefaultWorkunitSizeBits() {
        return this->SupportedHashTypes[this->CurrentHashPosition].DefaultWorkunitSize;
    }
    uint32_t GetMaxHashCount() {
        return this->SupportedHashTypes[this->CurrentHashPosition].MaxHashCount;
    }
    uint8_t GetHasWordlist() {
        return this->SupportedHashTypes[this->CurrentHashPosition].HasWordlistSupport;
    }
    
    MFNHashIdentifierData GetHashData() {
        return this->SupportedHashTypes[this->CurrentHashPosition];
    }

    /**
     * Prints out all present hash types.
     */
    void PrintAllHashTypes();

    
};

#endif
