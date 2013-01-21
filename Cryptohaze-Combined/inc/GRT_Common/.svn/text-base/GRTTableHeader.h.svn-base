/*
Cryptohaze GPU Rainbow Tables
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

// Table header for Cryptohaze GRT.  Abstract class.

#ifndef _GRTTTABLEHEADER_H
#define _GRTTTABLEHEADER_H

#include "GRT_Common/GRTCommon.h"
#include <limits.h>
#include <stdio.h>
#include <string>
#include <vector>

class GRTCrackDisplay;

// Abstract class for the table header types.
class GRTTableHeader {
public:
    GRTTableHeader() {
        this->Display = NULL;
    }

    // Test to see if a file looks like a valid file.  Returns true if it is.
    virtual char isValidTable(const char *filename, int hashType) = 0;

    // Reads the table header and sets the values in this class.
    // Explicitly stores and resets the file pointer.
    virtual char readTableHeader(const char *filename) = 0;

    // Writes the current table header to a file.
    // Explicitly stores and resets the file pointer.
    virtual char writeTableHeader(FILE *file) = 0;

    // Prints the current table header.
    virtual void printTableHeader() = 0;

    // Compares with another object and returns true if is a match.
    virtual char isCompatibleWithTable(GRTTableHeader* Table2) = 0;

    // Various getters and setters that MUST be present.

    virtual char getTableVersion() = 0;
    virtual void setTableVersion(char) = 0;

    virtual char getHashVersion() = 0;
    virtual void setHashVersion(char) = 0;

    virtual char* getHashName() = 0;
    virtual void setHashName(char*) = 0;

    virtual uint32_t getTableIndex() = 0;
    virtual void setTableIndex(uint32_t) = 0;

    virtual uint32_t getChainLength() = 0;
    virtual void setChainLength(uint32_t) = 0;

    virtual uint64_t getNumberChains() = 0;
    virtual void setNumberChains(uint64_t) = 0;

    virtual char getIsPerfect() = 0;
    virtual void setIsPerfect(char) = 0;

    virtual char getPasswordLength() = 0;
    virtual void setPasswordLength(char) = 0;

    virtual char getCharsetCount() = 0;
    virtual void setCharsetCount(char) = 0;

    virtual char* getCharsetLengths() = 0;
    virtual void setCharsetLengths(char*) = 0;

    virtual char** getCharset() = 0;
    virtual void setCharset(char**) = 0;

    virtual char* getComments() = 0;
    virtual void setComments(char*) = 0;

    virtual int getBitsInHash() = 0;
    virtual int getBitsInPassword() = 0;
    virtual void setBitsInHash(int) = 0;
    virtual void setBitsInPassword(int) = 0;

    // Web URL stuff - not implemented in any but VWeb
    virtual void setWebURL(std::string) { };
    virtual void setWebUsername(std::string) { };
    virtual void setWebPassword(std::string) { };

    // V3 table stuff - not implemented anywhere but V3 (though could be).
    virtual void setRandomSeedValue(uint32_t) { };
    virtual uint32_t getRandomSeedValue() {return 0;};

    virtual void setChainStartOffset(uint64_t) { };
    virtual uint64_t getChainStartOffset() {return 0;};

    // Copy the header string into the table header.
    // Better pass in 8192 bytes...
    virtual int setHeaderString(std::vector<uint8_t>) {return 0;};
    virtual std::vector<uint8_t> getHeaderString() {
        std::vector<uint8_t> returnVector;
        return returnVector;
    };
    
    virtual void setDisplay(GRTCrackDisplay *newDisplay) {
        this->Display = newDisplay;
    }
    
protected:
    GRTCrackDisplay *Display;

};


#endif
