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

// Implementation of the table header for web use.

#ifndef _GRTTTABLEHEADERVWEB_H
#define _GRTTTABLEHEADERVWEB_H

#include "GRT_Common/GRTTableHeader.h"

#include <vector>
#include <string>

#define WEB_TABLE_URL "http://localhost/webtables/webtables.php"

class GRTTableHeaderVWeb : public GRTTableHeader {
private:

    static const int CH_TABLE_HEADER_LENGTH = 8192;

    static const char MAGIC_0 = 'G';
    static const char MAGIC_1 = 'R';
    static const char MAGIC_2 = 'T';

    static const char TABLE_VERSION = 2;

    // Table header functions/definitions
    // Don't pad this structure with anything.
    #pragma pack(1)
    // Create a 8192 byte header structure with table data.
    union Table_Header{
        struct {
            char Magic0;
            char Magic1;
            char Magic2;
            char TableVersion;      // Table version
            // == 32
            char HashVersion;       // ID of the hash (numeric)
            char HashName[16];      // Name of the hash function (string)
            unsigned char BitsInPassword;  // Bits in the password field
            unsigned char BitsInHash;      // Bits in the hash field
            unsigned char Reserved1;       // Reserved
            uint32_t TableIndex;    // Index
            uint32_t ChainLength;   // Length of chains - 32 bits is /fine/ here.
            uint64_t NumberChains;  // Number of chains in this table
            char IsPerfect;         // 1 if the table is perfect, else 0
            unsigned char PasswordLength; // Length of the password in characters
            unsigned char CharsetCount;   // 1 for single charset, N for per-position
            unsigned char CharsetLength[16]; // Length of each character set
            char Charset[16][256];      // The charset array
            uint32_t randomSeedValue;   // The random seed used to generate the table - 4 bytes
            uint64_t chainStartOffset;       // How many chains have been generated prior to this table. - 8 bytes
            char Future_Use[1988];      // No idea what will go here, but space is left (2000 - 12)
            char Comments[1000];        // Comments
        };
        char Padding[8192];          // Pad the union to 8192 bytes of total length.
    }Table_Header;
    #pragma pack()

    // Some various things we need for the web stuff

    // Set if the table path is valid.
    char tableValid;

    std::string tableURL;
    std::string tableUsername;
    std::string tablePassword;

public:
    GRTTableHeaderVWeb();

    // Check the webserver for the table header.
    char isValidTable(const char *filename, int);

    // Get the table header from the webserver.
    char readTableHeader(const char *filename);

    // Not implemented.
    char writeTableHeader(FILE *file) {return 0;}

    void printTableHeader();

    // Not implemented.
    char isCompatibleWithTable(GRTTableHeader* Table2) {
        return 0;
    }

    // Getters are valid.  Setters are not.

    char getTableVersion() {
        return this->Table_Header.TableVersion;
    }
    void setTableVersion(char) { }

    char getHashVersion() {
        return this->Table_Header.HashVersion;
    }
    void setHashVersion(char) { }

    char* getHashName();
    void setHashName(char*) { }

    uint32_t getTableIndex() {
        return this->Table_Header.TableIndex;
    }
    void setTableIndex(uint32_t) { }

    uint32_t getChainLength() {
        return this->Table_Header.ChainLength;
    }
    void setChainLength(uint32_t) { }

    uint64_t getNumberChains() {
        return this->Table_Header.NumberChains;
    }
    void setNumberChains(uint64_t) { }

    char getIsPerfect() {
        return this->Table_Header.IsPerfect;
    }
    void setIsPerfect(char) { }

    char getPasswordLength() {
        return this->Table_Header.PasswordLength;
    }
    void setPasswordLength(char) { }

    char getCharsetCount() {
        return this->Table_Header.CharsetCount;
    }
    void setCharsetCount(char) { }

    char* getCharsetLengths();
    void setCharsetLengths(char*) { }

    char** getCharset();
    void setCharset(char**) { }

    char* getComments();
    void setComments(char*) { }

    int getBitsInHash() {
        return this->Table_Header.BitsInHash;
    }
    int getBitsInPassword() {
        return this->Table_Header.BitsInPassword;
    }
    void setBitsInHash(int) { }
    void setBitsInPassword(int) { }

    // Web URL stuff - not implemented in any but VWeb
    void setWebURL(std::string newWebURL) {
        this->tableURL = newWebURL;
    };
    void setWebUsername(std::string newWebUsername) {
        this->tableUsername = newWebUsername;
    };
    void setWebPassword(std::string newWebPassword) {
        this->tablePassword = newWebPassword;
    };

    // Sets the table filenames based on the server response
    std::vector<std::string> getHashesFromServerByType(int hashType);

    std::vector<uint8_t> getHeaderString();

};


#endif
