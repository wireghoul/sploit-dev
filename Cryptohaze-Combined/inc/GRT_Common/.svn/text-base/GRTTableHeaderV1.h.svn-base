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

// Implementation of the table header for table version 1.

#ifndef _GRTTTABLEHEADERV1_H
#define _GRTTTABLEHEADERV1_H

#include "GRT_Common/GRTTableHeader.h"

class GRTTableHeaderV1 : public GRTTableHeader {
private:

    static const int CH_TABLE_HEADER_LENGTH = 8192;

    static const char MAGIC_0 = 'G';
    static const char MAGIC_1 = 'R';
    static const char MAGIC_2 = 'T';

    static const char TABLE_VERSION = 1;

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
            char HashVersion;       // ID of the hash (numeric)
            char HashName[16];      // Name of the hash function (string)
			char Reserved1[3];
            uint32_t TableIndex;    // Index
            uint32_t ChainLength;   // Length of chains - 32 bits is /fine/ here.
            uint64_t NumberChains;  // Number of chains in this table
            char IsPerfect;         // 1 if the table is perfect, else 0
            unsigned char PasswordLength; // Length of the password in characters
            unsigned char CharsetCount; // 1 for single charset, N for per-position
            unsigned char CharsetLength[16]; // Length of each character set
            char Charset[16][256];      // The charset array
            char Future_Use[2000];      // No idea what will go here, but space is left
            char Comments[1000];        // Comments
        };
        char Padding[8192];          // Pad the union to 8192 bytes of total length.
    }Table_Header;
    #pragma pack()

public:
    GRTTableHeaderV1();

    char isValidTable(const char *filename, int);

    char readTableHeader(const char *filename);

    char writeTableHeader(FILE *file);

    void printTableHeader();

    char isCompatibleWithTable(GRTTableHeader* Table2);


    char getTableVersion();
    void setTableVersion(char);

    char getHashVersion();
    void setHashVersion(char);

    char* getHashName();
    void setHashName(char*);

    uint32_t getTableIndex();
    void setTableIndex(uint32_t);

    uint32_t getChainLength();
    void setChainLength(uint32_t);

    uint64_t getNumberChains();
    void setNumberChains(uint64_t);

    char getIsPerfect();
    void setIsPerfect(char);

    char getPasswordLength();
    void setPasswordLength(char);

    char getCharsetCount();
    void setCharsetCount(char);

    char* getCharsetLengths();
    void setCharsetLengths(char*);

    char** getCharset();
    void setCharset(char**);

    char* getComments();
    void setComments(char*);

    int getBitsInHash() {
        return 128;
    }
    int getBitsInPassword() {
        return 128;
    }
    void setBitsInHash(int) {
        return;
    }
    void setBitsInPassword(int) {
        return;
    }
};


#endif
