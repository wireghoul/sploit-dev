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

/**
 * This file implements functionality for reading RTI2 table formats as created
 * by the FreeRainbowTables project.  This is based on GPLv2 source available
 * http://gitorious.org/freerainbowtables-applications/
 * 
 * This is a rewrite of the code based on the specification available here:
 * http://freerainbowtables.com/download/rti2formatspec.pdf
 * 
 * Original code copyrights
 * Copyright 2011 Steve Thomas (Sc00bz)
 * Copyright 2011 James Nobis <quel@quelrod.net>
 * 
 * New stuff and the new format copyright Bitweasil 2012
 *  
 */

// Defines for algorithm types
#define RTI_20_ALGORITHM_CUSTOM            0
#define RTI_20_ALGORITHM_LM                1
#define RTI_20_ALGORITHM_NTLM              2
#define RTI_20_ALGORITHM_MD2               3
#define RTI_20_ALGORITHM_MD4               4
#define RTI_20_ALGORITHM_MD5               5
#define RTI_20_ALGORITHM_DOUBLE_MD5        6
#define RTI_20_ALGORITHM_DOUBLE_B_MD5      7
#define RTI_20_ALGORITHM_CISCO_PIX         8
#define RTI_20_ALGORITHM_SHA1              9
#define RTI_20_ALGORITHM_MYSQL_SHA1       10
#define RTI_20_ALGORITHM_SHA256           11
#define RTI_20_ALGORITHM_SHA384           12
#define RTI_20_ALGORITHM_SHA512           13
#define RTI_20_ALGORITHM_RIPEMD160        14
#define RTI_20_ALGORITHM_MSCACHE          15
#define RTI_20_ALGORITHM_1ST_HALF_LM_CHAL 16
#define RTI_20_ALGORITHM_2ND_HALF_LM_CHAL 17
#define RTI_20_ALGORITHM_NTLM_CHAL        18
#define RTI_20_ALGORITHM_ORACLE           19


// Defines for reduction algorithm
#define RTI_20_REDUCTION_RC  0
#define RTI_20_REDUCTION_FPM 1
#define RTI_20_REDUCTION_GRT 2


#ifndef _GRTTTABLEHEADERRTI2_H
#define _GRTTTABLEHEADERRTI2_H

#include "GRT_Common/GRTTableHeader.h"

class GRTTableHeaderRTI2 : public GRTTableHeader {
private:

    // First 4 bytes of a valid table: RTI2
    static const uint32_t RTI2_MAGIC = 0x32495452;

    struct Chain {
        uint64_t startPoint, endPoint;
    };

    struct CharacterSet {
        std::vector<uint8_t> characterSet1;
        std::vector<uint16_t> characterSet2;
        //std::vector<uint24_t> characterSet3;
        // No idea what a uint24_t type is defined in...
        std::vector<uint32_t> characterSet3;
        std::vector<uint32_t> characterSet4;
    };

    struct SubKeySpace {
        uint8_t hybridSets;
        std::vector<uint8_t> passwordLength;
        std::vector<uint8_t> charSetFlags;
        std::vector<CharacterSet> perPositionCharacterSets;
    };

    struct RTI20_Header_RainbowTableParameters {
        uint64_t minimumStartPoint;
        uint32_t chainLength;
        uint32_t tableIndex;
        uint8_t algorithm;
        uint8_t reductionFunction;
        std::string salt;
        std::vector<SubKeySpace> subKeySpaces;
        std::vector<uint32_t> checkPointPositions;
    };

    struct RTI20_Header {
        uint8_t major, minor; // '2', '0'
        uint8_t startPointBits, endPointBits, checkPointBits;
        uint32_t fileIndex, files;
        RTI20_Header_RainbowTableParameters rtParams;
    };

    struct RTI20_Index {
        uint64_t firstPrefix;
        std::vector<uint32_t> prefixIndex;
    };

    struct RTI20_File {
        RTI20_Header header;
        RTI20_Index index;

        struct // RTI20_Data
        {
            uint8_t *data;
        };
    };

#pragma pack(push)
#pragma pack(1)
    // RTI 2.0 file header.
    struct RTI20_File_Header {
        uint32_t tag; // "RTI2"
        uint8_t minor; // '0'
        uint8_t startPointBits, endPointBits, checkPointBits;
        uint32_t fileIndex, files;
        struct {
            uint64_t minimumStartPoint;
            uint32_t chainLength;
            uint32_t tableIndex;
            uint8_t algorithm;
            uint8_t reductionFunction;
        };
    };
#pragma pack(pop)

    uint8_t indexOffset;
    
    RTI20_File in;
    
    RTI20_File_Header Table_Header;
    RTI20_Index Prefix_Indexes;

    uint8_t *data;
    uint32_t chainCount;
    uint32_t chainSizeBytes;

    std::vector<SubKeySpace> subKeySpaces;
    std::vector<uint32_t> checkPointPositions;
    
    int readRTI2String( std::ifstream &fin, void *str, uint32_t charSize = 0 );
    void setMinimumStartPoint(uint64_t);

    // Determine the number of bits needed for the current password
    int determineBitsForPassword();
    

public:
    GRTTableHeaderRTI2();

    // Reads the full table header - not just the static part.
    void loadFullTableHeader(const char *filename);

    char isValidTable(const char *filename, int);

    // Reads the static table header section into memory.
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

    int getBitsInHash();
    int getBitsInPassword();
    void setBitsInHash(int);
    void setBitsInPassword(int);
    
};


#endif
