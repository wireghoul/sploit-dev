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

#include <GRT_Common/GRTTableHeaderRTI2.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>



GRTTableHeaderRTI2::GRTTableHeaderRTI2() {
    // Clear the table header and set the magic
    memset(&this->Table_Header, 0, sizeof(this->Table_Header));
}

char GRTTableHeaderRTI2::isValidTable(const char *filename, int hashVersion) {

    if (!this->readTableHeader(filename)) {
        printf("Unable to read the table header.\n");
        return 0;
    }
    
    // Checks per RTI2Reader.cpp constructor from FRT
    if (this->Table_Header.tag != this->RTI2_MAGIC) {
        printf("readHeader bad tag - this is not a RTI2 file\n"); 
        exit(3); // bad tag
    }
    
    if (this->Table_Header.minor != 0) {
        printf("readHeader bad minor version\n");
        exit(4); // bad minor version
    }

    if (this->Table_Header.startPointBits == 0 || this->Table_Header.startPointBits > 64) {
        printf("readHeader parsing error\n");
        printf("header.startPointBits: %u\n", this->Table_Header.startPointBits);
        exit(2); // invalid header
    }

    if (this->Table_Header.endPointBits == 0 || this->Table_Header.endPointBits > 64)
    {
        printf("readHeader parsing error\n");
        printf("header.endPointBits: %u\n", this->Table_Header.endPointBits);
        exit(2); // invalid header
    }

    if (this->Table_Header.checkPointBits > 64) {
        printf("readHeader parsing error\n");
        printf("header.checkPointBits: %u\n", this->Table_Header.checkPointBits);
        exit(2); // invalid header
    }

    if ((this->Table_Header.startPointBits + this->Table_Header.checkPointBits + this->Table_Header.endPointBits) > 64 ) {
        printf("readHeader parsing error\n");
        printf("header.startPointBits + header.checkPointBits + header.endPointBits > 64\n");
        printf("header.startPointBits: %u\n", this->Table_Header.startPointBits);
        printf("header.endPointBits: %u\n", this->Table_Header.endPointBits);
        printf("header.checkPointBits: %u\n", this->Table_Header.checkPointBits);
        exit(2); // invalid header
    }

    if (this->Table_Header.fileIndex > this->Table_Header.files) {
        printf("readHeader parsing error\n");
        printf("fileIndex:%u\n", this->Table_Header.fileIndex);
        printf("files: %u\n", this->Table_Header.files);
        exit(2); // invalid header
    }

    if (this->Table_Header.algorithm > 19) {
        printf("readHeader parsing error\n");
        printf("undefined algorithm: %u\n", this->Table_Header.algorithm);
        exit(2); // invalid header
    }

    if (  this->Table_Header.reductionFunction > 2 
            || (this->Table_Header.reductionFunction <  1 && this->Table_Header.tableIndex > 65535)
            || (this->Table_Header.reductionFunction == 1 && this->Table_Header.tableIndex >   255))
    {
        printf("readHeader parsing error\n");
        printf("invalid reductionFunction parameters\n");
        printf("header.reductionFunction: %u\n", this->Table_Header.reductionFunction);
        printf("header.tableIndex: %u\n", this->Table_Header.tableIndex);
        exit(2); // invalid header
    }

    if ( this->Table_Header.tableIndex != 0
            &&  ((this->Table_Header.reductionFunction < 1 && this->Table_Header.chainLength - 2 > this->Table_Header.tableIndex << 16)
                    || (this->Table_Header.reductionFunction == 2 && this->Table_Header.chainLength > this->Table_Header.tableIndex))) 
        // I think this might be "header.chainLength - 2 > header.tableIndex" need to double check
    {
        // Sc00bz remarks "(these tables suck)"
        printf("WARNING: Table index is not large enough for this chain length\n");
    }

    if ( this->Table_Header.algorithm == 0 ) {
        printf("readHeader fin.read() error, reserved algorithm\n");
        exit(2);
    }  

    // Passes basic sanity checks, free and return.
    return 1;
}

char GRTTableHeaderRTI2::readTableHeader(const char *filename){
    FILE *Table;

    // Open as a large file
    Table = fopen(filename, "rb");

    // If the table file can't be opened, return false.
    if (Table == NULL) {
        printf("Cannot open table %s: fopen failed.\n", filename);
        printf( "Error opening file: %s\n", strerror( errno ) );
        return 0;
    }

    memset(&this->Table_Header, 0, sizeof(this->Table_Header));

    // If the read fails, clean up and return false.
    if (fread(&this->Table_Header, sizeof(this->Table_Header), 1, Table) != 1) {
        fclose(Table);
        return 0;
    }

    fclose(Table);
    return 1;
};


// Do the *full* table header read.
void GRTTableHeaderRTI2::loadFullTableHeader(const char *filename) {
    FILE *Table;
    
    SubKeySpace subKeySpace;
    CharacterSet charSet;
    
    // Open as a large file
    Table = fopen(filename, "rb");

    // If the table file can't be opened, return false.
    if (Table == NULL) {
        printf("Cannot open table %s: fopen failed.\n", filename);
        printf( "Error opening file: %s\n", strerror( errno ) );
        exit(1);
    }

    memset(&this->Table_Header, 0, sizeof(this->Table_Header));

    // If the read fails, clean up and return false.
    if (fread(&this->Table_Header, sizeof(this->Table_Header), 1, Table) != 1) {
        fclose(Table);
        exit(1);
    }
    
    // Continue on the read process.
    // We're not touching salted algorithms right now, so skip the salt.
    if ( this->Table_Header.algorithm == 0
            || ( this->Table_Header.algorithm >= 15 && this->Table_Header.algorithm <= 19 )) {
        printf("ERROR: Salted rainbow tables NOT IMPLEMENTED!\n");
        exit(1);
    }
    
    //Subkey spaces
    char subKeySpacesCount = 0;
    if (!fread(&subKeySpacesCount, 1, 1, Table)) {
        printf("Error: Cannot read subKeySpaceCount!\n");
        exit(1);
    }
    printf("Read subKeySpacesCount %d\n", subKeySpacesCount);
    if (subKeySpacesCount == 0) {
        printf("invalid subKeySpacesCount 0!\n");
        exit(1);
    }
    
    for (int a = 0; a < subKeySpacesCount; a++ ) {
        uint8_t hybridSets;
        // Clear out the stuff for the position.
        subKeySpace.perPositionCharacterSets.clear();
        subKeySpace.passwordLength.clear();
        if (!fread(&hybridSets, 1, 1, Table)) {
                printf("Error: Cannot read hybridSets!\n");
                exit(1);
        }
        printf("hybridSets: %u\n", hybridSets);
        if (hybridSets == 0) {
            printf("Error: hybridSets == 0!\n");
            exit(1);
        }
        subKeySpace.hybridSets = hybridSets;
        for (int b = 0; b < hybridSets; b++ ) {
            uint8_t passwordLength, characterSetFlags;
            if (!fread(&passwordLength, 1, 1, Table)) {
                printf("Error: Cannot read passwordLength!\n");
                exit(1);
            }
            printf("passwordLength: %u\n", passwordLength);
            if (passwordLength == 0) {
                printf("Error: passwordLength == 0!\n");
                exit(1);
            }
            subKeySpace.passwordLength.push_back(passwordLength);

            if (!fread(&characterSetFlags, 1, 1, Table)) {
                printf("Error: Cannot read characterSetFlags!\n");
                exit(1);
            }
            printf("characterSetFlags: %u\n", characterSetFlags);
            if (characterSetFlags == 0) {
                printf("Error: characterSetFlags == 0!\n");
                exit(1);
            }
            subKeySpace.charSetFlags.push_back( characterSetFlags );
            
            charSet.characterSet1.clear();
            charSet.characterSet2.clear();
            charSet.characterSet3.clear();
            charSet.characterSet4.clear();
            
            if (characterSetFlags & 1) {
                // Single byte characters
                uint8_t charsetLength;
                if (!fread(&charsetLength, 1, 1, Table)) {
                    printf("Error: Cannot read charsetLength!\n");
                    exit(1);
                }
                printf("charsetLength: %u\n", charsetLength);
                // Not even going to ask why it's length + 1...
                for (int pos = 0; pos <= charsetLength; pos++) {
                    uint8_t character;
                    if (!fread(&character, 1, 1, Table)) {
                        printf("Error: Cannot read character!\n");
                        exit(1);
                    }
                    printf("%c", character);
                    charSet.characterSet1.push_back(character);
                }
                printf("\n");
                subKeySpace.perPositionCharacterSets.push_back(charSet);
                
            } else {
                printf("Unsupported charsetFlags %u\n", characterSetFlags);
                exit(1);
            }
            subKeySpaces.push_back(subKeySpace);
        }
    }
    
    if (this->Table_Header.checkPointBits) {
        printf("Checkpoint bits != 0 unsupported!\n");
        exit(1);
    }
    uint64_t firstPrefix;
    uint32_t prefixCount;
    uint8_t prefixChainCount;
    
    if (!fread(&firstPrefix, 8, 1, Table)) {
        printf("Cannot read first prefix!\n");
        exit(1);
    }
    printf("firstPrefix: 0x%016lx\n", firstPrefix);
    
    if (!fread(&prefixCount, 4, 1, Table)) {
        printf("Cannot read prefix count!\n");
        exit(1);
    }
    printf("prefixCount: 0x%08x\n", prefixCount);
    
    uint32_t sum = 0;
    uint8_t *indexTmp;
    
    indexTmp = new uint8_t[prefixCount];

    if (!fread(indexTmp, prefixCount, 1, Table)) {
        printf("Error reading prefixes\n");
        exit(1);
    }

    this->Prefix_Indexes.prefixIndex.reserve(prefixCount + 1);

    this->Prefix_Indexes.prefixIndex.push_back(sum);

    for (int a = 0; a < prefixCount; a++) {
        sum += indexTmp[a];
        this->Prefix_Indexes.prefixIndex.push_back(sum);
    }

    delete [] indexTmp;
    this->chainSizeBytes = (this->Table_Header.startPointBits + this->Table_Header.checkPointBits + this->Table_Header.endPointBits + 7) >> 3;

    this->chainCount = sum;
    
    fclose(Table);
    return;
}


// Not writing RTI2 tables for now...
char GRTTableHeaderRTI2::writeTableHeader(FILE *file){
    return 0;
};

void GRTTableHeaderRTI2::printTableHeader(){
    printf("header.tag: %u\n", this->Table_Header.tag);
    printf("header.minor: %u\n", (uint32_t)this->Table_Header.minor);
    printf("header.startPointBits: %u\n", (uint32_t)this->Table_Header.startPointBits);
    printf("header.endPointBits: %u\n", (uint32_t)this->Table_Header.endPointBits);
    printf("header.checkPointBits: %u\n", (uint32_t)this->Table_Header.checkPointBits);
    printf("header.fileIndex: %u\n", (uint32_t)this->Table_Header.fileIndex);
    printf("header.files: %u\n", (uint32_t)this->Table_Header.files);
    printf("header.minimumStartPoint: %lu\n", (uint64_t)this->Table_Header.minimumStartPoint);
    printf("header.chainLength: %u\n", (uint32_t)this->Table_Header.chainLength);
    printf("header.tableIndex: %u\n", (uint16_t)this->Table_Header.tableIndex);
    printf("header.algorithm: %u\n", (uint32_t)this->Table_Header.algorithm);
    printf("header.reductionFunction: %u\n", (uint32_t)this->Table_Header.reductionFunction);
};

char GRTTableHeaderRTI2::isCompatibleWithTable(GRTTableHeader* Table2){
    // Basically, match a bunch of important stuff together, and if it doesn't match,
    // return false.
    int i, j;
/*
    // Must be the same hash.
    if (this->Table_Header.HashVersion != Table2->getHashVersion()) {
        return 0;
    }

    // Must be the same table index.
    if (this->Table_Header.TableIndex != Table2->getTableIndex()) {
        return 0;
    }

    // Must have the same chain length
    if (this->Table_Header.ChainLength != Table2->getChainLength()) {
        return 0;
    }

    // Password length must be the same
    if (this->Table_Header.PasswordLength != Table2->getPasswordLength()) {
        return 0;
    }

    // We need the same charset setup.
    if (this->Table_Header.CharsetCount != Table2->getCharsetCount()) {
        return 0;
    }

    // Verify that the number of characters in each charset is the same
    char *Table2CharsetLength = Table2->getCharsetLengths();

    for (i = 0; i < this->Table_Header.CharsetCount; i++) {
        if (this->Table_Header.CharsetLength[i] != Table2CharsetLength[i]) {
            delete[] Table2CharsetLength;
            return 0;
        }
    }

    char **Table2Charset = Table2->getCharset();

    // If we are here, the charset metadata is OK - check the charset
    for (i = 0; i < this->Table_Header.CharsetCount; i++) {
        for (j = 0; j < this->Table_Header.CharsetLength[i]; j++) {
            if (this->Table_Header.Charset[i][j] != Table2Charset[i][j]) {
                for (i = 0; i < 16; i++)
                    delete[] Table2Charset[i];
                delete[] Table2Charset;
                return 0;
            }
        }
    }

    // Clean up the charset
    delete[] Table2CharsetLength;

    for (i = 0; i < 16; i++)
        delete[] Table2Charset[i];
    delete[] Table2Charset;
*/
    return 0;
    // If we haven't failed out so far, the tables are compatible for merging!
    return 1;

};


char GRTTableHeaderRTI2::getTableVersion(){return 0;};
void GRTTableHeaderRTI2::setTableVersion(char NewTableVersion){};

char GRTTableHeaderRTI2::getHashVersion(){return this->Table_Header.algorithm;};
void GRTTableHeaderRTI2::setHashVersion(char NewHashVersion){ };

char* GRTTableHeaderRTI2::getHashName(){
    return NULL;
};
void GRTTableHeaderRTI2::setHashName(char* NewHashName){
};

uint32_t GRTTableHeaderRTI2::getTableIndex(){return this->Table_Header.tableIndex; };
void GRTTableHeaderRTI2::setTableIndex(uint32_t NewTableIndex){this->Table_Header.tableIndex = NewTableIndex; };

uint32_t GRTTableHeaderRTI2::getChainLength(){return this->Table_Header.chainLength; };
void GRTTableHeaderRTI2::setChainLength(uint32_t NewChainLength){this->Table_Header.chainLength = NewChainLength; };

uint64_t GRTTableHeaderRTI2::getNumberChains(){return 0; };
void GRTTableHeaderRTI2::setNumberChains(uint64_t NewNumberChains){};

char GRTTableHeaderRTI2::getIsPerfect(){return 0; };
void GRTTableHeaderRTI2::setIsPerfect(char NewIsPerfect){};

char GRTTableHeaderRTI2::getPasswordLength(){return 0; };
void GRTTableHeaderRTI2::setPasswordLength(char NewPasswordLength){
};

char GRTTableHeaderRTI2::getCharsetCount(){return 0; };
void GRTTableHeaderRTI2::setCharsetCount(char NewCharsetCount){};

char* GRTTableHeaderRTI2::getCharsetLengths(){
    return NULL;
};
void GRTTableHeaderRTI2::setCharsetLengths(char* NewCharsetLengths){
    int i;
};

char** GRTTableHeaderRTI2::getCharset(){
};

void GRTTableHeaderRTI2::setCharset(char** NewCharsetArray){
};

char* GRTTableHeaderRTI2::getComments(){return NULL;};
void GRTTableHeaderRTI2::setComments(char*){ };

int GRTTableHeaderRTI2::getBitsInPassword() {
    return this->Table_Header.startPointBits;
}
int GRTTableHeaderRTI2::getBitsInHash() {
    return this->Table_Header.endPointBits;
}

void GRTTableHeaderRTI2::setBitsInHash(int newHashBits) {
}
void GRTTableHeaderRTI2::setBitsInPassword(int newPasswordBits) {
}

// Given the current password length and charset lengths, determine the number
// of bits required for the password
int GRTTableHeaderRTI2::determineBitsForPassword() {
    // Currently, we only deal with one charset.
}


#define UNIT_TEST 1
#if UNIT_TEST

#include <string.h>
int main(int argc, char *argv[]) {
    GRTTableHeaderRTI2 TableHeader;
    
    if (argc != 2) {
        printf("I needz filename!\n");
        exit(1);
    }
    
    if (TableHeader.isValidTable(argv[1], 0)) {
        printf("Table is valid!\n");
        TableHeader.printTableHeader();
    } else {
        printf("Table is not valid!\n");
    }
    
    TableHeader.loadFullTableHeader(argv[1]);

}
#endif



