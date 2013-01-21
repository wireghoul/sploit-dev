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

#include "CH_Common/CHCharsetSingle.h"


int CHCharsetSingle::getCharsetFromFile(const char *filename) {
    FILE *charsetFileHandle;
    char character;

    charsetFileHandle = NULL;

    charsetFileHandle = fopen(filename, "r");
    if (!charsetFileHandle) {
        printf("ERROR: Cannot open charset file %s\n", filename);
        exit(1);
    }
    // Read in the file
    this->CharsetLength = 0;
    while (!feof(charsetFileHandle) && (this->CharsetLength < MAX_CHARSET_LENGTH)) {
        character = fgetc(charsetFileHandle);

        // Break on newline or EOF
        if ((character == EOF) || (character == 0x0A) || (character == 0x0D)) {
            break;
        }
        //printf("Character %c\n", character);
        this->Charset[this->CharsetLength] = character;
        this->CharsetLength++;
    }
    return 1;
}


char *CHCharsetSingle::getCharset() {
    char* CharsetReturn;
    int i;

    CharsetReturn = new char[MAX_CHARSET_LENGTH];
    for (i = 0; i < MAX_CHARSET_LENGTH; i++) {
        CharsetReturn[i] = this->Charset[i];
    }
    return CharsetReturn;
}

int CHCharsetSingle::getCharsetNumberElements() {
    return 1;
}

// Since only the first "set" is populated, only return that length.
int CHCharsetSingle::getCharsetLength(int element = 0) {
    if (element == 0) {
        return this->CharsetLength;
    } else {
        return 0;
    }
}

uint64_t CHCharsetSingle::getPasswordSpaceSize(int passwordLength) {
    int i;
    uint64_t passwordSpaceSize = 1;

    for (i = 0; i < passwordLength; i++) {
        passwordSpaceSize *= this->CharsetLength;
    }
    return passwordSpaceSize;
}

char CHCharsetSingle::getIsMulti() {return 0;}

void CHCharsetSingle::convertSingleCharsetToMulti(char *singleCharset, int charsetLength) {
    return;
}

// Will never be used.
void CHCharsetSingle::loadRemoteCharsetIntoCharset(char* remoteCharset) {
    return;
}