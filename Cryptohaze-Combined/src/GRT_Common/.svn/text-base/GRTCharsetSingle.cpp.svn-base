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

#include "GRT_Common/GRTCharsetSingle.h"
#include <stdio.h>
#include <stdlib.h>

int GRTCharsetSingle::getCharsetFromFile(const char *filename) {
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
        this->Charset[this->CharsetLength] = character;
        this->CharsetLength++;
    }
    return 1;
}


char *GRTCharsetSingle::getCharset() {
    char* CharsetReturn;
    int i;

    CharsetReturn = new char[256];
    for (i = 0; i < 256; i++) {
        CharsetReturn[i] = this->Charset[i];
    }
    return CharsetReturn;
}
int GRTCharsetSingle::getCharsetLength() {
    return this->CharsetLength;
}
