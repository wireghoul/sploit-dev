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

#include "CH_Common/CHCharsetMulti.h"

extern struct global_commands global_interface;

CHCharsetMulti::CHCharsetMulti() {
    memset(&this->Charset, 0, MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN);
    this->CharsetNumberElements = 0;
    memset(this->CharsetLength, 0, sizeof (int) * MAX_PASSWORD_LEN);
}

int CHCharsetMulti::getCharsetFromFile(const char *filename) {
    FILE *charsetFileHandle;
    char character;

    charsetFileHandle = NULL;

    charsetFileHandle = fopen(filename, "r");
    if (!charsetFileHandle) {
        printf("ERROR: Cannot open charset file %s\n", filename);
        exit(1);
    }

    // Read in the file
    while (!feof(charsetFileHandle)) {
        if (this->CharsetLength[this->CharsetNumberElements] > MAX_CHARSET_LENGTH) {
            printf("Charset in row %d too long!\n", this->CharsetNumberElements);
            exit(1);
        }
        character = fgetc(charsetFileHandle);

        // Break on EOF
        if ((character == EOF)) {
            break;
        }
        //printf("Character %c (0x%02x)\n", character, (int)character);

        // Deal with newlines.
        if ((character == 0x0A) || (character == 0x0D)) {
            // Increment to the next row in the charset
            this->CharsetNumberElements++;
        } else {
            this->Charset[(this->CharsetNumberElements * MAX_CHARSET_LENGTH)
                    + this->CharsetLength[this->CharsetNumberElements]] = character;
            this->CharsetLength[this->CharsetNumberElements]++;
        }

    }

    if (false) {
        for (int hash_position = 0; hash_position < MAX_PASSWORD_LEN; hash_position++) {
            printf("Character %d (%d characters): ", hash_position + 1, this->CharsetLength[hash_position]);
            for (int i = 0; i < this->CharsetLength[hash_position]; i++) {
                printf("%c", this->Charset[(MAX_CHARSET_LENGTH * hash_position) + i]);
            }
            printf("\n");
        }
    }

    return 1;
}

char *CHCharsetMulti::getCharset() {
    char* CharsetReturn;
    int i;

    CharsetReturn = new char[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN];
    for (i = 0; i < MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN; i++) {
        CharsetReturn[i] = this->Charset[i];
    }
    return CharsetReturn;
}

int CHCharsetMulti::getCharsetNumberElements() {
    return this->CharsetNumberElements;
}

int CHCharsetMulti::getCharsetLength(int element = 0) {
    return this->CharsetLength[element];
}

uint64_t CHCharsetMulti::getPasswordSpaceSize(int passwordLength) {
    int i;
    uint64_t passwordSpaceSize = 1;

    for (i = 0; i < passwordLength; i++) {
        if (this->CharsetLength[i] == 0) {
            sprintf(global_interface.exit_message, "Charset does not extend to password length %d!\n", i + 1);
            global_interface.exit = 1;
            return 0;
        }
        passwordSpaceSize *= this->CharsetLength[i];
    }
    return passwordSpaceSize;
}

char CHCharsetMulti::getIsMulti() {
    return 1;
}

// Take a single charset and convert it into the multi charset format
void CHCharsetMulti::convertSingleCharsetToMulti(char *singleCharset, int charsetLength) {
    int i, j;

    printf("Got charset len %d\n", charsetLength);
    printf("Charset:\n");
    for (j = 0; j < charsetLength; j++) {
        printf("%c", singleCharset[j]);
    }
    printf("\n");
    


    for (i = 0; i < MAX_PASSWORD_LEN; i++) {
        for (j = 0; j < charsetLength; j++) {
            this->Charset[(i * MAX_CHARSET_LENGTH) + j] = singleCharset[j];
        }
        this->CharsetLength[i] = charsetLength;
        this->CharsetNumberElements = MAX_PASSWORD_LEN;
    }

}

// Load up the charset with a remote set of data of MAX_CHARSET_LEN * MAX_PASSWORD_LEN
void CHCharsetMulti::loadRemoteCharsetIntoCharset(char* remoteCharset) {
    int charsetNumber, position;
    char foundNonNull;

    
    // Copy the charset into the local array
    memcpy(this->Charset, remoteCharset, MAX_PASSWORD_LEN * MAX_CHARSET_LENGTH);

    // Determine the lengths/etc.
    for (charsetNumber = 0; charsetNumber < MAX_PASSWORD_LEN; charsetNumber++) {
        foundNonNull = 0;
        for (position = 0; position < MAX_CHARSET_LENGTH; position++) {
            // If a non-zero value is found, update as needed.
            if (this->Charset[charsetNumber * MAX_CHARSET_LENGTH + position]) {
                foundNonNull = 1;
                // Increment the length for each one found.
                this->CharsetLength[charsetNumber]++;
            }
        }
        //printf("Charset %d, length %d\n", charsetNumber, this->CharsetLength[charsetNumber]);
        if (foundNonNull) {
            this->CharsetNumberElements++;
        }
    }
    //printf("Number elements: %d\n", this->CharsetNumberElements);
}