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

#include "GRT_Common/GRTHashes.h"
#include <string.h>

// {"NTLM", "MD5", "MD4", "SHA1"}
GRTHashes::GRTHashes() {
    // Add hashes.
    strcpy(this->Hashes[0], "NTLM");
    this->HashLengths[0] = 32; // NTLMs are 32 hex characters
    strcpy(this->Hashes[1], "MD5");
    this->HashLengths[1] = 32; // MD5s are 32 hex characters
    strcpy(this->Hashes[2], "MD4");
    this->HashLengths[2] = 32; // MD4s are 32 hex characters
    strcpy(this->Hashes[3], "SHA1");
    this->HashLengths[3] = 40; // SHA1s are 40 hex characters
    strcpy(this->Hashes[4], "SHA256");
    this->HashLengths[4] = 64; // SHA256s are 64 hex characters
    this->NumberOfHashes = 5;
}

int GRTHashes::GetHashIdFromString(const char* HashString) {
    int i;

    for (i = 0; i < this->NumberOfHashes; i++) {
        if (strcmp(HashString, this->Hashes[i]) == 0) {
            return i;
        }
    }
    return -1;
}

int GRTHashes::GetHashLengthFromId(int HashId) {
    if (HashId < this->NumberOfHashes) {
        return this->HashLengths[HashId];
    }
    return 0;
}

int GRTHashes::GetNumberOfHashes() {
    return this->NumberOfHashes;
}

const char* GRTHashes::GetHashStringFromId(int HashId) {
    if (HashId < this->NumberOfHashes) {
        return this->Hashes[HashId];
    }
    return 0;
}

