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

#ifndef _GRTHASH_H
#define _GRTHASH_H

#define MAX_HASH_TYPES 16
#define MAX_HASH_STRING_LENGTH 16



class GRTHashes {
private:
    char Hashes[MAX_HASH_TYPES][MAX_HASH_STRING_LENGTH];
    int HashLengths[MAX_HASH_TYPES];
    int NumberOfHashes;
public:
    // Basic functions to support hashes
    int GetHashIdFromString(const char* HashString);
    int GetHashLengthFromId(int HashId);
    const char*GetHashStringFromId(int HashId);
    int GetNumberOfHashes();
    //CHHash* GetHashClassFromTypeId(int HashID);

    GRTHashes();
};


#endif
