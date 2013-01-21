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

#ifndef _CHCHARSET_H
#define _CHCHARSET_H

#include "Multiforcer_Common/CHCommon.h"

class CHCharset {
public:
    // Reads the character set from a file.  Returns 1 on success.
    virtual int getCharsetFromFile(const char *filename) = 0;
    // Returns an array of the valid character set.
    // If multi, returns a long array of 128xlen bytes
    virtual char *getCharset() = 0;
    // Returns the number of elements in the character set.
    // Single charset = 1, multi = "number of different sets"
    virtual int getCharsetNumberElements() = 0;
    // Returns the length of the specified element.
    // For single charset, len for element 0
    virtual int getCharsetLength(int element) = 0;
    // Returns the password space size for the given charset and passlen.
    virtual uint64_t getPasswordSpaceSize(int passwordLength) = 0;
    // Returns true if this is a multi charset
    virtual char getIsMulti() = 0;
    // Converts a single charset to a multi charset as this is all that is used.
    virtual void convertSingleCharsetToMulti(char *singleCharset, int charsetLength) = 0;
    // Load a network charset into the current charset.
    virtual void loadRemoteCharsetIntoCharset(char *remoteCharset) = 0;
};


#endif
