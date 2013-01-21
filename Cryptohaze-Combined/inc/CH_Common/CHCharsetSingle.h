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

#ifndef _CHCHARSETSINGLE_H
#define _CHCHARSETSINGLE_H

#include "CH_Common/CHCharset.h"
#include "Multiforcer_Common/CHCommon.h"

class CHCharsetSingle : public CHCharset {
private:
    char Charset[MAX_CHARSET_LENGTH];
    int CharsetLength;
public:
    int getCharsetFromFile(const char *filename);
    char *getCharset();
    int getCharsetNumberElements();
    int getCharsetLength(int element);
    uint64_t getPasswordSpaceSize(int passwordLength);
    char getIsMulti();
    void convertSingleCharsetToMulti(char *singleCharset, int charsetLength);
    void loadRemoteCharsetIntoCharset(char *remoteCharset);
};


#endif
