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

#ifndef __MFNGENERALINFO_H__
#define __MFNGENERALINFO_H__

/**
 * This class contains general "runtime information" about the environment.
 * It's used to pass information between classes, and specifically to pass
 * information to the network class if needed.  This eliminates the need to
 * pass a lot of information to the network class.
 */

#include <stdint.h>

class MFNGeneralInformation {
public:
    MFNGeneralInformation() {
        this->hashId = 0;
        this->charsetClassId = 0;
        this->passwordLength = 0;
    }
    ~MFNGeneralInformation() {
        
    }
    
    void setHashId(uint32_t newHashId) {
        this->hashId = newHashId;
    }
    uint32_t getHashId() {
        return this->hashId;
    }
    
    void setCharsetClassId(uint32_t newCharsetClassId) {
        this->charsetClassId = newCharsetClassId;
    }
    uint32_t getCharsetClassId() {
        return this->charsetClassId;
    }
    
    void setPasswordLength(uint32_t newPasswordLength) {
        this->passwordLength = newPasswordLength;
    }
    uint32_t getPasswordLength() {
        return this->passwordLength;
    }
    
private:
    uint32_t hashId;
    uint32_t charsetClassId;
    uint32_t passwordLength;
    
};

#endif