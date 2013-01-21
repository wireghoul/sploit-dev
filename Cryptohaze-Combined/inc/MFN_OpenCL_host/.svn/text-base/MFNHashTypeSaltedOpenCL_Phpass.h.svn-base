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

#ifndef __MFNHASHTYPEPLAINOPENCL_PHPASS_H
#define __MFNHASHTYPEPLAINOPENCL_PHPASS_H

#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL.h"

/**
 * This attacks Phpass hashes in brute force manner.  The hashes are an iterated
 * hash function, with the first round involving the password and the salt, and
 * subsequent rounds involving the hash and the password.
 */

class MFNHashTypeSaltedOpenCL_Phpass : public MFNHashTypePlainOpenCL {

public:
    MFNHashTypeSaltedOpenCL_Phpass();
protected:

    void copyConstantDataToDevice();
    void copySaltConstantsToDevice();

    void launchKernel();
    
    std::vector<std::string> getHashFileNames();
    std::string getKernelSourceString();
    std::string getHashKernelName();
    std::string getDefineStrings();

    /**
     * Salt will not be shifted - it's at the beginning of the block.
     */
    int getSaltOffset() {
        return 0;
    }

    /**
     * PHPBB appends the password to the salt, so do not set padding.
     */
    int getSetSaltPaddingBit() {
        return 0;
    }

};

#endif
