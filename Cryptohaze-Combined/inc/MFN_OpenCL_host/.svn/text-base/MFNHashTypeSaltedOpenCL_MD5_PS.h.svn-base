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

#ifndef __MFNHASHTYPEPLAINOPENCL_MD5_PS_H
#define __MFNHASHTYPEPLAINOPENCL_MD5_PS_H

#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL.h"

class MFNHashTypeSaltedOpenCL_MD5_PS : public MFNHashTypePlainOpenCL {

public:
    MFNHashTypeSaltedOpenCL_MD5_PS();
protected:
    std::vector<uint8_t> preProcessHash(std::vector<uint8_t> rawHash);

    std::vector<uint8_t> postProcessHash(std::vector<uint8_t> processedHash);

    void copyConstantDataToDevice();
    void copySaltConstantsToDevice();

    void launchKernel();

    /**
     * Prints debug data about the launch to verify the device has received
     * the same data.
     */
    void printLaunchDebugData();
    
    std::vector<std::string> getHashFileNames();
    std::string getKernelSourceString();
    std::string getHashKernelName();
    std::string getDefineStrings();

    /**
     * Return the salt offset - number of bytes to shift salt right.
     */
    int getSaltOffset() {
        return this->passwordLength % 4;
    }

    /**
     * We want the padding bit on the end of all the salts.
     */
    int getSetSaltPaddingBit() {
        return 1;
    }

};

#endif
