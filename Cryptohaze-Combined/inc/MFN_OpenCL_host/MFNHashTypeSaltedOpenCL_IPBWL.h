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

#ifndef __MFNHASHTYPEPLAINOPENCL_IPBWL_H
#define __MFNHASHTYPEPLAINOPENCL_IPBWL_H

#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL.h"
#include "MFN_Common/MFNDefines.h"

/**
 * This attacks IPBWL hashes in a wordlist manner.  IPBWL hashes are a 5
 * character salt, and the output in the form of:
 * md5(md5($salt).md5($password)), lowercase MD5 hex output for the first
 * rounds (as is typical).
 */

class MFNHashTypeSaltedOpenCL_IPBWL : public MFNHashTypePlainOpenCL {

public:
    MFNHashTypeSaltedOpenCL_IPBWL();
protected:

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
    std::vector<std::string> getHashKernelNamesVector();
    std::string getDefineStrings();

    /**
     * For MD5WL, plains up to 128 are supported.
     */
    void setMaxFoundPlainLength() {
        this->maxFoundPlainLength = MFN_HASH_TYPE_WORDLIST_MAX_PASSLEN;
    }
    
    /**
     * Get the kernel to run based on the word length in blocks.
     */
    cl_kernel getKernelToRun();

    /**
     * Salt will not be shifted - it's MD5'd alone.
     */
    int getSaltOffset() {
        return 0;
    }

    /**
     * We want the padding bit on the end of all the salts.
     */
    int getSetSaltPaddingBit() {
        return 1;
    }

    void copyWordlistSizeToDevice(cl_uint wordCount, cl_uchar blocksPerWord);

};

#endif
