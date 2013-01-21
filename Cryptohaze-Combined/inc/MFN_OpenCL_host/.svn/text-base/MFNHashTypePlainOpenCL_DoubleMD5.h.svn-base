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

#ifndef __MFNHASHTYPEPLAINOPENCL_DOUBLEMD5_H
#define __MFNHASHTYPEPLAINOPENCL_DOUBLEMD5_H

#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL.h"

class MFNHashTypePlainOpenCL_DoubleMD5 : public MFNHashTypePlainOpenCL {

public:
    MFNHashTypePlainOpenCL_DoubleMD5();
protected:
    void copyConstantDataToDevice();

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
};

#endif
