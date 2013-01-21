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

#include "MFN_CPU_host/MFNHashTypePlainCPU.h"



class MFNHashTypePlainCPU_MD5 : public MFNHashTypePlainCPU {

public:
    MFNHashTypePlainCPU_MD5();
protected:
    std::vector<uint8_t> preProcessHash(std::vector<uint8_t> rawHash);

    std::vector<uint8_t> postProcessHash(std::vector<uint8_t> processedHash);

    void copyConstantDataToDevice();

    void launchKernel();
    
    void cpuSSEThread(CPUSSEThreadData threadData);

    void printLaunchDebugData();

    /**
     * Check the hash against the list on the host and report it to the hash
     * class if it is found.  This is generic to allow for longer hash lengths,
     * and the unused variables should simply not be called.  This function
     * will verify that the hash exists in the hashlist and will call the
     * appropriate addPassword call as needed.
     *
     * @param b0-b3 Input segments to the hash (the password data)
     * @param a-f Hash words (32-bit)
     */
    void checkAndReportHashCpuMD5(uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
        uint32_t a, uint32_t b, uint32_t c, uint32_t d);
};