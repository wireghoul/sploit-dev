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

#include "MFN_Common/MFNDisplayDebug.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "CH_HashFiles/CHHashFileV.h"
#include "MFN_Common/MFNWorkunitBase.h"
#include <stdio.h>

extern MFNClassFactory MultiforcerGlobalClassFactory;

MFNDisplayDebug::MFNDisplayDebug() {
    this->HashFileClass = MultiforcerGlobalClassFactory.getHashfileClass();
    this->WorkunitClass = MultiforcerGlobalClassFactory.getWorkunitClass();
    this->printDebugOutput = 1;
}

void MFNDisplayDebug::Refresh() {
    printf("MFND: WU status: %d/%d\n", 
            this->WorkunitClass->GetNumberOfCompletedWorkunits(), 
            this->WorkunitClass->GetNumberOfWorkunits());
    printf("MFND: Hash status: %d/%d\n", 
            this->HashFileClass->GetCrackedHashCount(),
            this->HashFileClass->GetTotalHashCount());
    printf("MFND: Total rate: %s\n", this->getConvertedRateString(this->getCurrentCrackRate()).c_str());
}

void MFNDisplayDebug::setHashName(std::string newHashName) {
    printf("MFND: Setting hash name %s\n", newHashName.c_str());
}

void MFNDisplayDebug::setPasswordLen(uint16_t newPasswordLength) {
    printf("MFND: Setting password len %d\n", newPasswordLength);
}

void MFNDisplayDebug::addCrackedPassword(std::vector<uint8_t> newFoundPassword) {
    // Vector is passed in - ensure it is null terminated.
    newFoundPassword.push_back(0);
    printf("MFND: Found password %s\n", (char *)&newFoundPassword[0]);
}

void MFNDisplayDebug::addStatusLine(std::string newStatusLine) {
    printf("MFND: Status: %s\n", newStatusLine.c_str());
}

void MFNDisplayDebug::addStatusLine(char * newStatusLine) {
    printf("MFND: Status: %s\n", newStatusLine);
}

void MFNDisplayDebug::setSystemMode(int systemMode, std::string modeString) {
    printf("MFND: System status %d (%s)\n", systemMode, modeString.c_str());
}

void MFNDisplayDebug::alterNetworkClientCount(int networkClientCount) {
    printf("MFND: New network count: %d\n", networkClientCount);
}
