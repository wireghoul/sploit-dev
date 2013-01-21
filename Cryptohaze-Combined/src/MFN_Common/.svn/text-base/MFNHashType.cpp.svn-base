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


#include "MFN_Common/MFNHashType.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_Common/MFNDisplay.h"

// Global class constructor factory
extern MFNClassFactory MultiforcerGlobalClassFactory;

// Static variable storage
MFNCommandLineData *MFNHashType::CommandLineData = NULL;
CHCharsetNew *MFNHashType::Charset = NULL;
MFNWorkunitBase *MFNHashType::Workunit = NULL;
CHHashFileV *MFNHashType::HashFile = NULL;
MFNDisplay *MFNHashType::Display = NULL;
uint8_t MFNHashType::HashIsBigEndian = 0;
uint32_t MFNHashType::numberThreads = 0;


// Implementation of MFNHashType common functions.

MFNHashType::MFNHashType() {
    trace_printf("MFNHashType::MFNHashType()\n");

    // Most hashes are not big endian.  Specify this as the default.
    this->HashIsBigEndian = 0;

    // Clear the class pointers
    this->CommandLineData = MultiforcerGlobalClassFactory.getCommandlinedataClass();
    this->Charset = MultiforcerGlobalClassFactory.getCharsetClass();
    this->Workunit = MultiforcerGlobalClassFactory.getWorkunitClass();
    this->HashFile = MultiforcerGlobalClassFactory.getHashfileClass();
    this->Display = MultiforcerGlobalClassFactory.getDisplayClass();

    // Set the rendezvous flag to null.
    this->threadRendezvous = 0;

    // Sort out my thread ID.
//    this->MFNHashTypeMutex.lock();
//    this->threadId = MultiforcerGlobalClassFactory.getDisplayClass()->getFreeThreadId();
//    this->numberThreads++;
//    printf("MFNHashType Thread ID %d\n", this->threadId);
//    this->MFNHashTypeMutex.unlock();
}