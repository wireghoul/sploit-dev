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

#include "CUDA_Common/CHCudaUtils.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"

#include "CH_Common/CHCharsetNew.h"

#include "MFN_Common/MFNWorkunitRobust.h"
#include "MFN_Common/MFNWorkunitNetwork.h"
#include "MFN_Common/MFNWorkunitWordlist.h"

#include "MFN_Common/MFNDisplayDaemon.h"
#include "MFN_Common/MFNDisplayDebug.h"
#include "MFN_Common/MFNDisplayCurses.h"

#include "CH_HashFiles/CHHashFileVPlain.h"
#include "CH_HashFiles/CHHashFileVLM.h"
#include "CH_HashFiles/CHHashFileVSalted.h"
#include "CH_HashFiles/CHHashFileVPhpass.h"

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_MD5.h"

#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_MD5.h"

#include "MFN_Common/MFNCommandLineData.h"
#include "MFN_Common/MFNHashIdentifiers.h"

#include "MFN_Common/MFNGeneralInformation.h"

#include "MFN_Common/MFNNetworkServer.h"
#include "MFN_Common/MFNNetworkClient.h"

MFNClassFactory::MFNClassFactory() {
    // Charset class: Default CHCharsetNew
    this->CharsetClass = NULL;
    this->CharsetClassId = CH_CHARSET_NEW_CLASS_ID;

    // Workunit class: Default MFNWorkunitRobust
    this->WorkunitClass = NULL;
    this->WorkunitClassId = MFN_WORKUNIT_ROBUST_CLASS_ID;

    this->DisplayClass = NULL;
    this->DisplayClassId = MFN_DISPLAY_CLASS_CURSES;

    this->HashfileClass = NULL;
    this->HashfileClassId = NULL;

    this->CommandlinedataClass = NULL;
    this->CommandlinedataClassId = CH_COMMANDLINEDATA;

    this->CudaUtilsClass = NULL;
    this->CudaUtilsClassId = CH_CUDAUTILS;

    this->HashIdentifiersClass = NULL;
    this->HashIdentifiersClassId = MFN_HASHIDENTIFIERS;
    
    this->NetworkServerClass = NULL;
    this->NetworkServerPort = MFN_NETWORK_DEFAULT_PORT;
    
    this->NetworkClientClass = NULL;
    this->NetworkClientPort = MFN_NETWORK_DEFAULT_PORT;
    this->NetworkClientRemoteHost = std::string("localhost");
    this->NetworkClientOneshot = 0;
}

/**
 * Creates the charset class.  If an invalid ID is present, leaves it null.
 */
void MFNClassFactory::createCharsetClass() {
   switch(this->CharsetClassId) {
       case CH_CHARSET_NEW_CLASS_ID:
           this->CharsetClass = new CHCharsetNew();
           break;
       default:
           this->CharsetClass = NULL;
           break;
   }
}
void MFNClassFactory::destroyCharsetClass() {
    if (this->CharsetClass) {
        delete this->CharsetClass;
        this->CharsetClass = NULL;
        this->CharsetClassId = CH_CHARSET_NEW_CLASS_ID;
    }
}


/**
 * Creates the workunit class.
 */
void MFNClassFactory::createWorkunitClass() {
    switch(this->WorkunitClassId) {
        case MFN_WORKUNIT_ROBUST_CLASS_ID:
            this->WorkunitClass = new MFNWorkunitRobust();
            break;
        case MFN_WORKUNIT_NETWORK_CLASS_ID:
            this->WorkunitClass = new MFNWorkunitNetworkClient();
            break;
        case MFN_WORKUNIT_WORDLIST_CLASS_ID:
            this->WorkunitClass = new MFNWorkunitWordlist();
            break;
        default:
            this->WorkunitClass = NULL;
            break;
    }
}

void MFNClassFactory::destroyWorkunitClass() {
    if (this->WorkunitClass) {
        delete this->WorkunitClass;
        this->WorkunitClass = NULL;
        this->WorkunitClassId = MFN_WORKUNIT_ROBUST_CLASS_ID;
    }
}

void MFNClassFactory::createDisplayClass() {
    switch(this->DisplayClassId) {
        case MFN_DISPLAY_CLASS_CURSES:
            this->DisplayClass = new MFNDisplayCurses();
            break;
        case MFN_DISPLAY_CLASS_DEBUG:
            this->DisplayClass = new MFNDisplayDebug();
            break;
        case MFN_DISPLAY_CLASS_DAEMON:
            this->DisplayClass = new MFNDisplayDaemon();
            break;
        case MFN_DISPLAY_CLASS_GUI:
            /* if using interactive GUI, must call setDisplayClass() with the 
                MFNDisplay GUI implementation */
            this->DisplayClass = NULL;
            break;
        default:
            this->DisplayClass = NULL;
            break;
    }
}
void MFNClassFactory::destroyDisplayClass() {
    if (this->DisplayClass) {
        delete this->DisplayClass;
        this->DisplayClass = NULL;
        this->DisplayClassId = MFN_DISPLAY_CLASS_CURSES;
    }
}


void MFNClassFactory::createHashfileClass() {
    switch (this->HashfileClassId) {
        case CH_HASHFILE_PLAIN_16:
            this->HashfileClass = new CHHashFileVPlain(16);
            break;
        case CH_HASHFILE_PLAIN_20:
            this->HashfileClass = new CHHashFileVPlain(20);
            break;
        case CH_HASHFILE_PLAIN_32:
            this->HashfileClass = new CHHashFileVPlain(32);
            break;
        case CH_HASHFILE_LM:
            this->HashfileClass = new CHHashFileVPlainLM();
            break;
        case CH_HASHFILE_SALTED_32_PASS_SALT:
            this->HashfileClass = new CHHashFileVSalted(16, 16, 1, 0, ':');
            break;
        case CH_HASHFILE_IPB:
            this->HashfileClass = new CHHashFileVSalted(16, 5, 0, 1, ':');
            // IPB pre-hashes the salt into ASCII.
            this->HashfileClass->setSaltPrehashAlgorithm(CH_HASHFILE_MD5_ASCII);
            break;
        case CH_HASHFILE_PHPASS:
            this->HashfileClass = new CHHashFileVPhpass();
            break;
        default:
            this->HashfileClass = NULL;
            break;
    }
}
void MFNClassFactory::destroyHashfileClass() {
    if (this->HashfileClass) {
        delete this->HashfileClass;
        this->HashfileClass = NULL;
        this->HashfileClassId = NULL;
    }
}

void MFNClassFactory::createCommandlinedataClass() {
    switch (this->CommandlinedataClassId) {
        case CH_COMMANDLINEDATA:
            this->CommandlinedataClass = new MFNCommandLineData();
            break;
        case CH_COMMANDLINEDATAGUI:
            /* do nothing, the GUI will set the class once the window
            is created via setCommandlinedataClass() */
            break;
        default:
            this->CommandlinedataClass = NULL;
            break;
    }
}


void MFNClassFactory::createCudaUtilsClass() {
    switch(this->CudaUtilsClassId) {
        case CH_CUDAUTILS:
            this->CudaUtilsClass = new CHCUDAUtils();
            break;
        default:
            this->CudaUtilsClass = NULL;
            break;
    }
}

void MFNClassFactory::createHashIdentifiersClass() {
    switch(this->HashIdentifiersClassId) {
        case MFN_HASHIDENTIFIERS:
            this->HashIdentifiersClass = new MFNHashIdentifiers();
            break;
        default:
            this->HashIdentifiersClass = NULL;
            break;
    }
}

void MFNClassFactory::createGeneralInformationClass() {
    this->GeneralInformationClass = new MFNGeneralInformation();
}

void MFNClassFactory::createNetworkServerClass() {
    this->NetworkServerClass = new MFNNetworkServer(this->NetworkServerPort);
}

void MFNClassFactory::createNetworkClientClass() {
    this->NetworkClientClass = new MFNNetworkClient(this->NetworkClientRemoteHost,
            this->NetworkClientPort, this->NetworkClientOneshot);
}

void MFNClassFactory::destroyNetworkClientClass() {
    if (this->NetworkClientClass) {
        delete this->NetworkClientClass;
        this->NetworkClientClass = NULL;
    }
}
