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

#ifndef __MFNCLASSFACTORY_H__
#define __MFNCLASSFACTORY_H__

/**
 * @section DESCRIPTION
 *
 * This file implements a basic hash factory for the new Cryptohaze Multiforcer.
 *
 * The factory will be a global class, and will be accessed by all classes,
 * removing the need for the setClass type functions.  This should clean
 * code and ensure that everyone gets what they need.
 *
 * This class will return the default type of class unless another type has been
 * specified before the class is returned.  Classes will have a unique ID
 * assigned to them.  Or something.  Still working on details.
 */

#include <stdlib.h>
#include <stdint.h>


// Forward declare classes
class CHCharsetNew;
class MFNWorkunitBase;
class MFNDisplay;
class CHHashFileV;
class MFNCommandLineData;
class CHCUDAUtils;
class MFNHashIdentifiers;
class MFNGeneralInformation;
class MFNNetworkServer;
class MFNNetworkClient;

// Defines for class types
#include "MFN_Common/MFNDefines.h"

#include <string>


class MFNClassFactory {
public:
    MFNClassFactory();

    /**
     * Set the Charset class type.  Returns true on success, false on failure.
     */
    void setCharsetClassType(uint32_t newCharsetClassId) {
        this->CharsetClassId = newCharsetClassId;
    }
    CHCharsetNew *getCharsetClass() {
        if (!this->CharsetClass) {
            this->createCharsetClass();
        }
        return this->CharsetClass;
    }
    void destroyCharsetClass();


    void setWorkunitClassType(uint32_t newWorkunitClassId) {
        this->WorkunitClassId = newWorkunitClassId;
    }
    MFNWorkunitBase *getWorkunitClass() {
        if (!this->WorkunitClass) {
            this->createWorkunitClass();
        }
        return this->WorkunitClass;
    }
    void destroyWorkunitClass();
    
    
    void setDisplayClassType(uint32_t newDisplayClassId) {
        this->DisplayClassId = newDisplayClassId;
    }
    void setDisplayClass(MFNDisplay *display) {
        this->DisplayClass = display;
    }
    
    MFNDisplay *getDisplayClass() {
        if (!this->DisplayClass) {
            this->createDisplayClass();
        }
        return this->DisplayClass;
    }
    void destroyDisplayClass();

    void setHashfileClassType(uint32_t newHashfileClassId) {
        this->HashfileClassId = newHashfileClassId;
    }
    CHHashFileV *getHashfileClass() {
        if (!this->HashfileClass) {
            this->createHashfileClass();
        }
        return this->HashfileClass;
    }
    void destroyHashfileClass();
    
    void setCommandlinedataClassType(uint32_t newCommandlinedataClassId) {
        this->CommandlinedataClassId = newCommandlinedataClassId;
    }
    void setCommandlinedataClass(MFNCommandLineData *c) {
        this->CommandlinedataClass = c;
    }

    MFNCommandLineData *getCommandlinedataClass() {
        if (!this->CommandlinedataClass) {
            this->createCommandlinedataClass();
        }
        return this->CommandlinedataClass;
    }
    // The command line data class should not be destroyed.  I see no reason
    // to do so.

    void setCudaUtilsClassType(uint32_t newCudaUtilsClassId) {
        this->CudaUtilsClassId = newCudaUtilsClassId;
    }
    CHCUDAUtils *getCudaUtilsClass() {
        if (!this->CudaUtilsClass) {
            this->createCudaUtilsClass();
        }
        return this->CudaUtilsClass;
    }
    // Not sure why there's any reason to delete this class either.

    void setHashIdentifiersClassType(uint32_t newHashIdentifiersClassId) {
        this->HashIdentifiersClassId = newHashIdentifiersClassId;
    }
    MFNHashIdentifiers *getHashIdentifiersClass() {
        if (!this->HashIdentifiersClass) {
            this->createHashIdentifiersClass();
        }
        return this->HashIdentifiersClass;
    }
    // No point in deleting this class.

    MFNGeneralInformation *getGeneralInformationClass() {
        if (!this->GeneralInformationClass) {
            this->createGeneralInformationClass();
        }
        return this->GeneralInformationClass;
    }
    
    void setNetworkServerPort(uint16_t newNetworkServerPort) {
        this->NetworkServerPort = newNetworkServerPort;
    }
    MFNNetworkServer *getNetworkServerClass() {
        if (!this->NetworkServerClass) {
            this->createNetworkServerClass();
        }
        return this->NetworkServerClass;
    }
    // The network server should not need to be deleted - it will exist until
    // the end of the run - at least for now.

    void setNetworkClientPort(uint16_t newNetworkClientPort) {
        this->NetworkClientPort = newNetworkClientPort;
    }
    void setNetworkClientRemoteHost(std::string newNetworkClientRemoteHost) {
        this->NetworkClientRemoteHost = newNetworkClientRemoteHost;
    }
    void setNetworkClientOneshot(uint8_t newNetworkClientOneshot) {
        this->NetworkClientOneshot = newNetworkClientOneshot;
    }
    MFNNetworkClient *getNetworkClientClass() {
        if (!this->NetworkClientClass) {
            this->createNetworkClientClass();
        }
        return this->NetworkClientClass;
    }
    void destroyNetworkClientClass();

    
protected:
    // Charset class variables
    void createCharsetClass();
    uint32_t CharsetClassId;
    CHCharsetNew *CharsetClass;

    // Workunit variables
    void createWorkunitClass();
    uint32_t WorkunitClassId;
    MFNWorkunitBase *WorkunitClass;
    
    // Display classes
    void createDisplayClass();
    uint32_t DisplayClassId;
    MFNDisplay *DisplayClass;

    // Hashfile classes
    void createHashfileClass();
    uint32_t HashfileClassId;
    CHHashFileV *HashfileClass;

    // Command line data
    void createCommandlinedataClass();
    uint32_t CommandlinedataClassId;
    MFNCommandLineData *CommandlinedataClass;
    
    //CHCUDAUtils data
    void createCudaUtilsClass();
    uint32_t CudaUtilsClassId;
    CHCUDAUtils *CudaUtilsClass;

    //CHCUDAUtils data
    void createHashIdentifiersClass();
    uint32_t HashIdentifiersClassId;
    MFNHashIdentifiers *HashIdentifiersClass;
    
    void createGeneralInformationClass();
    MFNGeneralInformation *GeneralInformationClass;
    
    void createNetworkServerClass();
    MFNNetworkServer *NetworkServerClass;
    uint16_t NetworkServerPort;

    void createNetworkClientClass();
    MFNNetworkClient *NetworkClientClass;
    uint16_t NetworkClientPort;
    std::string NetworkClientRemoteHost;
    uint8_t NetworkClientOneshot;
    
};


#endif

