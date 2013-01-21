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
#include "MFN_Common/MFNHashClassLauncher.h"

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_MD5.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_MD5.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_MD5WL.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_SMD5.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_DupMD5.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_DoubleMD5.h"
#include "MFN_CPU_host/MFNHashTypePlainCPU_MD5.h"
#include "MFN_CUDA_host/MFNHashTypePlainCUDA_DoubleMD5.h"

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_NTLM.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_NTLM.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_NTLMWL.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_SNTLM.h"
#include "MFN_CPU_host/MFNHashTypePlainCPU_NTLM.h"

#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_SHA1.h"
#include "MFN_CUDA_host/MFNHashTypePlainCUDA_SHA1.h"

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_SHA256.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_SHA256.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_DoubleSHA256.h"

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_LM.h"

#include "MFN_CUDA_host/MFNHashTypeSaltedCUDA_MD5_PS.h"
#include "MFN_OpenCL_host/MFNHashTypeSaltedOpenCL_MD5_PS.h"

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_16HEX.h"

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_DupMD5.h"
#include "MFN_CUDA_host/MFNHashTypePlainCUDA_DupNTLM.h"

#include "MFN_CUDA_host/MFNHashTypePlainCUDA_LOTUS.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_LOTUS.h"

#include "MFN_OpenCL_host/MFNHashTypeSaltedOpenCL_IPB.h"
#include "MFN_OpenCL_host/MFNHashTypeSaltedOpenCL_IPBWL.h"

#include "MFN_OpenCL_host/MFNHashTypeSaltedOpenCL_Phpass.h"
#include "MFN_OpenCL_host/MFNHashTypeSaltedOpenCL_PhpassWL.h"

#include "MFN_CUDA_host/MFNHashTypeSaltedCUDA_IPBWL.h"

#include "MFN_Common/MFNHashIdentifiers.h"
#include "MFN_Common/MFNDebugging.h"

extern "C" {
    void *MFNHashClassLaunchTrampoline(void *);
}

void *MFNHashClassLaunchTrampoline(void * pointer) {
    struct ClassLauncherData *data;

    data = (ClassLauncherData *) pointer;

    mt_printf("IN THREAD %d\n", data->threadID);
    //data->HashTypeClass->crackPasswordLength(data->passwordLength);
    mt_printf("Thread %d Back from crackPasswordLength\n", data->threadID);
    fflush(stdout);
    return NULL;
}

MFNHashClassLauncher::MFNHashClassLauncher() {
    this->ClassVector.clear();
    this->ThreadObjects.clear();
    this->HashType = MFN_HASHTYPE_UNDEFINED;
}


MFNHashClassLauncher::~MFNHashClassLauncher() {
//    // Delete all the classes so they clean up properly.
//    for (int i = 0; i < this->ClassVector.size(); i++) {
//        delete this->ClassVector[i];
//        this->ClassVector[i] = NULL;
//    }
    this->ClassVector.clear();
    this->ThreadObjects.clear();
}

int MFNHashClassLauncher::setHashType(uint32_t newHashType) {
    this->HashType = newHashType;
    return true;
}

bool MFNHashClassLauncher::addCPUThreads(uint16_t numberCPUThreads) {
    MFNHashType *newCPUThread = NULL;
    
    switch(this->HashType) {
        case MFN_HASHTYPE_UNDEFINED:
            newCPUThread = NULL;
            break;
        case MFN_HASHTYPE_PLAIN_MD5:
            newCPUThread = new MFNHashTypePlainCPU_MD5();
            break;
        case MFN_HASHTYPE_NTLM:
            newCPUThread = new MFNHashTypePlainCPU_NTLM();
            break;
        default:
            newCPUThread = NULL;
            break;
    }
    
    // If the thread is created, add it and return true.
    if (newCPUThread != NULL) {
        if (newCPUThread->setCPUThreads(numberCPUThreads)) {
            this->ClassVector.push_back(newCPUThread);
            return true;
        }
        // If not successful, don't leak memory!
        delete newCPUThread;
    }
    return false;
}

bool MFNHashClassLauncher::addCUDAThread(uint16_t newCudaDeviceId) {
    MFNHashType *newCUDAThread = NULL;
    
    mt_printf("Trying to add CUDA thread for hashId %08x\n", this->HashType);
    switch(this->HashType) {
        case MFN_HASHTYPE_UNDEFINED:
            newCUDAThread = NULL;
            break;
        case MFN_HASHTYPE_PLAIN_MD5:
            newCUDAThread = new MFNHashTypePlainCUDA_MD5();
            break;
        case MFN_HASHTYPE_SHA1:
            newCUDAThread = new MFNHashTypePlainCUDA_SHA1();
            break;
        case MFN_HASHTYPE_SHA256:
            newCUDAThread = new MFNHashTypePlainCUDA_SHA256();
            break;
        case MFN_HASHTYPE_NTLM:
            newCUDAThread = new MFNHashTypePlainCUDA_NTLM();
            break;
        case MFN_HASHTYPE_LM:
            newCUDAThread = new MFNHashTypePlainCUDA_LM();
            break;
        case MFN_HASHTYPE_DOUBLE_MD5:
            newCUDAThread = new MFNHashTypePlainCUDA_DoubleMD5();
            break;
        case MFN_HASHTYPE_MD5_PS:
            newCUDAThread = new MFNHashTypeSaltedCUDA_MD5_PS();
            break;
        case MFN_HASHTYPE_16HEX:
            newCUDAThread = new MFNHashTypePlainCUDA_16HEX();
            break;
        case MFN_HASHTYPE_DUPLICATED_MD5:
            newCUDAThread = new MFNHashTypePlainCUDA_DupMD5();
            break;
        case MFN_HASHTYPE_PLAIN_LOTUS:
            newCUDAThread = new MFNHashTypePlainCUDA_LOTUS();
            break;
        case MFN_HASHTYPE_IPBWL:
            newCUDAThread = new MFNHashTypeSaltedCUDA_IPBWL();
            break;
        default:
            newCUDAThread = NULL;
            break;
    }
    
    // If the thread is created, add it and return true.
    if (newCUDAThread != NULL) {
        if (newCUDAThread->setCUDADeviceID(newCudaDeviceId)) {
            this->ClassVector.push_back(newCUDAThread);
            return true;
        }
        // If not successful, don't leak memory!
        delete newCUDAThread;
    }
    return false;
}

bool MFNHashClassLauncher::addOpenCLThread(uint16_t newOpenCLPlatform, uint16_t newOpenCLDevice) {
    MFNHashType *newOpenCLThread = NULL;
    
    
    switch(this->HashType) {
        case MFN_HASHTYPE_UNDEFINED:
            newOpenCLThread = NULL;
            break;
        case MFN_HASHTYPE_PLAIN_MD5:
            newOpenCLThread = new MFNHashTypePlainOpenCL_MD5();
            break;
        case MFN_HASHTYPE_PLAIN_MD5WL:
            newOpenCLThread = new MFNHashTypePlainOpenCL_MD5WL();
            break;
        case MFN_HASHTYPE_NTLM:
            newOpenCLThread = new MFNHashTypePlainOpenCL_NTLM();
            break;
        case MFN_HASHTYPE_NTLM_SINGLE:
            newOpenCLThread = new MFNHashTypePlainOpenCL_SNTLM();
            break;
        case MFN_HASHTYPE_PLAIN_NTLMWL:
            newOpenCLThread = new MFNHashTypePlainOpenCL_NTLMWL();
            break;
        case MFN_HASHTYPE_SHA1:
            newOpenCLThread = new MFNHashTypePlainOpenCL_SHA1();
            break;
        case MFN_HASHTYPE_SHA256:
            newOpenCLThread = new MFNHashTypePlainOpenCL_SHA256();
            break;
        case MFN_HASHTYPE_DOUBLE_SHA256:
            newOpenCLThread = new MFNHashTypePlainOpenCL_DoubleSHA256();
            break;
        case MFN_HASHTYPE_MD5_PS:
            newOpenCLThread = new MFNHashTypeSaltedOpenCL_MD5_PS();
            break;
        case MFN_HASHTYPE_DUPLICATED_MD5:
            newOpenCLThread = new MFNHashTypePlainOpenCL_DupMD5();
            break;
        case MFN_HASHTYPE_DOUBLE_MD5:
            newOpenCLThread = new MFNHashTypePlainOpenCL_DoubleMD5();
            break;
        case MFN_HASHTYPE_PLAIN_MD5_SINGLE:
            newOpenCLThread = new MFNHashTypePlainOpenCL_SMD5();
            break;
        case MFN_HASHTYPE_PLAIN_LOTUS:
            newOpenCLThread = new MFNHashTypePlainOpenCL_LOTUS();
            break;
        case MFN_HASHTYPE_IPB:
            newOpenCLThread = new MFNHashTypeSaltedOpenCL_IPB();
            break;
        case MFN_HASHTYPE_IPBWL:
            newOpenCLThread = new MFNHashTypeSaltedOpenCL_IPBWL();
            break;
        case MFN_HASHTYPE_PHPASS:
            newOpenCLThread = new MFNHashTypeSaltedOpenCL_Phpass();
            break;
        case MFN_HASHTYPE_PHPASSWL:
            newOpenCLThread = new MFNHashTypeSaltedOpenCL_PhpassWL();
            break;
        default:
            newOpenCLThread = NULL;
            break;
    }
    
    // If the thread is created, add it and return true.
    if (newOpenCLThread != NULL) {
        if (newOpenCLThread->setOpenCLDeviceID(newOpenCLPlatform, newOpenCLDevice)) {
            this->ClassVector.push_back(newOpenCLThread);
            return true;
        }
        // If not successful, don't leak memory!
        delete newOpenCLThread;
    }
    return false;
}

MFNHashType *MFNHashClassLauncher::getClassById(uint16_t classId) {
    if (classId < this->ClassVector.size()) {
        return this->ClassVector.at(classId);
    } else {
        return NULL;
    }
}

bool MFNHashClassLauncher::launchThreads(uint16_t passwordLength) {
    int i;
    ClassLauncherData data;

    // Clear out the thread object of any old threads.
    this->ThreadObjects.clear();

    data.HashTypeClass = this->ClassVector.at(0);
    data.threadID = 0;
    data.passwordLength = passwordLength;
    
    for (i = 0; i < this->ClassVector.size(); i++) {
        mt_printf("MFNHashClassLauncher launching thread %d\n", i);
        fflush(stdout);
        this->ThreadObjects.push_back(
                new boost::thread(&MFNHashType::crackPasswordLength, this->ClassVector[i], passwordLength));
                //new boost::thread(&MFNHashClassLaunchTrampoline, &data));
    }
    // Wait for threads.
    for (i = 0; i < this->ThreadObjects.size(); i++) {
        mt_printf("MFNHashClassLauncher trying to join thread %d\n", i);
        fflush(stdout);
        if (this->ThreadObjects[i]->joinable()) {
            mt_printf("Thread %d is joinable\n", i);
            this->ThreadObjects[i]->join();
            mt_printf("Thread %d is joined\n", i);
        }
    }
    return 1;
}

bool MFNHashClassLauncher::addAllDevices(std::vector<MFNDeviceInformation> allDevices) {
    // Iterate through the vector, adding all the specified devices.
    std::vector<MFNDeviceInformation>::iterator currentDevice;
    
    for (currentDevice = allDevices.begin(); currentDevice < allDevices.end(); currentDevice++) {
        if (currentDevice->IsCUDADevice) {
            mt_printf("MFNHashClassLauncher adding CUDA device %d\n", currentDevice->GPUDeviceId);
            // If the add is not successful, return false.
            if (!this->addCUDAThread(currentDevice->GPUDeviceId)) {
                return false;
            }
        } else if (currentDevice->IsOpenCLDevice) {
            if (!this->addOpenCLThread(currentDevice->OpenCLPlatformId, currentDevice->GPUDeviceId)) {
                return false;
            }
        } else if (currentDevice->IsCPUDevice) {
            if (!this->addCPUThreads(currentDevice->DeviceThreads)) {
                return false;
            }
        }
    }
    return true;
}
