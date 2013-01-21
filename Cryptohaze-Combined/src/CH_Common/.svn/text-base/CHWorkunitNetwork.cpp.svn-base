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


// Implementation for the network client workunit type...

#include <deque>
#include <list>
#include <vector>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "Multiforcer_Common/CHCommon.h"
#include "CH_Common/CHWorkunitNetwork.h"
#include "Multiforcer_Common/CHNetworkClient.h"

CHWorkunitNetworkClient::CHWorkunitNetworkClient() {
    // Initialize pthread mutexes if needed.
#if !USE_BOOST_THREADS
    pthread_mutexattr_init(&this->workunitMutexPthreadsAttributes);
    pthread_mutex_init(&this->workunitMutexPthreads, &this->workunitMutexPthreadsAttributes);
#endif

    this->NetworkClient = NULL;
    this->CurrentPasswordLength = 0;
}

CHWorkunitNetworkClient::~CHWorkunitNetworkClient() {
    
}

void CHWorkunitNetworkClient::setNetworkClient(CHNetworkClient* newNetworkClient) {
    this->NetworkClient = newNetworkClient;
}


void CHWorkunitNetworkClient::LockMutex() {
#if USE_BOOST_THREADS
    this->workunitMutexBoost.lock();
#else
    pthread_mutex_lock(&this->workunitMutexPthreads);
#endif
}

void CHWorkunitNetworkClient::UnlockMutex() {
#if USE_BOOST_THREADS
        this->workunitMutexBoost.unlock();
#else
        pthread_mutex_unlock(&this->workunitMutexPthreads);
#endif
}

struct CHWorkunitRobustElement CHWorkunitNetworkClient::GetNextWorkunit(uint16_t NetworkClientId) {
    CHWorkunitRobustElement WorkunitToReturn;
    
    memset(&WorkunitToReturn, 0, sizeof(CHWorkunitRobustElement));

    this->LockMutex();
    if (this->NetworkClient) {
        // Get us some general data to get the password length.
        // If it does NOT match what we think it is, return null to reset stuff.
        CHMultiforcerNetworkGeneral NetworkGeneralData;
        NetworkClient->updateGeneralInfo();
        NetworkClient->provideGeneralInfo(&NetworkGeneralData);
        if (this->CurrentPasswordLength != NetworkGeneralData.structure.password_length) {
            WorkunitToReturn.Flags |= WORKUNIT_TERMINATE;
            this->CurrentPasswordLength = NetworkGeneralData.structure.password_length;
        } else {
            WorkunitToReturn = this->NetworkClient->getNextNetworkWorkunit();
        }
    }

    this->UnlockMutex();
    return WorkunitToReturn;
}


void CHWorkunitNetworkClient::SubmitWorkunit(struct CHWorkunitRobustElement workunitToSubmit) {
    this->LockMutex();
    if (this->NetworkClient) {
        //printf("About to submit workunit to network.\n");
        this->NetworkClient->submitNetworkWorkunit(workunitToSubmit, 0);
        //printf("Back from submitting workunit to network.\n");
        // Unlock and return here.
    }
    this->UnlockMutex();
}

