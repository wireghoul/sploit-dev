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

/*
 * This is a class for the network client.  It simply takes the various
 * workunit calls, translates them to the network, and returns the results.
 * This lets me get away with having one set of code to deal with both
 * a normal system and a network system.
 */

#ifndef __CHWORKUNITNETWORK_H
#define __CHWORKUNITNETWORK_H

#define WU_DEBUG 0

#include <vector>
#include <list>
#include <string>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "Multiforcer_Common/CHCommon.h"
#include "Multiforcer_Common/CHNetworkClient.h"

// We need to include the mutex defines.
#if USE_BOOST_THREADS
        #include <boost/thread.hpp>
        #include <boost/thread/mutex.hpp>
#else
        #include <pthread.h>
#endif

#include "CH_Common/CHWorkunitBase.h"

class CHWorkunitNetworkClient : public CHWorkunitBase {
private:
    CHNetworkClient *NetworkClient;
    
#if USE_BOOST_THREADS
    boost::mutex workunitMutexBoost;
#else
    pthread_mutex_t workunitMutexPthreads;
    pthread_mutexattr_t workunitMutexPthreadsAttributes;
#endif

    int CurrentPasswordLength;

    void LockMutex();
    void UnlockMutex();

public:
    CHWorkunitNetworkClient();
    ~CHWorkunitNetworkClient();

    // Default returns for most of these.
    int CreateWorkunits(uint64_t, uint8_t, uint8_t) {return 0;}
    int LoadStateFromFile(std::string) {return 0;}
    void SetResumeFile(std::string) { }
    void SetResumeMetadata(std::vector<uint8_t>) { }
    std::vector<uint8_t> GetResumeMetadata() {
        std::vector<uint8_t> returnValue;
        return returnValue;
    }
    struct CHWorkunitRobustElement GetNextWorkunit(uint16_t NetworkClientId);

    void SubmitWorkunit(struct CHWorkunitRobustElement);
    void CancelWorkunit(struct CHWorkunitRobustElement) { }
    void CancelAllWorkunitsByClientId(uint16_t ClientId) { }
    uint64_t GetNumberOfWorkunits() {return 0;}
    uint64_t GetNumberOfCompletedWorkunits() {return 0;}
    uint8_t GetWorkunitBits() {return 0;}
    float GetAverageRate() {return 0;}
    uint16_t GetClientId() {return 0;}
    void FreeClientId(uint16_t) { }
    void PrintInternalState() { }
    void setNetworkClient(CHNetworkClient *newNetworkClient);
    void setPasswordLength(int newPasswordLength) {
        this->CurrentPasswordLength = newPasswordLength;
    }
    void EnableDebugOutput() { };
};
#endif
