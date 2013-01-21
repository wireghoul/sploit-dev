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

#ifndef __MFNWORKUNITNETWORK_H
#define __MFNWORKUNITNETWORK_H

#define WU_DEBUG 0

#include <vector>
#include <list>
#include <string>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "MFN_Common/MFNNetworkClient.h"

// We need to include the mutex defines.
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "MFN_Common/MFNWorkunitBase.h"
#include "MFN_Common/MFNWorkunit.pb.h"

/**
 * How many workunits to fetch at a time, if the server has enough.
 */
#define MFN_NETWORK_WORKUNIT_NUMBER_WUS_TO_FETCH 10

class MFNWorkunitNetworkClient : public MFNWorkunitBase {
private:
    boost::mutex workunitMutexBoost;

    int CurrentPasswordLength;
    
    uint8_t DebugOutput;

    // Storage for workunits that are waiting to be assigned.
    std::deque <MFNWorkunitRobustElement> pendingWorkunits;

    // Store for in-flight workunits.  These need random access
    // and the ability to remove them as completed.
    std::list <MFNWorkunitRobustElement> assignedWorkunits;
    
    // Protobufs for serializing and deserializing
    MFNWorkunitProtobuf WorkunitGroupProtobuf;
    MFNWorkunitProtobuf_SingleWorkunit WorkunitSingleProtobuf;
    // Mutex to protect the protobuf
    boost::mutex WorkunitProtobufMutex;
    
    // If true, requests will return a "wait" workunit.
    uint8_t returnWaitWorkunit;
    
    // If true, requests will return a "terminate" workunit.
    uint8_t returnTerminateWorkunit;


public:
    MFNWorkunitNetworkClient();
    ~MFNWorkunitNetworkClient();

    struct MFNWorkunitRobustElement GetNextWorkunit(uint32_t NetworkClientId);

    void SubmitWorkunit(struct MFNWorkunitRobustElement);
    void SubmitWorkunitById(uint64_t completedWorkunitId);
    
    void PrintInternalState();
    
    void setPasswordLength(int newPasswordLength) {
        this->CurrentPasswordLength = newPasswordLength;
        this->returnWaitWorkunit = 0;
        this->returnTerminateWorkunit = 0;
    }
    void EnableDebugOutput() {
        this->DebugOutput = 1;
    };
    
    void ImportWorkunitsFromProtobuf(std::string &protobufData);
};
#endif
