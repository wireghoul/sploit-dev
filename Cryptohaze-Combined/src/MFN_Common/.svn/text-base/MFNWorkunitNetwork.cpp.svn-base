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

#include "MFN_Common/MFNWorkunitNetwork.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"

extern void PrintRobustWorkunit(struct MFNWorkunitRobustElement ElementToPrint);

extern MFNClassFactory MultiforcerGlobalClassFactory;

MFNWorkunitNetworkClient::MFNWorkunitNetworkClient() {
    this->CurrentPasswordLength = 0;
    this->DebugOutput = 0;
    this->returnWaitWorkunit = 0;
    this->returnTerminateWorkunit = 0;
}

MFNWorkunitNetworkClient::~MFNWorkunitNetworkClient() {
    
}

struct MFNWorkunitRobustElement MFNWorkunitNetworkClient::GetNextWorkunit(uint32_t NetworkClientId) {
    trace_printf("MFNWorkunitNetworkClient::GetNextWorkunit(%u)\n", NetworkClientId);

    struct MFNWorkunitRobustElement Workunit;

    this->workunitMutexBoost.lock();
    
    if (this->pendingWorkunits.size() == 0) {
        network_printf("No workunits - trying to fetch 10.\n");
        MultiforcerGlobalClassFactory.getNetworkClientClass()->
                fetchWorkunits(MFN_NETWORK_WORKUNIT_NUMBER_WUS_TO_FETCH, 
                this->CurrentPasswordLength);
        network_printf("pending size: %d\n", this->pendingWorkunits.size());
    }

    // Check to see if there are valid workunits left.
    if (this->pendingWorkunits.size() == 0) {
        // If not, return a unit with the specified flags.
        if (this->DebugOutput) {
            printf("pendingWorkunits.size() == 0; returning.\n");
        }
        memset(&Workunit, 0, sizeof(MFNWorkunitRobustElement));
        // Set the flags specified if needed.
        if (this->returnWaitWorkunit) {
            Workunit.Flags = WORKUNIT_DELAY;
        } else if (this->returnTerminateWorkunit) {
            Workunit.Flags = WORKUNIT_TERMINATE;
        }
        // Otherwise, return a null workunit, which is a term signal too.
        this->workunitMutexBoost.unlock();
        if (this->DebugOutput) {
            PrintRobustWorkunit(Workunit);
        }
        return Workunit;
    }

    // We still have workunits left.

    // Get the next waiting workunit from the main queue.
    Workunit = this->pendingWorkunits.front();
    this->pendingWorkunits.pop_front();

    if (this->DebugOutput) {
        printf("Popped WU ID %lu\n", Workunit.WorkUnitID);
    }

    // Set some variables we can make use of.
    Workunit.IsAssigned = 1;

    // Add the workunit to the in-flight queue.
    this->assignedWorkunits.push_back(Workunit);
    if (this->DebugOutput) {
        printf("In flight WUs: %lu\n", this->assignedWorkunits.size());
    }  

    this->workunitMutexBoost.unlock();
    if (this->DebugOutput) {
        PrintRobustWorkunit(Workunit);
    }
    return Workunit;
}


void MFNWorkunitNetworkClient::SubmitWorkunit(struct MFNWorkunitRobustElement workunitToSubmit) {
    this->SubmitWorkunitById(workunitToSubmit.WorkUnitID);
}

void MFNWorkunitNetworkClient::ImportWorkunitsFromProtobuf(std::string &protobufData) {
    // We've been provided a protobuf!  Import it!
    struct MFNWorkunitRobustElement Workunit;

    // This will get called from inside GetNextWorkunits, with the main mutex
    // already locked.
    this->WorkunitProtobufMutex.lock();
    
    this->WorkunitGroupProtobuf.Clear();
    this->WorkunitGroupProtobuf.ParseFromString(protobufData);
    
    // If the server has requested we wait, do so.
    if (this->WorkunitGroupProtobuf.workunit_wait()) {
        this->returnWaitWorkunit = 1;
    }
    // If the server has requested we terminate the task, do so.
    if (this->WorkunitGroupProtobuf.no_more_workunits()) {
        this->returnWaitWorkunit = 0;
        this->returnTerminateWorkunit = 1;
    }
    
    network_printf("Received %d workunits!\n", this->WorkunitGroupProtobuf.workunits_size());
    for (int i = 0; i < this->WorkunitGroupProtobuf.workunits_size(); i++) {
        memset(&Workunit, 0, sizeof(Workunit));
        this->WorkunitSingleProtobuf.Clear();
        this->WorkunitSingleProtobuf = this->WorkunitGroupProtobuf.workunits(i);
        
        Workunit.WorkUnitID = this->WorkunitSingleProtobuf.workunit_id();
        Workunit.StartPoint = this->WorkunitSingleProtobuf.start_point();
        Workunit.EndPoint = this->WorkunitSingleProtobuf.end_point();
        Workunit.IsValid = 1;
        Workunit.PasswordLength = this->WorkunitSingleProtobuf.password_length();
        Workunit.WorkunitAdditionalData = 
                std::vector<uint8_t>(this->WorkunitSingleProtobuf.additional_data().begin(), 
                this->WorkunitSingleProtobuf.additional_data().end());
        Workunit.WorkunitRequestedTimestamp = this->WorkunitSingleProtobuf.workunit_requested_timestamp();
        // If words are loaded, unpack them.
        if (this->WorkunitSingleProtobuf.number_words_loaded()) {
            Workunit.NumberWordsLoaded = this->WorkunitSingleProtobuf.number_words_loaded();
            Workunit.WordBlockLength = this->WorkunitSingleProtobuf.wordlist_block_length();
            Workunit.WordLengths = 
                std::vector<uint8_t>(this->WorkunitSingleProtobuf.wordlist_lengths().begin(), 
                this->WorkunitSingleProtobuf.wordlist_lengths().end());
            // Handle the 32-bit data specially.
            // Resize to the proper length, zero-filled.
            Workunit.WordlistData.resize(this->WorkunitSingleProtobuf.wordlist_data().length() / 4, 0);
            // Copy the data into place.
            memcpy(&Workunit.WordlistData[0], this->WorkunitSingleProtobuf.wordlist_data().c_str(),
                    this->WorkunitSingleProtobuf.wordlist_data().length());
        }
        this->pendingWorkunits.push_back(Workunit);
    }
    this->WorkunitSingleProtobuf.Clear();
    this->WorkunitGroupProtobuf.Clear();
    
    this->WorkunitProtobufMutex.unlock();
}

void MFNWorkunitNetworkClient::SubmitWorkunitById(uint64_t completedWorkunitId) {
    trace_printf("MFNWorkunitNetworkClient::SubmitWorkunitById(%u)\n", completedWorkunitId);
    
    std::list<MFNWorkunitRobustElement>::iterator inflightWorkunit;

    this->workunitMutexBoost.lock();

    // Look for workunit in the list
    for (inflightWorkunit = this->assignedWorkunits.begin(); inflightWorkunit != this->assignedWorkunits.end(); inflightWorkunit++) {
        // Check for the unique Workunit ID
        if (inflightWorkunit->WorkUnitID == completedWorkunitId) {
            if (this->DebugOutput) {
                printf("Found inflight WU ID: %lu\n", inflightWorkunit->WorkUnitID);
            }
            this->assignedWorkunits.erase(inflightWorkunit);
            if (this->DebugOutput) {
                printf("Inflight left: %d\n", this->assignedWorkunits.size());
            }
            break;
        }
    }
    if (this->DebugOutput) {
        this->PrintInternalState();
    }
    this->workunitMutexBoost.unlock();
    MultiforcerGlobalClassFactory.getNetworkClientClass()->
        submitWorkunit(completedWorkunitId);   
}

void MFNWorkunitNetworkClient::PrintInternalState() {

}
