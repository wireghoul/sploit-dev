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

// Include file for the CHWorkunit class to handle workunit distribution.

#ifndef _CHWORKUNIT_H
#define _CHWORKUNIT_H

#include "Multiforcer_Common/CHCommon.h"

// Predefine for network class stuff.
class CHWorkunit;


#define WORKUNIT_SAVE_INTERVAL 120

// Define all ones in the workunit ID to tell clients to please hold for a bit.
#define WORKUNIT_PLZ_HOLD (CHWorkunitElement *)0xFFFFFFFFFFFFFFFF

// Stale age of a workunit - 60 seconds.
#define WORKUNIT_STALE_AGE 60


typedef struct CHWorkunitElement {
    uint64_t WorkUnitID;     // Workunit unique ID
    uint64_t StartPoint;     // Starting offset ID
    uint64_t EndPoint;       // Ending offset ID
    float    SecondsToFinish;// How many seconds of work it took
    float    NetworkRateSubmit; // Submit the remote network client rate
    uint32_t TimeRequested;  // Seconds since start it was requested
    uint32_t TimeReturned;   // Seconds since start it was returned
    uint32_t PasswordsFound; // How many passwords were found in this workunit
} CHWorkunitElement;


#if USE_NETWORK
#include "Multiforcer_Common/CHNetworkClient.h"
#endif

class CHWorkunit {
private:
    char WorkunitInitialized;
    struct CHWorkunitElement *CHWorkunits;
    uint64_t LastAllocatedWorkunitId;
    uint64_t ExecutionStartTimeMs; // Execution start in MS
    float CurrentRuntime; // Current runtime in seconds
    uint64_t NumberOfWorkunits;
    uint64_t NumberOfWorkunitsCompleted;
    uint64_t ElementsPerWorkunit;
    uint64_t TotalPasswordsFound;

#if USE_BOOST_THREADS
    boost::mutex mutex1Boost;
#else
    pthread_mutex_t mutex1;  // Mutex for important things.
    pthread_mutexattr_t mutex1attr;
#endif

#if USE_NETWORK
    CHNetworkClient *NetworkClient;
    int currentPasswordLength;
#endif

    //timer values
    struct timeval start, end;

    // Number of bits per workunit, as needed
    int WorkunitBits;

    char ResumeFilename[1024];
    FILE *ResumeSaveState;
    time_t LastStateSave;

    // Write out the current state.
    void WriteSaveState();

public:
    CHWorkunit();
    ~CHWorkunit();
    // Create workunits: N workunits to fill NumberOfUnits, with BitsPerUnit as the size of each unit
    int CreateWorkunits(uint64_t NumberOfUnits, unsigned char BitsPerUnit);
    // Load the current state from a resume file
    int LoadStateFromFile(const char *);
    struct CHWorkunitElement* GetNextWorkunit();
    int SubmitWorkunit(struct CHWorkunitElement* Workunit, uint32_t FoundPasswords);
    void PrintStatusLine();
    uint64_t GetNumberOfWorkunits();
    uint64_t GetNumberOfCompletedWorkunits();
    float GetAverageRate();
    int GetWorkunitBits();

    // If network is being used, add a function to add network support for all
    // tasks.
#if USE_NETWORK
    void setNetworkClient(CHNetworkClient *);
    void setPasswordLength(int);
#endif

};


#endif
