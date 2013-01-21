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
#ifndef __CHWORKUNITROBUST_H
#define __CHWORKUNITROBUST_H

/*
 * This is a more robust workunit class that handles resending workunits
 * as needed if a client disappears, and adds save/resume support.
 *
 * It also allows the calling application to send a blob of data
 * that will be stored along with the resume file.  This can be used
 * to store hash filenames or other similar information.
 *
 * Offsets are inclusive: Start 100, end 199: Execute [100-199]
 *
 * We also support resumes now.  The client application can store
 * arbitrary data with the resume file - hash file paths, output data, etc.
 * We just save & restore that data as a binary blob.  They can do whaver
 * they want with it.  I don't care! :)
 */

#define WU_DEBUG 0


// Define the max memory we will use.
// Right now, 100MB should be PLENTY...

#define MAX_WORKUNIT_MEMORY 100*1024*1024

#define RETURN_TOO_MANY_WORKUNITS -100

// Save every 2 minutes
#define SAVE_TIME_INTERVAL 120.0

#define RESUME_MAGIC0 'C'
#define RESUME_MAGIC1 'H'
#define RESUME_MAGIC2 'R'
#define RESUME_MAGIC3 'F'

#include <deque>
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
#include "CH_Common/Timer.h"

// This is a header for the resume data.
typedef struct CHRobustWorkunitResumeHeader {
    uint8_t  Magic0; // C
    uint8_t  Magic1; // H
    uint8_t  Magic2; // R
    uint8_t  Magic3; // F
    uint32_t BinaryBlobSize;
    uint64_t NumberWorkunitsInFile;
    uint64_t NumberWorkunitsTotal;
    uint8_t  PasswordLength;
} CHRobustWorkunitResumeHeader;

class CHWorkunitRobust : public CHWorkunitBase {
private:
    // Set to true once workunits have been created, else false.
    char WorkunitInitialized;

    // Store for unassigned workunits.  Standard queue.
    // Workunits are pushed on the back, popped from the front.
    // Uncompleted workunits are pushed back onto the rear.
    std::deque <CHWorkunitRobustElement> pendingWorkunits;

    // Store for in-flight workunits.  These need random access
    // and the ability to remove them as completed.
    std::list <CHWorkunitRobustElement> assignedWorkunits;

    // A vector of active network client IDs.
    // Used to ensure an in-use one is not returned.
    std::vector <uint16_t> inUseClientIds;

    // Mutexes for access to the workunit stores.
    // As stl containers are NOT threadsafe, we cannot have multiple threads
    // in the workunits at any given point in time.
#if USE_BOOST_THREADS
    boost::mutex workunitMutexBoost;
#else
    pthread_mutex_t workunitMutexPthreads;
    pthread_mutexattr_t workunitMutexPthreadsAttributes;
#endif

    // Mutex functions for the class
    void LockMutex();
    void UnlockMutex();

    // Total number of workunits in this instance
    uint64_t NumberOfWorkunitsTotal;

    // Number of workunits completed
    uint64_t NumberOfWorkunitsCompleted;

    // How many elements are present in each workunit (except the last).
    uint64_t ElementsPerWorkunit;

    // How many passwords have been found in this instantiation
    uint64_t TotalPasswordsFound;

    // The current password length being processed if relevant.
    uint8_t CurrentPasswordLength;

    // Number of bits per workunit, as needed
    uint8_t WorkunitBits;

    // ===== Resume File variables =====

    // Set to true if a resume file is being used.
    uint8_t UseResumeFile;

    // The filename to save to, if set.
    std::string ResumeFilename;

    // Arbitrary resume metadata
    std::vector<uint8_t> ResumeMetadata;
    
    double LastStateSaveTime;

    // Timer since start of the workunit
    Timer WorkunitTimer;

    // Verbose debug output
    uint8_t DebugOutput;

    // Write out the current state.
    // Will block for the duration.
    // forceWrite will write even if the time is not done.
    void WriteSaveState(char forceWrite);

    // Clears the internal state for re-initialization
    // Also deletes the resume file created if present.
    void ClearAllInternalState();

public:
    CHWorkunitRobust();
    ~CHWorkunitRobust();

    int CreateWorkunits(uint64_t NumberOfUnits, uint8_t BitsPerUnit, uint8_t PasswordLength);
    int LoadStateFromFile(std::string);
    void SetResumeFile(std::string newResumeFilename) {
        this->ResumeFilename = newResumeFilename;
        this->UseResumeFile = 1;
    }
    void SetResumeMetadata(std::vector<uint8_t> newResumeMetadata) {
        this->ResumeMetadata = newResumeMetadata;
    }
    std::vector<uint8_t> GetResumeMetadata() {
        return this->ResumeMetadata;
    }
    struct CHWorkunitRobustElement GetNextWorkunit(uint16_t NetworkClientId);
    void SubmitWorkunit(struct CHWorkunitRobustElement);
    void CancelWorkunit(struct CHWorkunitRobustElement);
    void CancelAllWorkunitsByClientId(uint16_t ClientId);
    uint64_t GetNumberOfWorkunits() {
        return this->NumberOfWorkunitsTotal;
    }
    uint64_t GetNumberOfCompletedWorkunits() {
        return this->NumberOfWorkunitsCompleted;
    }
    uint8_t GetWorkunitBits() {
        return this->WorkunitBits;
    }
    float GetAverageRate();
    uint16_t GetClientId();
    void FreeClientId(uint16_t);
    void PrintInternalState();
    void setNetworkClient(CHNetworkClient *newNetworkClient) { }
    void setPasswordLength(int newPasswordLength) {
        this->CurrentPasswordLength = newPasswordLength;
    }
    void EnableDebugOutput() {
        this->DebugOutput = 1;
    }

};
#endif

