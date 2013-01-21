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
#ifndef __MFNWORKUNITROBUST_H
#define __MFNWORKUNITROBUST_H

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

/**
 * The maximum number of workunits to be generated - anything beyond this will
 * be generated on the fly later.
 */
#define MFN_WORKUNIT_MAX_PENDING_WUS 1000
#define MFN_WORKUNIT_MIN_PENDING_WUS 10
#define MFN_WORKUNIT_WU_REFILL_SIZE 500

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


#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "MFN_Common/MFNWorkunitBase.h"
#include "CH_Common/CHHiresTimer.h"
#include "MFN_Common/MFNWorkunit.pb.h"

// This is a header for the resume data.
typedef struct MFNRobustWorkunitResumeHeader {
    uint8_t  Magic0; // C
    uint8_t  Magic1; // H
    uint8_t  Magic2; // R
    uint8_t  Magic3; // F
    uint32_t BinaryBlobSize;
    uint64_t NumberWorkunitsInFile;
    uint64_t NumberWorkunitsTotal;
    uint8_t  PasswordLength;
} MFNRobustWorkunitResumeHeader;

class MFNWorkunitRobust : public MFNWorkunitBase {
protected:
    /**
     * Sanity check - set to 1 if the WU class is initialized sanely, otherwise
     * set to 0.
     */
    char WorkunitClassInitialized;

    /**
     * Store for unassigned workunits.  Standard queue.
     * Workunits are pushed on the back, popped from the front.
     * Uncompleted workunits are pushed back onto the rear.
     */
    std::deque <MFNWorkunitRobustElement> pendingWorkunits;

    /**
     * Store for in-flight workunits.  These need random access
     * and the ability to remove them as completed, so a list is used.
     */
    std::list <MFNWorkunitRobustElement> assignedWorkunits;

    /**
     * A vector of active network client IDs.
     * Used to ensure an in-use one is not returned.
     * This could likely be more efficient and/or sorted.
     */
    std::vector <uint32_t> inUseClientIds;

    /**
     * Mutex for access to the workunit STL types.  STL containers are not
     * threadsafe, so we must enforce one thread in a container at a time.
     */
    boost::mutex workunitMutexBoost;

    // Total number of workunits in this instance
    uint64_t NumberOfWorkunitsTotal;

    // Number of workunits completed
    uint64_t NumberOfWorkunitsCompleted;

    // How many elements are present in each workunit (except the last).
    uint64_t ElementsPerWorkunit;
    
    // The last start offset created if not all workunits are created at once.
    uint64_t LastStartOffsetCreated;
    
    // How many workunits have actually been created - if this is not equal to
    // NumberOfWorkunitsTotal, more will have to be created at some point.
    uint64_t ActualWorkunitsCreated;
    
    // Total number of passwords for this set of workunits.
    uint64_t NumberPasswordsTotal;
    
    

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
    CHHiresTimer WorkunitTimer;

    // Verbose debug output
    uint8_t DebugOutput;
    
    // Protobufs for serializing and deserializing
    MFNWorkunitProtobuf WorkunitGroupProtobuf;
    
    boost::mutex WorkunitProtobufMutex;


    // Write out the current state.
    // Will block for the duration.
    // forceWrite will write even if the time is not done.
    void WriteSaveState(char forceWrite);

    // Clears the internal state for re-initialization
    // Also deletes the resume file created if present.
    void ClearAllInternalState();
    
    /**
     * Create more workunits.  This appends more pending workunits to the 
     * queue.  It will not create units "past the end," but will create
     * up to the number requested if needed.  DOES NOT LOCK MUTEX!
     * 
     * @parm numberToAdd How many more to create.
     */
    virtual void CreateMorePendingWorkunits(uint32_t numberToAdd);

public:
    MFNWorkunitRobust();
    ~MFNWorkunitRobust();

    /**
     * Creates workunits for later use.  This will either create all the
     * workunits, or create a number and store the creation point for later use.
     * 
     * @param NumberOfPasswords How many passwords are in this task
     * @param BitsPerUnit Workunit size in bits (2^BitsPerUnit)
     * @param PasswordLength Password length in characters
     * @return true if successful.
     */
    virtual int CreateWorkunits(uint64_t NumberOfPasswords, uint8_t BitsPerUnit,
        uint8_t PasswordLength);
    
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

    virtual struct MFNWorkunitRobustElement GetNextWorkunit(uint32_t NetworkClientId);
    
    void SubmitWorkunit(struct MFNWorkunitRobustElement);
    void SubmitWorkunitById(uint64_t completedWorkunitId);
    
    void CancelWorkunit(struct MFNWorkunitRobustElement);
    void CancelWorkunitById(uint64_t cancelledWorkunitId);
    
    void CancelAllWorkunitsByClientId(uint32_t ClientId);
    
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
    
    uint32_t GetClientId();
    
    void FreeClientId(uint32_t);
    
    void PrintInternalState();
    
    void EnableDebugOutput() {
        this->DebugOutput = 1;
    }
    
    void ExportWorkunitsAsProtobuf(uint32_t numberWorkunits, 
        uint32_t networkClientId, std::string *protobufData, 
        uint32_t passwordLength);

};
#endif

