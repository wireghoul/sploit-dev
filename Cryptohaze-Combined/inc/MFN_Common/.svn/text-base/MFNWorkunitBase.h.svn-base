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

#ifndef __MFNWORKUNITBASE_H
#define __MFNWORKUNITBASE_H

/**
 * MFNWorkunit is a fork of the existing CHWorkunit class to add some new
 * features that are otherwise incompatible.  This will eventually replace
 * the old CHWorkunit class entirely.
 */

#define RETURN_TOO_MANY_WORKUNITS -100

// Workunit flags for the status reporting

// Valid workunit, work assigned, start execution.
#define WORKUNIT_VALID (1 << 0)
// Delay unit: Please wait for a period then try again.
#define WORKUNIT_DELAY (1 << 1)
// Termination unit: Please immediately terminate execution
#define WORKUNIT_TERMINATE (1 << 2)


#include <deque>
#include <vector>
#include <list>
#include <string>

class MFNNetworkClient;

/**
 * This element contains the data for the workunits.  It is passed around
 * as needed and queued as needed.  The structure should NOT contain the 
 */
typedef struct MFNWorkunitRobustElement {
    // Unique ID for the workunit: Must be non-repeating!
    uint64_t WorkUnitID;

    // Starting offset for the workunit (if needed), else NULL
    uint64_t StartPoint;

    // Ending offset for the workunit (if needed), else NULL
    // The client should execute this value in the execution.
    uint64_t EndPoint;

    // Timestamp with fractions when the workunit was requested
    double WorkunitRequestedTimestamp;

    // Timestamp with fractions when the workunit was submitted
    double WorkunitCompletedTimestamp;

    // The client ID that requested the workunit, if applicable.
    uint32_t ClientId;

    // Set to true if the workunit is assigned, else false.
    // Should duplicate the different queues. :)
    uint8_t IsAssigned;

    // The password length being processed.  Useful for verifying
    // that we haven't ended up somewhere new without realizing it.
    uint8_t PasswordLength;

    // Set to true if this is a valid workunit.
    // Set to zero indicates "out of workunits" or some other failure.
    // Avoids returning a NULL pointer to indicate failure.
    uint8_t IsValid;

    // Flags for the workunit
    uint8_t Flags;

    // Additional data for the workunit if needed - perhaps a wordlist?
    std::vector<uint8_t> WorkunitAdditionalData;

    // Wordlist data elements.
    std::vector<uint32_t> WordlistData;
    std::vector<uint8_t> WordLengths;
    // Wordlist length in 32-bit block count
    uint8_t WordBlockLength;
    // Number of passwords loaded in this WU
    uint32_t NumberWordsLoaded;
    
} MFNWorkunitRobustElement;


class MFNWorkunitBase {
public:

    MFNWorkunitBase() {
        return;
    }
    virtual ~MFNWorkunitBase() {
        return;
    }
    /**
     * Creates a series of workunits based on the provided parameters.
     * 
     * This function takes the work size and desired workunit size and creates
     * the set of workunits needed to fill the workspace.  Not all the workunits
     * may be created initially - this uses too much memory for very large
     * jobs for no good reason.  They may be created as needed.
     * 
     * @param NumberOfUnits How many items need to be in the workunit space
     * @param BitsPerUnit How many items in each workunit: 2^(bits) per WU
     * @param PasswordLength The length of the password, or 0 if irrelevent.
     * @return true if successful.
     */
    virtual int CreateWorkunits(uint64_t NumberOfUnits, uint8_t BitsPerUnit, 
        uint8_t PasswordLength) {return 0;}

    /**
     * Sets the current password length.
     * @param newPasswordLength
     */
    virtual void setPasswordLength(int newPasswordLength) { };
    
    /**
     * This loads the workunit state from the specified file.
     *
     * @param newResumeFilename The name of the file to attempt to restore from
     * @return true on success, false if the load fails.
     */
    virtual int LoadStateFromFile(std::string newResumeFilename) {return 0;}

    /**
     * Sets the filename to use for the save state file
     *
     * @param newResumeFilename The filename to save state to periodically
     */
    virtual void SetResumeFile(std::string newResumeFilename) { }

    /**
     * Sets the arbitrary metadata that is stored with the resume filename.
     *
     * The save state file can contain a length of arbitrary data (hashes, 
     * hash file names, output data, etc.  This is saved and restored
     * as needed and is available to the application.
     *
     * @param newResumeMetadata A uint8_t vector of data to save.
     */
    virtual void SetResumeMetadata(std::vector<uint8_t> newResumeMetadata) { }

    /**
     * Returns the metadata present if a resume file is loaded.
     *
     * @return A uint8_t vector of the resume metadata
     */
    virtual std::vector<uint8_t> GetResumeMetadata() {return std::vector<uint8_t>();}

    /**
     * Get the next workunit.  The relevant client ID is passed in.
     *
     * This function returns the next workunit to execute.  It is threadsafe.
     * The client ID should be passed in to allow for easy cancelling of all
     * workunits related to a client ID.
     *
     * @param ClientId The client thread ID of the calling thread.
     * @return A CHRobustWorkunitElement structure of the next workunit.
     */
    virtual struct MFNWorkunitRobustElement GetNextWorkunit(uint32_t ClientId) = 0;

    /**
     * Submits a fully completed workunit.  The workunit will not be re-assigned.
     *
     * @param completedWorkunit The completed workunit structure.
     */
    virtual void SubmitWorkunit(struct MFNWorkunitRobustElement completedWorkunit) = 0;
    
    /**
     * Submits a completed workunit.  This just takes the workunit ID, because in
     * most cases, we don't actually care about the other stuff or it is
     * created by the workunit class.
     * 
     * @param completedWorkunitId ID field from the workunit that was finished.
     */
    virtual void SubmitWorkunitById(uint64_t completedWorkunitId) = 0;

    /**
     * Cancels a workunit that has not been completed.  The workunit will be reassigned.
     *
     * @param cancelledWorkunit The workunit to cancel
     */
    virtual void CancelWorkunit(struct MFNWorkunitRobustElement cancelledWorkunit) { }
    virtual void CancelWorkunitById(uint64_t cancelledWorkunitId) { }

    /**
     * Cancels all workunits with a given ClientId.
     * 
     * This function will cancel and reassign all in-flight workunits by the
     * given ClientId.  This is for things like network clients disconnecting
     * to allow all work to be reassigned.
     * 
     * @param ClientId The ClientId to cancel all workunits for
     */
    virtual void CancelAllWorkunitsByClientId(uint32_t ClientId) { }

    /**
     * Returns the total number of workunits in the present task
     *
     * @return Total number of workunits generated in the present task.
     */
    virtual uint64_t GetNumberOfWorkunits() {return 0;}

    /**
     * Returns the completed number of workunits in the present task
     *
     * @return Total number of completed workunits in the present task.
     */
    virtual uint64_t GetNumberOfCompletedWorkunits() {return 0;}

    /**
     * Returns the number of bits per workunit in the present task
     *
     * @return Bits per workunit
     */
    virtual uint8_t GetWorkunitBits() {return 0;}

    /**
     * Returns the average rate of processing so far
     *
     * This returns the determined average rate based on the workunit class:
     * Number of items completed / time since creation.  It may be useful.
     *
     * @return Average rate of processing
     */
    virtual float GetAverageRate() {return 0;}
    
    /**
     * Provides a free client ID to use for requesting workunits.
     *
     * This provides a not-in-use client ID for use in requesting workunits and
     * cancelling workunits based on a client ID.
     *
     * @return A uint32_t client ID for use in requesting workunits.
     */
    virtual uint32_t GetClientId() {return 0;}

    /**
     * Frees a client ID that is no longer in use.
     *
     * @param ClientId The client ID to remove from the list of in-use ones.
     */
    virtual void FreeClientId(uint32_t ClientId) { }

    /**
     * Prints the internal state of the workunit for debugging purposes.
     */
    virtual void PrintInternalState() { }

    /**
     * Enables debug output in the workunit class.
     */
    virtual void EnableDebugOutput() { }
    
    /**
     * Export the specified number of workunits (or as many as remain) as a
     * protobuf string.
     * 
     * This function is used by the network stack to export workunits to a
     * remote host, and is probably only implemented on the "server" side class.
     * It does the same thing as getting workunits locally (puts them in the 
     * in-flight queue), and the remote side must either cancel them or complete
     * them.  If there are fewer workunits available than requested, it will
     * just provide what is available.
     * 
     * @param numberWorkunits How many workunits to request at once.
     * @param networkClientId The client ID retrieving the units.
     * @param protobufData A string to contain the packed protobuf.
     */
    virtual void ExportWorkunitsAsProtobuf(uint32_t numberWorkunits, 
        uint32_t networkClientId, std::string *protobufData, uint32_t passwordLength) {
    }
    
    /**
     * Import workunits from a protobuf.
     * 
     * This function is used to import pending workunits from a protobuf.  The
     * protobuf of workunits is passed in, and it populates its internal queue
     * with them.  These are then handed out through the normal method to the
     * threads.
     * 
     * @param protobufData
     */
    virtual void ImportWorkunitsFromProtobuf(std::string &protobufData) {
    }
    
    /**
     * Prints details about a workunit.  This is a utility function.
     */
    virtual void PrintWorkunit(struct MFNWorkunitRobustElement &WUToPrint) {
        printf("WorkUnitID: %lu\n", WUToPrint.WorkUnitID);
        printf("StartPoint: %lu\n", WUToPrint.StartPoint);
        printf("EndPoint: %lu\n", WUToPrint.EndPoint);
        printf("WorkunitRequestedTimestamp: %f\n", WUToPrint.WorkunitRequestedTimestamp);
        printf("WorkunitCompletedTimestamp: %f\n", WUToPrint.WorkunitCompletedTimestamp);
        printf("ClientId: %u\n", WUToPrint.ClientId);
        printf("IsAssigned: %d\n", WUToPrint.IsAssigned);
        printf("PasswordLength: %d\n", WUToPrint.PasswordLength);
        printf("IsValid: %d\n", WUToPrint.IsValid);
        printf("Flags: 0x%02x\n", WUToPrint.Flags);
        printf("Data size: %d\n", WUToPrint.WorkunitAdditionalData.size());
    }

    /**
     * Clears out a workunit completely.  Useful for initialization.
     * 
     * @param ElementToPrint The workunit to clear.
     */
    virtual void ClearWorkunit(struct MFNWorkunitRobustElement &WUToClear) {
        WUToClear.WorkUnitID = 0;
        WUToClear.StartPoint = 0;
        WUToClear.EndPoint = 0;
        WUToClear.WorkunitRequestedTimestamp = 0;
        WUToClear.WorkunitCompletedTimestamp = 0;
        WUToClear.ClientId = 0;
        WUToClear.IsAssigned = 0;
        WUToClear.PasswordLength = 0;
        WUToClear.IsValid = 0;
        WUToClear.Flags = 0;
        WUToClear.WorkunitAdditionalData.clear();
        WUToClear.WordlistData.clear();
        WUToClear.WordLengths.clear();;
        WUToClear.WordBlockLength = 0;
        WUToClear.NumberWordsLoaded = 0;
    }

};

#endif
