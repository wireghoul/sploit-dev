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

#ifndef __CHWORKUNITBASE_H
#define __CHWORKUNITBASE_H

// Base workunit class to extend

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

class CHNetworkClient;

// Structure to store workunits and related information
// By putting 64-bit types, then 32-bit types, then smaller,
// it should be more or less aligned without much wasted space.
typedef struct CHWorkunitRobustElement {
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

    // How long (in seconds) the workunit took to complete.
    float SecondsRequiredToComplete;

    // Search rate for the workunit, else NULL
    float SearchRate;

    // How many passwords were found in the workunit
    uint32_t PasswordsFound;

    // The client ID that requested the workunit, if applicable.
    uint16_t ClientId;

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

} CHWorkunitRobustElement;


class CHWorkunitBase {
public:

    CHWorkunitBase() {
        return;
    }
    virtual ~CHWorkunitBase() {
        return;
    }
    /**
     * Creates a series of workunits based on the provided parameters.
     * 
     * This function takes the work size and desired workunit size and creates
     * the set of workunits needed to fill the workspace.
     * @param NumberOfUnits How many items need to be in the workunit space
     * @param BitsPerUnit How many items in each workunit: 2^(bits) per WU
     * @param PasswordLength The length of the password, or 0 if irrelevent.
     * @return true if successful, RETURN_TOO_MANY_WORKUNITS if too many WU
     */
    virtual int CreateWorkunits(uint64_t NumberOfUnits, uint8_t BitsPerUnit, uint8_t PasswordLength) = 0;

    /**
     * This loads the workunit state from the specified file.
     *
     * @param newResumeFilename The name of the file to attempt to restore from
     * @return true on success, false if the load fails.
     */
    virtual int LoadStateFromFile(std::string newResumeFilename) = 0;

    /**
     * Sets the filename to use for the save state file
     *
     * @param newResumeFilename The filename to save state to periodically
     */
    virtual void SetResumeFile(std::string newResumeFilename) = 0;

    /**
     * Sets the arbitrary metadata that is stored with the resume filename.
     *
     * The save state file can contain a length of arbitrary data (hashes, 
     * hash file names, output data, etc.  This is saved and restored
     * as needed and is available to the application.
     *
     * @param newResumeMetadata A uint8_t vector of data to save.
     */
    virtual void SetResumeMetadata(std::vector<uint8_t> newResumeMetadata) = 0;

    /**
     * Returns the metadata present if a resume file is loaded.
     *
     * @return A uint8_t vector of the resume metadata
     */
    virtual std::vector<uint8_t> GetResumeMetadata() = 0;

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
    virtual struct CHWorkunitRobustElement GetNextWorkunit(uint16_t ClientId) = 0;

    /**
     * Submits a fully completed workunit.  The workunit will not be re-assigned.
     *
     * @param completedWorkunit The completed workunit structure.
     */
    virtual void SubmitWorkunit(struct CHWorkunitRobustElement completedWorkunit) = 0;

    /**
     * Cancels a workunit that has not been completed.  The workunit will be reassigned.
     *
     * @param cancelledWorkunit The workunit to cancel
     */
    virtual void CancelWorkunit(struct CHWorkunitRobustElement cancelledWorkunit) = 0;

    /**
     * Cancels all workunits with a given ClientId.
     * 
     * This function will cancel and reassign all in-flight workunits by the
     * given ClientId.  This is for things like network clients disconnecting
     * to allow all work to be reassigned.
     * 
     * @param ClientId The ClientId to cancel all workunits for
     */
    virtual void CancelAllWorkunitsByClientId(uint16_t ClientId) = 0;

    /**
     * Returns the total number of workunits in the present task
     *
     * @return Total number of workunits generated in the present task.
     */
    virtual uint64_t GetNumberOfWorkunits() = 0;

    /**
     * Returns the completed number of workunits in the present task
     *
     * @return Total number of completed workunits in the present task.
     */
    virtual uint64_t GetNumberOfCompletedWorkunits() = 0;

    /**
     * Returns the number of bits per workunit in the present task
     *
     * @return Bits per workunit
     */
    virtual uint8_t GetWorkunitBits() = 0;

    /**
     * Returns the average rate of processing so far
     *
     * This returns the determined average rate based on the workunit class:
     * Number of items completed / time since creation.  It may be useful.
     *
     * @return Average rate of processing
     */
    virtual float GetAverageRate() = 0;

    /**
     * Provides a free client ID to use for requesting workunits.
     *
     * This provides a not-in-use client ID for use in requesting workunits and
     * cancelling workunits based on a client ID.
     *
     * @return A uint16_t client ID for use in requesting workunits.
     */
    virtual uint16_t GetClientId() = 0;

    /**
     * Frees a client ID that is no longer in use.
     *
     * @param ClientId The client ID to remove from the list of in-use ones.
     */
    virtual void FreeClientId(uint16_t ClientId) = 0;

    /**
     * Prints the internal state of the workunit for debugging purposes.
     */
    virtual void PrintInternalState() = 0;

    /**
     * Sets the CHNetworkClient class for use with networking if needed.
     *
     * @param newNetworkClient A pointer to the active CHNetworkClient class.
     */
    virtual void setNetworkClient(CHNetworkClient *newNetworkClient) = 0;

    /**
     * Sets the current password length - used in network client mode.
     *
     * @param newPasswordLength The current password length being worked on.
     */
    virtual void setPasswordLength(int newPasswordLength) = 0;

    /**
     * Enables debug output in the workunit class.
     */
    virtual void EnableDebugOutput() = 0;
};

#endif
