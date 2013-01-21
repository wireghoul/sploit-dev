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


// Implementation for the robust workunit type...

#include <deque>
#include <list>
#include <vector>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "MFN_Common/MFNWorkunitRobust.h"

//#define TRACE_PRINTF 1
#include "MFN_Common/MFNDebugging.h"


void PrintRobustWorkunit(struct MFNWorkunitRobustElement ElementToPrint) {
    printf("WorkUnitID: %lu\n", ElementToPrint.WorkUnitID);
    printf("StartPoint: %lu\n", ElementToPrint.StartPoint);
    printf("EndPoint: %lu\n", ElementToPrint.EndPoint);
    printf("WorkunitRequestedTimestamp: %f\n", ElementToPrint.WorkunitRequestedTimestamp);
    printf("WorkunitCompletedTimestamp: %f\n", ElementToPrint.WorkunitCompletedTimestamp);
    printf("ClientId: %u\n", ElementToPrint.ClientId);
    printf("IsAssigned: %d\n", ElementToPrint.IsAssigned);
    printf("PasswordLength: %d\n", ElementToPrint.PasswordLength);
    printf("IsValid: %d\n", ElementToPrint.IsValid);
    printf("Flags: 0x%02x\n", ElementToPrint.Flags);
    printf("Data size: %d\n", ElementToPrint.WorkunitAdditionalData.size());
}

MFNWorkunitRobust::MFNWorkunitRobust() {
    trace_printf("MFNWorkunitRobust::MFNWorkunitRobust()\n");
    // Clear out the internal state
    this->UseResumeFile = 0;
    this->ResumeFilename.clear();
    this->DebugOutput = 0;
    this->ClearAllInternalState();
}

MFNWorkunitRobust::~MFNWorkunitRobust() {
    trace_printf("MFNWorkunitRobust::~MFNWorkunitRobust()\n");
    // If the workunit is complete, delete the resume file.
    if (this->NumberOfWorkunitsCompleted == this->NumberOfWorkunitsTotal) {
        if (this->UseResumeFile) {
            if (this->DebugOutput) {
                printf("Deleting file %s\n", this->ResumeFilename.c_str());
            }
            unlink(this->ResumeFilename.c_str());
        }
    }
}

void MFNWorkunitRobust::ClearAllInternalState() {
    trace_printf("MFNWorkunitRobust::ClearAllInternalState()\n");
    // Clear all the various internal states.
    this->WorkunitClassInitialized = 0;
    this->pendingWorkunits.clear();
    this->assignedWorkunits.clear();
    this->inUseClientIds.clear();
    this->NumberOfWorkunitsTotal = 0;
    this->NumberOfWorkunitsCompleted = 0;
    this->ElementsPerWorkunit = 0;
    this->CurrentPasswordLength = 0;
    this->WorkunitBits = 0;
    this->LastStartOffsetCreated = 0;
    this->ActualWorkunitsCreated = 0;
    this->NumberPasswordsTotal = 0;
}


void MFNWorkunitRobust::CreateMorePendingWorkunits(uint32_t numberToAdd) {
    trace_printf("MFNWorkunitRobust::CreateMorePendingWorkunits(%d)\n", numberToAdd);

    MFNWorkunitRobustElement NewWorkunit;
    uint64_t StartPoint = 0;
    uint64_t EndPoint = 0;
    uint64_t WorkunitId;
    uint64_t LastWorkunitToCreate = 0;
    
    /**
     * If there are more than the max pending workunits, just return.  This
     * probably indicates a bug, but is a reasonable sanity check to prevent
     * creating too many workunits.
     */
    if (this->pendingWorkunits.size() > MFN_WORKUNIT_MAX_PENDING_WUS) {
        return;
    }

    /**
     * Create numberToAdd workunits, starting from LastStartOffsetCreated.
     * 
     * This is called from inside other functions, and DOES NOT LOCK THE MUTEX.
     */

    // Clear the workunit, then set some variables
    memset(&NewWorkunit, 0, sizeof(MFNWorkunitRobustElement));

    NewWorkunit.PasswordLength = this->CurrentPasswordLength;
    NewWorkunit.IsValid = 1;

    StartPoint = this->LastStartOffsetCreated;
    
    // Determine the last workunit ID to create at this point.
    LastWorkunitToCreate = (this->ActualWorkunitsCreated + numberToAdd);
    if (LastWorkunitToCreate > this->NumberOfWorkunitsTotal) {
        LastWorkunitToCreate = this->NumberOfWorkunitsTotal;
    }
    
    
    // For each work unit, set things up.
    for (WorkunitId = this->ActualWorkunitsCreated; 
            WorkunitId < LastWorkunitToCreate; WorkunitId++) {
        // Calculate the endpoint.
        EndPoint = StartPoint + this->ElementsPerWorkunit - 1;
        // If the endpoint of this workunit would go past the number
        // of units, set it to however many is left.
        if (EndPoint > this->NumberPasswordsTotal) {
            EndPoint = this->NumberPasswordsTotal - 1;
        }
        
        if (this->DebugOutput) {
            printf("WU %d: SP: %lu  EP: %lu\n", WorkunitId, StartPoint, EndPoint);
        }
        
        NewWorkunit.WorkUnitID = WorkunitId;
        NewWorkunit.StartPoint = StartPoint;
        NewWorkunit.EndPoint = EndPoint;

        this->pendingWorkunits.push_back(NewWorkunit);
        this->ActualWorkunitsCreated++;
        
        StartPoint += this->ElementsPerWorkunit;
    }
    
    // Store the next start point.
    this->LastStartOffsetCreated = StartPoint;
}

int MFNWorkunitRobust::CreateWorkunits(uint64_t NumberOfPasswords, uint8_t BitsPerUnit, uint8_t PasswordLength) {
    trace_printf("MFNWorkunitRobust::CreateWorkunits(%lu, %u, %u)\n", NumberOfPasswords, BitsPerUnit, PasswordLength);

    uint64_t NumberOfWorkunits = 0;
    
    this->workunitMutexBoost.lock();
    
    this->ClearAllInternalState();

    this->WorkunitBits = BitsPerUnit;
    this->CurrentPasswordLength = PasswordLength;

    this->NumberPasswordsTotal = NumberOfPasswords;
    
    // Calculate how many elements are needed per workunit
    this->ElementsPerWorkunit = pow(2.0, (int)BitsPerUnit);
    
    if (this->DebugOutput) {
        printf("Elements per unit: %llu\n", this->ElementsPerWorkunit);
    }

    // If the number of workunits fits perfectly, no need for an extra "end" unit.
    if ((NumberOfPasswords % this->ElementsPerWorkunit) == 0) {
        NumberOfWorkunits = (NumberOfPasswords / this->ElementsPerWorkunit);
    } else {
        NumberOfWorkunits = (NumberOfPasswords / this->ElementsPerWorkunit) + 1;
    }
    this->NumberOfWorkunitsTotal = NumberOfWorkunits;

    if (this->DebugOutput) {
        printf("Total number of workunits: %llu\n", NumberOfWorkunits);
        
    }
    
    // Create the initial batch of workunits
    this->CreateMorePendingWorkunits(MFN_WORKUNIT_WU_REFILL_SIZE);
    
    // At this point, we should be done with creating the workunits.
    // Do some final cleanup and go.

    this->NumberOfWorkunitsCompleted = 0;
    this->WorkunitClassInitialized = 1;

    // Start the execution timer
    this->WorkunitTimer.start();
    this->LastStateSaveTime = 0;

    this->workunitMutexBoost.unlock();

    this->WriteSaveState(1);

    return 1;
}


struct MFNWorkunitRobustElement MFNWorkunitRobust::GetNextWorkunit(uint32_t ClientId) {
    trace_printf("MFNWorkunitRobust::GetNextWorkunit(%u)\n", ClientId);

    struct MFNWorkunitRobustElement Workunit;

    this->workunitMutexBoost.lock();
    
    // Check to see if we need to make more workunits.
    if ((this->ActualWorkunitsCreated != this->NumberOfWorkunitsTotal) &&
            (this->pendingWorkunits.size() < MFN_WORKUNIT_MIN_PENDING_WUS)) {
        this->CreateMorePendingWorkunits(MFN_WORKUNIT_WU_REFILL_SIZE);
    }

    // Check to see if there are valid workunits left.
    if (this->pendingWorkunits.size() == 0) {
        // If not, return a unit with isValid = 0.
        if (this->DebugOutput) {
            printf("pendingWorkunits.size() == 0; returning.\n");
        }
        memset(&Workunit, 0, sizeof(MFNWorkunitRobustElement));
        Workunit.Flags = WORKUNIT_TERMINATE;
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
    Workunit.WorkunitRequestedTimestamp = this->WorkunitTimer.getElapsedTime();
    Workunit.ClientId = ClientId;

    // Add the workunit to the in-flight queue.
    this->assignedWorkunits.push_back(Workunit);
    if (this->DebugOutput) {
        printf("In flight WUs: %lu\n", this->assignedWorkunits.size());
    }  

    this->workunitMutexBoost.unlock();
    this->WriteSaveState(0);
    if (this->DebugOutput) {
        PrintRobustWorkunit(Workunit);
    }
    return Workunit;
}


uint32_t MFNWorkunitRobust::GetClientId() {
    trace_printf("MFNWorkunitRobust::GetClientId()\n");
    // Return an unused network ID
    uint32_t NetworkIdCandidate;
    uint32_t i;
    uint8_t NetworkIdFound;

    this->workunitMutexBoost.lock();

    while(1) {
        NetworkIdFound = 0;
        NetworkIdCandidate = (uint32_t)rand();
        for (i = 0; i < this->inUseClientIds.size(); i++) {
            if (this->inUseClientIds[i] == NetworkIdCandidate) {
                NetworkIdFound = 1;
                break;
            }
        }
        // If we haven't found the ID, it's good.  Add it & return it!
        if (!NetworkIdFound) {
            this->inUseClientIds.push_back(NetworkIdCandidate);
            this->workunitMutexBoost.unlock();
            if (this->DebugOutput) {
                printf("Returning ClientID: %d\n", NetworkIdCandidate);
            }

            return NetworkIdCandidate;
        }
    }
}

void MFNWorkunitRobust::FreeClientId(uint32_t ClientId) {
    trace_printf("MFNWorkunitRobust::FreeClientId(%d)\n", ClientId);
    
    int i;

    this->workunitMutexBoost.lock();
    for (i = 0; i < this->inUseClientIds.size(); i++) {
        if (this->inUseClientIds[i] == ClientId) {
            if (this->DebugOutput) {
                printf("Found ClientID at position %d\n", i);
            }
            // Erase the item at the given position.
            this->inUseClientIds.erase(this->inUseClientIds.begin() + i);
        }
    }
    this->workunitMutexBoost.unlock();
}

void MFNWorkunitRobust::SubmitWorkunit(struct MFNWorkunitRobustElement ReturnedWorkunit) {
    trace_printf("MFNWorkunitRobust::SubmitWorkunit(...)\n");
    this->SubmitWorkunitById(ReturnedWorkunit.WorkUnitID);
}

void MFNWorkunitRobust::SubmitWorkunitById(uint64_t completedWorkunitId) {
    trace_printf("MFNWorkunitRobust::SubmitWorkunitById(%u)\n", completedWorkunitId);
    
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
            this->NumberOfWorkunitsCompleted++;
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
    this->WriteSaveState(0);
}

void MFNWorkunitRobust::CancelWorkunit(struct MFNWorkunitRobustElement CancelledWorkunit) {
    trace_printf("MFNWorkunitRobust::CancelWorkunit(...)\n");
    this->CancelWorkunitById(CancelledWorkunit.WorkUnitID);
}

void MFNWorkunitRobust::CancelWorkunitById(uint64_t cancelledWorkunitId) {
    trace_printf("MFNWorkunitRobust::CancelWorkunitById(%u)\n", cancelledWorkunitId);
    
    std::list<MFNWorkunitRobustElement>::iterator inflightWorkunit;
    struct MFNWorkunitRobustElement CancelledWorkunitCopy;

    memset(&CancelledWorkunitCopy, 0, sizeof(struct MFNWorkunitRobustElement));

    this->workunitMutexBoost.lock();

    // Look for workunit in the list
    for (inflightWorkunit = this->assignedWorkunits.begin(); inflightWorkunit != this->assignedWorkunits.end(); inflightWorkunit++) {
        // Check for the unique Workunit ID
        if (inflightWorkunit->WorkUnitID == cancelledWorkunitId) {
            if (this->DebugOutput) {
                printf("Found inflight WU ID to cancel: %lu\n", inflightWorkunit->WorkUnitID);
            }
            // Copy the data back into a fresh workunit to put back in the queue.
            CancelledWorkunitCopy.WorkUnitID = inflightWorkunit->WorkUnitID;
            CancelledWorkunitCopy.StartPoint = inflightWorkunit->StartPoint;
            CancelledWorkunitCopy.EndPoint = inflightWorkunit->EndPoint;
            CancelledWorkunitCopy.PasswordLength = inflightWorkunit->PasswordLength;
            CancelledWorkunitCopy.IsValid = 1;
            this->pendingWorkunits.push_back(CancelledWorkunitCopy);

            this->assignedWorkunits.erase(inflightWorkunit);
            if (this->DebugOutput) {
                printf("Inflight left: %d\n", this->assignedWorkunits.size());
            }            
            // Clean out the fields in the cancelled WU

            break;
        }
    }
    this->workunitMutexBoost.unlock();
}

void MFNWorkunitRobust::CancelAllWorkunitsByClientId(uint32_t ClientId) {
    trace_printf("MFNWorkunitRobust::CancelAllWorkunitsByClientId(%d)\n", ClientId);
    
    std::list<MFNWorkunitRobustElement>::iterator inflightWorkunit;
    struct MFNWorkunitRobustElement CancelledWorkunitCopy;

    if (this->DebugOutput) {
        printf("MFNWorkunitRobust::CancelAllWorkunitsByClientId(%d)\n", ClientId);
    }

    memset(&CancelledWorkunitCopy, 0, sizeof(struct MFNWorkunitRobustElement));

    this->workunitMutexBoost.lock();

    // Look for workunit in the list
    for (inflightWorkunit = this->assignedWorkunits.begin(); inflightWorkunit != this->assignedWorkunits.end(); inflightWorkunit++) {
        // Check for the unique Workunit ID
        if (inflightWorkunit->ClientId == ClientId) {
#if WU_DEBUG
            printf("Found inflight WU ID to cancel: %lu\n", inflightWorkunit->WorkUnitID);
#endif
            // Copy the data back into a fresh workunit to put back in the queue.
            CancelledWorkunitCopy.WorkUnitID = inflightWorkunit->WorkUnitID;
            CancelledWorkunitCopy.StartPoint = inflightWorkunit->StartPoint;
            CancelledWorkunitCopy.EndPoint = inflightWorkunit->EndPoint;
            CancelledWorkunitCopy.PasswordLength = inflightWorkunit->PasswordLength;
            CancelledWorkunitCopy.IsValid = 1;
            this->pendingWorkunits.push_back(CancelledWorkunitCopy);

            this->assignedWorkunits.erase(inflightWorkunit);
#if WU_DEBUG
            printf("Inflight left: %d\n", this->assignedWorkunits.size());
#endif
            // Clean out the fields in the cancelled WU

            break;
        }
    }
    this->workunitMutexBoost.unlock();
}

void MFNWorkunitRobust::PrintInternalState() {
    trace_printf("MFNWorkunitRobust::PrintInternalState()\n");
    int i;

    printf("Number WU total: %lu\n", this->NumberOfWorkunitsTotal);
    printf("Number WU completed: %lu\n", this->NumberOfWorkunitsCompleted);
    printf("Number WU left: %lu\n", this->pendingWorkunits.size());
    printf("Number WU inflight: %lu\n", this->assignedWorkunits.size());
    printf("Active client IDs: ");
    for (i = 0 ; i < this->inUseClientIds.size(); i++) {
        printf("%d, ", this->inUseClientIds[i]);
    }
    printf("\n");
}

void MFNWorkunitRobust::WriteSaveState(char forceWrite) {
    trace_printf("MFNWorkunitRobust::WriteSaveState(%u)\n", forceWrite);
    
    if (this->DebugOutput) {
        forceWrite = 1;
    }

    struct MFNRobustWorkunitResumeHeader ResumeHeader;
    FILE *resumeFile;
    std::list<MFNWorkunitRobustElement>::iterator inflightWorkunit;
    MFNWorkunitRobustElement WorkunitElement;
    uint64_t i;

    // Only execute if a resume file is being used.
    if (!this->UseResumeFile) {
        if (this->DebugOutput) {
            printf("this->UseResumeFile not set: Returning.\n");
        }
        return;
    }

    // Check to see if we should write the save state - if not, return.
    if (!forceWrite &&
            ((this->WorkunitTimer.getElapsedTime() - this->LastStateSaveTime) < SAVE_TIME_INTERVAL)) {
        if (this->DebugOutput) {
            printf("No forcewrite, no interval timeout: Returning.\n");
        }

        return;
    }

    // Hate doing it, but we have to lock the global mutex here.
    this->workunitMutexBoost.lock();

    // Init the ResumeHeader to write
    ResumeHeader.Magic0 = RESUME_MAGIC0;
    ResumeHeader.Magic1 = RESUME_MAGIC1;
    ResumeHeader.Magic2 = RESUME_MAGIC2;
    ResumeHeader.Magic3 = RESUME_MAGIC3;
    ResumeHeader.BinaryBlobSize = this->ResumeMetadata.size();
    ResumeHeader.NumberWorkunitsInFile = this->pendingWorkunits.size() + this->assignedWorkunits.size();
    ResumeHeader.PasswordLength = this->CurrentPasswordLength;
    ResumeHeader.NumberWorkunitsTotal = this->NumberOfWorkunitsTotal;
    
    if (this->DebugOutput) {
        printf("BinaryBlobSize: %lu\n", ResumeHeader.BinaryBlobSize);
        printf("NumberWorkunitsInFile: %lu\n", ResumeHeader.NumberWorkunitsInFile);
        printf("NumberWorkunitsTotal: %lu\n", ResumeHeader.NumberWorkunitsTotal);
        printf("PasswordLength: %d\n", ResumeHeader.PasswordLength);
    }
    // Open the output file.
    resumeFile = fopen(this->ResumeFilename.c_str(), "wb");
    if (!resumeFile) {
        // Cannot open the file.  Unlock & return.
        this->workunitMutexBoost.unlock();
        return;
    }

    // If the output file is open, write out header.
    fwrite(&ResumeHeader, sizeof(struct MFNRobustWorkunitResumeHeader), 1, resumeFile);

    // Now write out the binary resume data.
    fwrite(&this->ResumeMetadata[0], 1, this->ResumeMetadata.size(), resumeFile);

    for (inflightWorkunit = this->assignedWorkunits.begin(); inflightWorkunit != this->assignedWorkunits.end(); inflightWorkunit++) {
        WorkunitElement = *inflightWorkunit;
        fwrite(&WorkunitElement, sizeof(MFNWorkunitRobustElement), 1, resumeFile);
    }

    // Finally, write out all the inflight and pending workunits.
    for (i = 0; i < this->pendingWorkunits.size(); i++) {
        fwrite(&this->pendingWorkunits[i], sizeof(MFNWorkunitRobustElement), 1, resumeFile);
    }

    fclose(resumeFile);

    this->workunitMutexBoost.unlock();
}

int MFNWorkunitRobust::LoadStateFromFile(std::string resumeFilename) {
    trace_printf("MFNWorkunitRobust::LoadStateFromFile(%s)\n", 
            resumeFilename.c_str());
    
    // Read the resume file in.
    struct MFNRobustWorkunitResumeHeader ResumeHeader;
    FILE *resumeFile;
    MFNWorkunitRobustElement WorkunitElement;
    uint64_t i;

    this->ClearAllInternalState();

    // Try to open the resume file.  If this does not work, just fail.
    resumeFile = fopen(resumeFilename.c_str(), "rb");
    if (!resumeFile) {
        if (this->DebugOutput) {
            printf("Could not open resume file!\n");
        }
        return 0;
    }

    if (!fread(&ResumeHeader, sizeof(MFNRobustWorkunitResumeHeader), 1, resumeFile)) {
        // The read failed - return 0.
        if (this->DebugOutput) {
            printf("Header read failed!\n");
        }
        return 0;
    }

    // At this point, the resume header should be populated.
    // Check magic.

    if (ResumeHeader.Magic0 != RESUME_MAGIC0) {
        return 0;
    }
    if (ResumeHeader.Magic1 != RESUME_MAGIC1) {
        return 0;
    }
    if (ResumeHeader.Magic2 != RESUME_MAGIC2) {
        return 0;
    }
    if (ResumeHeader.Magic3 != RESUME_MAGIC3) {
        return 0;
    }
    if (this->DebugOutput) {
        printf("Resume file magic checks succeeded... continuing.\n");
    }

    // Magic succeeded, we can go ahead with reading everything!

    this->ResumeMetadata.clear();
    this->ResumeMetadata.resize(ResumeHeader.BinaryBlobSize);
    if (!fread(&this->ResumeMetadata[0], ResumeHeader.BinaryBlobSize, 1, resumeFile)) {
        // The read failed - return 0.
        return 0;
    }

    if (this->DebugOutput) {
        printf("Got resume file header.\n");
        printf("Blob size: %d\n", ResumeHeader.BinaryBlobSize);
        printf("Number workunits in file: %d\n", ResumeHeader.NumberWorkunitsInFile);
        printf("Number workunits total: %d\n", ResumeHeader.NumberWorkunitsTotal);
        printf("Passlen: %d\n", ResumeHeader.PasswordLength);
    }

    for (i = 0; i < ResumeHeader.NumberWorkunitsInFile; i++) {
        if (fread(&WorkunitElement, sizeof(MFNWorkunitRobustElement), 1, resumeFile)) {
            // The size matches, so we read a full workunit.
            this->pendingWorkunits.push_back(WorkunitElement);
            if (this->DebugOutput) {
                printf("Read WU ID %lu\n", WorkunitElement.WorkUnitID);
            }
        }
    }

    // Do final init.
    this->CurrentPasswordLength = ResumeHeader.PasswordLength;
    this->NumberOfWorkunitsTotal = ResumeHeader.NumberWorkunitsTotal;
    this->NumberOfWorkunitsCompleted = ResumeHeader.NumberWorkunitsTotal - ResumeHeader.NumberWorkunitsInFile;
    this->WorkunitClassInitialized = 1;
    return 1;
}

float MFNWorkunitRobust::GetAverageRate() {
    trace_printf("MFNWorkunitRobust::GetAverageRate()\n");
    float workRate;

    workRate = (float)(this->NumberOfWorkunitsCompleted * this->ElementsPerWorkunit);
    workRate /= this->WorkunitTimer.getElapsedTime();
    return workRate;
}

void MFNWorkunitRobust::ExportWorkunitsAsProtobuf(uint32_t numberWorkunits, 
        uint32_t networkClientId, std::string *protobufData, 
        uint32_t passwordLength) {
    trace_printf("MFNWorkunitRobust::ExportWorkunitsAsProtobuf()\n");
    
    struct MFNWorkunitRobustElement Workunit;
    MFNWorkunitProtobuf_SingleWorkunit *WorkunitSingleProtobuf;
    
    
    // Check to see if we need to make more workunits.  Do this BEFORE checking
    // to see if workunits are created/need sending...
    if (this->pendingWorkunits.size() < MFN_WORKUNIT_MIN_PENDING_WUS) {
        this->CreateMorePendingWorkunits(MFN_WORKUNIT_WU_REFILL_SIZE);
    }

    // This function simply calls the existing "Get Workunit" function from
    // inside the class.  As a result, we don't need to worry about the main
    // mutex - but we do have to lock the protobuf mutex.
    
    this->WorkunitProtobufMutex.lock();
    
    this->WorkunitGroupProtobuf.Clear();

    // If the password length does not match, the client needs to reset.
    if (passwordLength != this->CurrentPasswordLength) {
        trace_printf("Password length mismatch\n");
        trace_printf("Requested %d, current is %d\n", passwordLength, this->CurrentPasswordLength);
        this->WorkunitGroupProtobuf.set_no_more_workunits(1);
    } 
    // If the pending size is zero, but there are inflight WUs, tell them to wait.
    else if (this->pendingWorkunits.size() == 0 && this->assignedWorkunits.size()) {
        trace_printf("Workunit wait\n");
        this->WorkunitGroupProtobuf.set_workunit_wait(1);
    }
    // There are workunits left - return at least some.
    else {
        trace_printf("Getting Workunits\n");
        uint32_t tenPercent = (uint32_t)(0.10 * (float)this->pendingWorkunits.size());
        uint32_t numberWorkunitsToReturn;
        char *wordlistDataPointer;
        
        // If it's less than one, make it return one WU.
        if (tenPercent == 0) {
            tenPercent = 1;
        }
        
        if (tenPercent < numberWorkunits) {
            numberWorkunitsToReturn = tenPercent;
        } else {
            numberWorkunitsToReturn = numberWorkunits;
        }
        // TODO: Fix this so it can request multiple WUs.
        numberWorkunitsToReturn = 1;
        for (uint32_t i = 0; i < numberWorkunitsToReturn; i++) {
            Workunit = this->GetNextWorkunit(networkClientId);

            // Ensure the unit is valid before packing it.
            if (Workunit.IsValid) {
                WorkunitSingleProtobuf = this->WorkunitGroupProtobuf.add_workunits();
                WorkunitSingleProtobuf->set_workunit_id(Workunit.WorkUnitID);
                WorkunitSingleProtobuf->set_start_point(Workunit.StartPoint);
                WorkunitSingleProtobuf->set_end_point(Workunit.EndPoint);
                WorkunitSingleProtobuf->set_workunit_requested_timestamp(Workunit.WorkunitRequestedTimestamp);
                WorkunitSingleProtobuf->set_password_length(Workunit.PasswordLength);
                WorkunitSingleProtobuf->set_is_valid(Workunit.IsValid);
                WorkunitSingleProtobuf->set_flags(Workunit.Flags);
                
                // Default value is 0 - no words.
                WorkunitSingleProtobuf->set_number_words_loaded(0);

                // Pack additional data if present.
                if (Workunit.WorkunitAdditionalData.size()) {
                    WorkunitSingleProtobuf->set_additional_data(
                        std::string(Workunit.WorkunitAdditionalData.begin(), 
                        Workunit.WorkunitAdditionalData.end()));
                }
                
                // Pack wordlist if present.
                if (Workunit.WordLengths.size()) {
                    WorkunitSingleProtobuf->set_wordlist_block_length(Workunit.WordBlockLength);
                    WorkunitSingleProtobuf->set_number_words_loaded(Workunit.NumberWordsLoaded);
                    WorkunitSingleProtobuf->set_wordlist_lengths(
                        std::string(Workunit.WordLengths.begin(), 
                        Workunit.WordLengths.end()));
                    // Wordlist data is a vector of uint32_t - needs to be
                    // handled specially.
                    wordlistDataPointer = (char *)&Workunit.WordlistData[0];
                    // Create a string from a byte sequence.
                    WorkunitSingleProtobuf->set_wordlist_data(
                        std::string(wordlistDataPointer, 
                        Workunit.WordlistData.size() * 4));
                }
            }
        }
    }
    
    
    
    this->WorkunitGroupProtobuf.SerializeToString(protobufData);
    this->WorkunitGroupProtobuf.Clear();
    
    this->WorkunitProtobufMutex.unlock();
}

//#define ROBUST_WU_UNIT_TEST 1

#if ROBUST_WU_UNIT_TEST

// Test with a lot of workunits to ensure refill is working properly.
void testLargeSpaces() {
    printf("MFNWorkunitRobust Unit Test for large spaces!\n");
    // How about length 13?
    uint64_t numberPasswords = (uint64_t)pow(95, 9);
    MFNWorkunitRobustElement WU;
    
    uint64_t lastEnd = 0xffffffffffffffff;
    uint32_t clientId = 0;

    MFNWorkunitRobust *WorkunitClass;
    WorkunitClass = new MFNWorkunitRobust();
    //WorkunitClass->EnableDebugOutput();
    
    printf("Creating workunits for %lu passwords.\n", numberPasswords);
    
    WorkunitClass->CreateWorkunits(numberPasswords, 38, 9);
    clientId = WorkunitClass->GetClientId();
    
    printf("Total WU: %llu\n", (unsigned long int)WorkunitClass->GetNumberOfWorkunits());
    
    // Let's get some workunits!
    while (WorkunitClass->GetNumberOfCompletedWorkunits() != WorkunitClass->GetNumberOfWorkunits()) {
        WU = WorkunitClass->GetNextWorkunit(clientId);
        
        if ((WU.WorkUnitID % 10000) == 0) {
            printf(".");
        }
        if ((WU.WorkUnitID % 500000) == 0) {
            printf(": %llu (%0.2f%%)\n", WU.WorkUnitID, 100.0 * 
                    ((float)WU.WorkUnitID / (float)WorkunitClass->GetNumberOfWorkunits()));
            printf("SP: %llu  EP: %llu\n", WU.StartPoint, WU.EndPoint);
        }
        
        lastEnd ++;
        if (WU.StartPoint != lastEnd) {
            printf("Start point mismatch in WU ID %llu\n", WU.WorkUnitID);
        }
        lastEnd = WU.EndPoint;
        WorkunitClass->SubmitWorkunit(WU);
    }
    printf("\nSP: %llu  EP: %llu\n", WU.StartPoint, WU.EndPoint);
    printf("\n\n");
}

int main() {
    printf("MFNWorkunitRobust Unit Test!\n");
    int i;
    
    testLargeSpaces();
    return 0;

    MFNWorkunitRobust *Workunit;
    MFNWorkunitRobustElement MyWorkunits[10];
    
    std::vector<uint32_t> activeClientIds;

    Workunit = new MFNWorkunitRobust();

    Workunit->EnableDebugOutput();
    
    //Workunit->LoadStateFromFile("resumeFile.chr");
    //Workunit->PrintInternalState();


    //Workunit->SetResumeFile("resumeFile.chr");

    Workunit->CreateWorkunits(16, 0, 3);
    for (i = 0 ; i < 10; i ++) {
        uint32_t clientId = Workunit->GetClientId();
        activeClientIds.push_back(clientId);
        printf("Client ID: %d\n", clientId);
        Workunit->PrintInternalState();
    }

    for (i = 0 ; i < 10; i ++) {
        printf("Removing client ID: %d\n", activeClientIds[i]);
        Workunit->FreeClientId(activeClientIds[i]);
        Workunit->PrintInternalState();
    }

    

    for (i = 0 ; i < 10; i ++) {
        MyWorkunits[i] = Workunit->GetNextWorkunit(i);
        printf("Got workunit ID %lu: %lu-%lu\n", 
            MyWorkunits[i].WorkUnitID, MyWorkunits[i].StartPoint, MyWorkunits[i].EndPoint);
    }

    Workunit->CancelWorkunit(MyWorkunits[5]);
    Workunit->CancelAllWorkunitsByClientId(3);
    
    for (i = 0 ; i < 10; i ++) {
        Workunit->SubmitWorkunit(MyWorkunits[i]);
    }

    for (i = 0 ; i < 10; i ++) {
        MyWorkunits[i] = Workunit->GetNextWorkunit(i);
        printf("Got workunit ID %lu: %lu-%lu\n",
            MyWorkunits[i].WorkUnitID, MyWorkunits[i].StartPoint, MyWorkunits[i].EndPoint);
    }
}

#endif
