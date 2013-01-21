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

#include "Multiforcer_Common/CHCommon.h"
#include "CH_Common/CHWorkunitRobust.h"


void PrintRobustWorkunit(struct CHWorkunitRobustElement ElementToPrint) {
    printf("WorkUnitID: %lu\n", ElementToPrint.WorkUnitID);
    printf("StartPoint: %lu\n", ElementToPrint.StartPoint);
    printf("EndPoint: %lu\n", ElementToPrint.EndPoint);
    printf("WorkunitRequestedTimestamp: %f\n", ElementToPrint.WorkunitRequestedTimestamp);
    printf("WorkunitCompletedTimestamp: %f\n", ElementToPrint.WorkunitCompletedTimestamp);
    printf("SecondsRequiredToComplete: %f\n", ElementToPrint.SecondsRequiredToComplete);
    printf("SearchRate: %f\n", ElementToPrint.SearchRate);
    printf("PasswordsFound: %u\n", ElementToPrint.PasswordsFound);
    printf("ClientId: %u\n", ElementToPrint.ClientId);
    printf("IsAssigned: %d\n", ElementToPrint.IsAssigned);
    printf("PasswordLength: %d\n", ElementToPrint.PasswordLength);
    printf("IsValid: %d\n", ElementToPrint.IsValid);
    printf("Flags: 0x%02x\n", ElementToPrint.Flags);
}

CHWorkunitRobust::CHWorkunitRobust() {
    // Initialize pthread mutexes if needed.
#if !USE_BOOST_THREADS
    pthread_mutexattr_init(&this->workunitMutexPthreadsAttributes);
    pthread_mutex_init(&this->workunitMutexPthreads, &this->workunitMutexPthreadsAttributes);
#endif
    
    // Clear out the internal state
    this->UseResumeFile = 0;
    this->ResumeFilename.clear();
    this->DebugOutput = 0;
    this->ClearAllInternalState();
}

CHWorkunitRobust::~CHWorkunitRobust() {
    if (this->DebugOutput) {
        printf("CHWorkunitRobust::~CHWorkunitRobust()\n");
    }
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

void CHWorkunitRobust::ClearAllInternalState() {
    if (this->DebugOutput) {
        printf("CHWorkunitRobust::ClearAllInternalState()\n");
    }
    // Clear all the various internal states.
    this->WorkunitInitialized = 0;
    this->pendingWorkunits.clear();
    this->assignedWorkunits.clear();
    this->inUseClientIds.clear();
    this->NumberOfWorkunitsTotal = 0;
    this->NumberOfWorkunitsCompleted = 0;
    this->ElementsPerWorkunit = 0;
    this->TotalPasswordsFound = 0;
    this->CurrentPasswordLength = 0;
    this->WorkunitBits = 0;
}

void CHWorkunitRobust::LockMutex() {
#if USE_BOOST_THREADS
    this->workunitMutexBoost.lock();
#else
    pthread_mutex_lock(&this->workunitMutexPthreads);
#endif
}

void CHWorkunitRobust::UnlockMutex() {
#if USE_BOOST_THREADS
        this->workunitMutexBoost.unlock();
#else
        pthread_mutex_unlock(&this->workunitMutexPthreads);
#endif
}

int CHWorkunitRobust::CreateWorkunits(uint64_t NumberOfUnits, uint8_t BitsPerUnit, uint8_t PasswordLength) {
    if (this->DebugOutput) {
        printf("CHWorkunitRobust::CreateWorkunits(%lu, %u, %u)\n", NumberOfUnits, BitsPerUnit, PasswordLength);
    }

    uint64_t StartPoint = 0;
    uint64_t EndPoint = 0;
    uint64_t WorkunitId;
    uint64_t NumberOfWorkunits = 0;
    
    CHWorkunitRobustElement NewWorkunit;

    this->LockMutex();

    this->WorkunitBits = BitsPerUnit;
    this->CurrentPasswordLength = PasswordLength;

    // Calculate how many elements are needed per workunit
    this->ElementsPerWorkunit = pow(2.0, (int)BitsPerUnit);

#if WU_DEBUG
    printf("Elements per unit: %llu\n", this->ElementsPerWorkunit);
#endif

    // If the number of workunits fits perfectly, no need for an extra "end" unit.
    if ((NumberOfUnits % this->ElementsPerWorkunit) == 0) {
        NumberOfWorkunits = (NumberOfUnits / this->ElementsPerWorkunit);
    } else {
        NumberOfWorkunits = (NumberOfUnits / this->ElementsPerWorkunit) + 1;
    }
    this->NumberOfWorkunitsTotal = NumberOfWorkunits;

#if WU_DEBUG
    printf("Number of workunits: %ld\n", NumberOfWorkunits);
#endif

    // If we are likely to use too much memory, abort.
    if ((this->NumberOfWorkunitsTotal * sizeof(CHWorkunitRobustElement)) > MAX_WORKUNIT_MEMORY) {
        // TODO: Define global_interface or figure this out.
        //sprintf(global_interface.exit_message,
        //    "Too many workunits!\nPlease use more bits per unit or a smaller problem size.\n");
        //global_interface.exit = 1;
        this->UnlockMutex();
        return 0;
    }

    // Clear the workunit, then set some variables
    memset(&NewWorkunit, 0, sizeof(CHWorkunitRobustElement));

    NewWorkunit.PasswordLength = PasswordLength;
    NewWorkunit.IsValid = 1;

    // For each work unit, set things up.
    for (WorkunitId = 0; WorkunitId < NumberOfWorkunits; WorkunitId++) {
        // Calculate the endpoint.
        EndPoint = StartPoint + this->ElementsPerWorkunit - 1;
        // If the endpoint of this workunit would go past the number
        // of units, set it to however many is left.
        if (EndPoint > NumberOfUnits) {
            EndPoint = NumberOfUnits - 1;
        }
#if WU_DEBUG
        printf("WU %d: SP: %lu  EP: %lu\n", WorkunitId, StartPoint, EndPoint);
#endif
        NewWorkunit.WorkUnitID = WorkunitId;
        NewWorkunit.StartPoint = StartPoint;
        NewWorkunit.EndPoint = EndPoint;

        this->pendingWorkunits.push_back(NewWorkunit);

        StartPoint += this->ElementsPerWorkunit;
    }

    // At this point, we should be done with creating the workunits.
    // Do some final cleanup and go.

    this->NumberOfWorkunitsCompleted = 0;
    this->WorkunitInitialized = 1;
    this->TotalPasswordsFound = 0;

    // Start the execution timer
    this->WorkunitTimer.start();
    this->LastStateSaveTime = 0;

    this->UnlockMutex();

    this->WriteSaveState(1);

    return 1;
}


struct CHWorkunitRobustElement CHWorkunitRobust::GetNextWorkunit(uint16_t ClientId) {
    if (this->DebugOutput) {
        printf("CHWorkunitRobust::GetNextWorkunit(%u)\n", ClientId);
    }

    struct CHWorkunitRobustElement Workunit;

    this->LockMutex();

    // Check to see if there are valid workunits left.
    if (this->pendingWorkunits.size() == 0) {
        // If not, return a unit with isValid = 0.
        if (this->DebugOutput) {
            printf("pendingWorkunits.size() == 0; returning.\n");
        }
        memset(&Workunit, 0, sizeof(CHWorkunitRobustElement));
        this->UnlockMutex();
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
    Workunit.WorkunitRequestedTimestamp = this->WorkunitTimer.stop();
    Workunit.ClientId = ClientId;

    // Add the workunit to the in-flight queue.
    this->assignedWorkunits.push_back(Workunit);
    if (this->DebugOutput) {
        printf("In flight WUs: %lu\n", this->assignedWorkunits.size());
    }  

    this->UnlockMutex();
    this->WriteSaveState(0);
    if (this->DebugOutput) {
        PrintRobustWorkunit(Workunit);
    }
    return Workunit;
}


uint16_t CHWorkunitRobust::GetClientId() {
    // Return an unused network ID
    uint16_t NetworkIdCandidate;
    uint32_t i;
    uint8_t NetworkIdFound;

    if (this->DebugOutput) {
        printf("CHWorkunitRobust::GetClientId()\n");
    }

    this->LockMutex();

    while(1) {
        NetworkIdFound = 0;
        NetworkIdCandidate = (uint16_t)(rand() & 0xffff);
        for (i = 0; i < this->inUseClientIds.size(); i++) {
            if (this->inUseClientIds[i] == NetworkIdCandidate) {
                NetworkIdFound = 1;
                break;
            }
        }
        // If we haven't found the ID, it's good.  Add it & return it!
        if (!NetworkIdFound) {
            this->inUseClientIds.push_back(NetworkIdCandidate);
            this->UnlockMutex();
            if (this->DebugOutput) {
                printf("Returning ClientID: %d\n", NetworkIdCandidate);
            }

            return NetworkIdCandidate;
        }
    }
}

void CHWorkunitRobust::FreeClientId(uint16_t ClientId) {
    int i;

    if (this->DebugOutput) {
        printf("CHWorkunitRobust::FreeClientId(%d)\n", ClientId);
    }

    this->LockMutex();
    for (i = 0; i < this->inUseClientIds.size(); i++) {
        if (this->inUseClientIds[i] == ClientId) {
#if WU_DEBUG
            printf("Found ClientID at position %d\n", i);
#endif
            // Erase the item at the given position.
            this->inUseClientIds.erase(this->inUseClientIds.begin() + i);
        }
    }
    this->UnlockMutex();
}

void CHWorkunitRobust::SubmitWorkunit(struct CHWorkunitRobustElement ReturnedWorkunit) {
    std::list<CHWorkunitRobustElement>::iterator inflightWorkunit;

    this->LockMutex();

    ReturnedWorkunit.WorkunitCompletedTimestamp = this->WorkunitTimer.stop();
    ReturnedWorkunit.SecondsRequiredToComplete = 
            ReturnedWorkunit.WorkunitCompletedTimestamp - ReturnedWorkunit.WorkunitRequestedTimestamp;
    
    // Look for workunit in the list
    for (inflightWorkunit = this->assignedWorkunits.begin(); inflightWorkunit != this->assignedWorkunits.end(); inflightWorkunit++) {
        // Check for the unique Workunit ID
        if (inflightWorkunit->WorkUnitID == ReturnedWorkunit.WorkUnitID) {
#if WU_DEBUG
            printf("Found inflight WU ID: %lu\n", inflightWorkunit->WorkUnitID);
#endif
            this->assignedWorkunits.erase(inflightWorkunit);
            this->NumberOfWorkunitsCompleted++;
            this->TotalPasswordsFound += ReturnedWorkunit.PasswordsFound;
#if WU_DEBUG
            printf("Inflight left: %d\n", this->assignedWorkunits.size());
#endif
            break;
        }
    }
    if (this->DebugOutput) {
        this->PrintInternalState();
    }
    this->UnlockMutex();
    this->WriteSaveState(0);
}

void CHWorkunitRobust::CancelWorkunit(struct CHWorkunitRobustElement CancelledWorkunit) {
    std::list<CHWorkunitRobustElement>::iterator inflightWorkunit;
    struct CHWorkunitRobustElement CancelledWorkunitCopy;

    memset(&CancelledWorkunitCopy, 0, sizeof(struct CHWorkunitRobustElement));

    this->LockMutex();

    // Look for workunit in the list
    for (inflightWorkunit = this->assignedWorkunits.begin(); inflightWorkunit != this->assignedWorkunits.end(); inflightWorkunit++) {
        // Check for the unique Workunit ID
        if (inflightWorkunit->WorkUnitID == CancelledWorkunit.WorkUnitID) {
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
    this->UnlockMutex();
}

void CHWorkunitRobust::CancelAllWorkunitsByClientId(uint16_t ClientId) {
    std::list<CHWorkunitRobustElement>::iterator inflightWorkunit;
    struct CHWorkunitRobustElement CancelledWorkunitCopy;

    if (this->DebugOutput) {
        printf("CHWorkunitRobust::CancelAllWorkunitsByClientId(%d)\n", ClientId);
    }

    memset(&CancelledWorkunitCopy, 0, sizeof(struct CHWorkunitRobustElement));

    this->LockMutex();

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
    this->UnlockMutex();
}

void CHWorkunitRobust::PrintInternalState() {
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

void CHWorkunitRobust::WriteSaveState(char forceWrite) {
    if (this->DebugOutput) {
        printf("CHWorkunitRobust::WriteSaveState(%u)\n", forceWrite);
        forceWrite = 1;
    }

    struct CHRobustWorkunitResumeHeader ResumeHeader;
    FILE *resumeFile;
    std::list<CHWorkunitRobustElement>::iterator inflightWorkunit;
    CHWorkunitRobustElement WorkunitElement;
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
            ((this->WorkunitTimer.stop() - this->LastStateSaveTime) < SAVE_TIME_INTERVAL)) {
        if (this->DebugOutput) {
            printf("No forcewrite, no interval timeout: Returning.\n");
        }

        return;
    }

    // Hate doing it, but we have to lock the global mutex here.
    this->LockMutex();

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
        this->UnlockMutex();
        return;
    }

    // If the output file is open, write out header.
    fwrite(&ResumeHeader, sizeof(struct CHRobustWorkunitResumeHeader), 1, resumeFile);

    // Now write out the binary resume data.
    fwrite(&this->ResumeMetadata[0], 1, this->ResumeMetadata.size(), resumeFile);

    for (inflightWorkunit = this->assignedWorkunits.begin(); inflightWorkunit != this->assignedWorkunits.end(); inflightWorkunit++) {
        WorkunitElement = *inflightWorkunit;
        fwrite(&WorkunitElement, sizeof(CHWorkunitRobustElement), 1, resumeFile);
    }

    // Finally, write out all the inflight and pending workunits.
    for (i = 0; i < this->pendingWorkunits.size(); i++) {
        fwrite(&this->pendingWorkunits[i], sizeof(CHWorkunitRobustElement), 1, resumeFile);
    }

    fclose(resumeFile);

    this->UnlockMutex();
}

int CHWorkunitRobust::LoadStateFromFile(std::string resumeFilename) {
    // Read the resume file in.
    struct CHRobustWorkunitResumeHeader ResumeHeader;
    FILE *resumeFile;
    CHWorkunitRobustElement WorkunitElement;
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

    if (!fread(&ResumeHeader, sizeof(CHRobustWorkunitResumeHeader), 1, resumeFile)) {
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
        if (fread(&WorkunitElement, sizeof(CHWorkunitRobustElement), 1, resumeFile)) {
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
    this->WorkunitInitialized = 1;
    return 1;
}

float CHWorkunitRobust::GetAverageRate() {
    float workRate;

    workRate = (float)(this->NumberOfWorkunitsCompleted * this->ElementsPerWorkunit);
    workRate /= this->WorkunitTimer.stop();
    return workRate;
}


#if ROBUST_WU_UNIT_TEST

int main() {
    printf("CHWorkunitRobust Unit Test!\n");
    int i;

    CHWorkunitRobust *Workunit;
    CHRobustWorkunitElement MyWorkunits[10];
    
    std::vector<uint16_t> activeClientIds;

    Workunit = new CHWorkunitRobust();

    Workunit->LoadStateFromFile("resumeFile.chr");
    Workunit->PrintInternalState();


    Workunit->SetResumeFile("resumeFile.chr");

    Workunit->CreateWorkunits(8, 0, 3);
    for (i = 0 ; i < 10; i ++) {
        uint16_t clientId = Workunit->GetClientId();
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
