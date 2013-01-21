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

#include "CH_Common/CHWorkunit.h"
#include "Multiforcer_Common/CHCommon.h"

#ifdef _WIN32

using namespace std;
 
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

#endif


using namespace std;
extern struct global_commands global_interface;


CHWorkunit::CHWorkunit() {
    struct timeval resume_time;
    time_t resume_time_t;


    // Set flag to indicate this class is created but not set up with a WU
    this->WorkunitInitialized = 0;

    this->NumberOfWorkunits = 0;
    this->NumberOfWorkunitsCompleted = 0;

    this->WorkunitBits = 0;

    this->CHWorkunits = NULL;

    // Boost mutex does not require init.
#if !USE_BOOST_THREADS
    pthread_mutexattr_init(&this->mutex1attr);
    pthread_mutex_init(&this->mutex1, &this->mutex1attr);
#endif

    // Create a resume filename
    gettimeofday(&resume_time, NULL);
    resume_time_t=resume_time.tv_sec;

    memset(this->ResumeFilename, 0, sizeof(this->ResumeFilename));
    strftime(this->ResumeFilename,128,"%Y-%m-%d-%H-%M-%S",localtime(&resume_time_t));
    sprintf(this->ResumeFilename, "%s.crs",this->ResumeFilename);

    this->ResumeSaveState = fopen(this->ResumeFilename, "w");
    if (!this->ResumeSaveState) {
        printf("Cannot create resume file %s!\n", this->ResumeFilename);
        exit(1);
    }
    // Init the last save point to now.
    this->LastStateSave = time (NULL);

    // Clear the network client if networking is in use.
#if USE_NETWORK
    this->NetworkClient = NULL;
#endif
}

CHWorkunit::~CHWorkunit(){
    if (this->ResumeSaveState) {
        fclose(this->ResumeSaveState);
        // If we are done, unlink the resume file
        if (this->NumberOfWorkunitsCompleted == this->NumberOfWorkunits) {
            unlink(this->ResumeFilename);
            printf("Deleting resume file; work is done!\n");
        }
    }
}

// Given the size of the work to be done, create the workunits
// Returns 0 on failure, 1 on success.
int CHWorkunit::CreateWorkunits(uint64_t NumberOfUnits, unsigned char BitsPerUnit) {
    uint64_t StartPoint = 0;
    uint64_t EndPoint = 0;
    uint64_t WorkunitId;
    uint64_t NumberOfWorkunits = 0;

    if (BitsPerUnit < 16) {
        printf("Need >16 bits per unit.\n");
        return 0;
    }

    this->WorkunitBits = BitsPerUnit;

    // Calculate how many elements are needed per workunit
    this->ElementsPerWorkunit = pow(2.0, (int)BitsPerUnit);

    //printf("Elements per unit: %llu\n", this->ElementsPerWorkunit);

    NumberOfWorkunits = (NumberOfUnits / this->ElementsPerWorkunit) + 1;
    this->NumberOfWorkunits = NumberOfWorkunits;
    //printf("Number of workunits: %ld\n", NumberOfWorkunits);

    // If workunits has been allocated previously, delete it.
    if (this->CHWorkunits) {
        delete [] this->CHWorkunits;
    }

    // Catch 'new' exception should we have too much to allocate.
    try {
        this->CHWorkunits = new CHWorkunitElement[NumberOfWorkunits];
    } catch (exception& e) {
        sprintf(global_interface.exit_message, 
            "Cannot allocate memory: %s.\nPlease use more bits per unit or a smaller problem size.\n", e.what());
        global_interface.exit = 1;
        return 0;
    }

    // For each work unit, set things up.
    for (WorkunitId = 0; WorkunitId < NumberOfWorkunits; WorkunitId++) {
        // Calculate the endpoint.
        EndPoint = StartPoint + this->ElementsPerWorkunit - 1;
        // If the endpoint of this workunit would go past the number
        // of units, set it to however many is left.
        if (EndPoint > NumberOfUnits) {
            EndPoint = NumberOfUnits;
        }
        //printf("WU %d: SP: %lu  EP: %lu\n", WorkunitId, StartPoint, EndPoint);

        this->CHWorkunits[WorkunitId].WorkUnitID = WorkunitId;
        this->CHWorkunits[WorkunitId].StartPoint = StartPoint;
        this->CHWorkunits[WorkunitId].EndPoint = EndPoint;
        this->CHWorkunits[WorkunitId].SecondsToFinish = 0;
        this->CHWorkunits[WorkunitId].TimeRequested = 0;
        this->CHWorkunits[WorkunitId].TimeReturned = 0;
        this->CHWorkunits[WorkunitId].PasswordsFound = 0;
        StartPoint += this->ElementsPerWorkunit;
    }

    // At this point, we should be done with creating the workunits.
    // Do some final cleanup and go.

    this->LastAllocatedWorkunitId = 0;
    this->NumberOfWorkunitsCompleted = 0;
    this->WorkunitInitialized = 1;
    this->TotalPasswordsFound = 0;

    
    this->ExecutionStartTimeMs = time(NULL);
    gettimeofday(&this->start, NULL);

    return 1;
}


struct CHWorkunitElement* CHWorkunit::GetNextWorkunit(){
    struct CHWorkunitElement* WorkunitToReturn;
    uint64_t i;

    // Mutexes matter...
#if USE_BOOST_THREADS
    this->mutex1Boost.lock();
#else
    pthread_mutex_lock(&this->mutex1);
#endif
    //printf("In GetNextWorkunit\n");
    // If the network client is in use, get data from it.
#if USE_NETWORK && 0
    if (this->NetworkClient) {
        // Get us some general data to get the password length.
        // If it does NOT match what we think it is, return null to reset stuff.
        CHMultiforcerNetworkGeneral NetworkGeneralData;
        NetworkClient->updateGeneralInfo();
        NetworkClient->provideGeneralInfo(&NetworkGeneralData);
        if (this->currentPasswordLength != NetworkGeneralData.structure.password_length) {
            WorkunitToReturn = NULL;
            this->currentPasswordLength = NetworkGeneralData.structure.password_length;
        } else {
            WorkunitToReturn = this->NetworkClient->getNextNetworkWorkunit();
        }


        // Unlock and return here.
#if USE_BOOST_THREADS
        this->mutex1Boost.unlock();
#else
        pthread_mutex_unlock(&this->mutex1);
#endif
        return WorkunitToReturn;
    }
#endif

    if (this->LastAllocatedWorkunitId < this->NumberOfWorkunits) {
        // We have not gone all the way through the workunit space.
        // Grab the next workunit and assign it.
        this->CHWorkunits[this->LastAllocatedWorkunitId].TimeRequested = time(NULL);
        WorkunitToReturn = new CHWorkunitElement;

        WorkunitToReturn->WorkUnitID = this->CHWorkunits[this->LastAllocatedWorkunitId].WorkUnitID;
        WorkunitToReturn->StartPoint = this->CHWorkunits[this->LastAllocatedWorkunitId].StartPoint;
        WorkunitToReturn->EndPoint = this->CHWorkunits[this->LastAllocatedWorkunitId].EndPoint;

        this->LastAllocatedWorkunitId++;
    } else /*if (this->NumberOfWorkunitsCompleted == this->NumberOfWorkunits)*/ {
        // Everything is legitimately done.
        WorkunitToReturn = NULL;
    } /*else {
        int foundStaleWorkunit = 0;
        // We are at the end, but there are still uncompleted workunits.
        // Check to see if any are "expired" - old.
        for (i = 0; i < this->NumberOfWorkunits; i++) {
            if ((time(NULL) - this->CHWorkunits[i].TimeRequested) > WORKUNIT_STALE_AGE) {
                // This is a stale workunit!
                printf("Found stale workunit!\n");

                this->CHWorkunits[i].TimeRequested = time(NULL);

                // Go ahead & allocate it.
                WorkunitToReturn = new CHWorkunitElement;
                WorkunitToReturn->WorkUnitID = this->CHWorkunits[i].WorkUnitID;
                WorkunitToReturn->StartPoint = this->CHWorkunits[i].StartPoint;
                WorkunitToReturn->EndPoint = this->CHWorkunits[i].EndPoint;
                foundStaleWorkunit = 1;
                break;
            }
        }
        // break goes to here
        if (!foundStaleWorkunit) {
            // If we have NOT found a stale workunit, tell threads to hang out.
            printf("Returning workunit plz hold.\n");
            WorkunitToReturn = WORKUNIT_PLZ_HOLD;
        }
    }*/

    // Make sure we unlock.
#if USE_BOOST_THREADS
    this->mutex1Boost.unlock();
#else
    pthread_mutex_unlock(&this->mutex1);
#endif
    return WorkunitToReturn;
}


int CHWorkunit::SubmitWorkunit(struct CHWorkunitElement* Workunit, uint32_t FoundPasswords){

#if USE_BOOST_THREADS
    this->mutex1Boost.lock();
#else
    pthread_mutex_lock(&this->mutex1);
#endif


    // If the network client is in use, send data to it.
#if USE_NETWORK && 0
    if (this->NetworkClient) {
        int returnValue;
        //printf("About to submit workunit to network.\n");
        returnValue = this->NetworkClient->submitNetworkWorkunit(Workunit, FoundPasswords);
        //printf("Back from submitting workunit to network.\n");
        // Unlock and return here.
#if USE_BOOST_THREADS
        this->mutex1Boost.unlock();
#else
        pthread_mutex_unlock(&this->mutex1);
#endif
        delete Workunit;
        return returnValue;
    }
#endif
    this->CHWorkunits[Workunit->WorkUnitID].TimeReturned = time(NULL);
    this->CHWorkunits[Workunit->WorkUnitID].PasswordsFound = Workunit->PasswordsFound;
    this->CHWorkunits[Workunit->WorkUnitID].SecondsToFinish = Workunit->SecondsToFinish;

    // Increment the number back.
    this->NumberOfWorkunitsCompleted++;
    this->TotalPasswordsFound += FoundPasswords;
    // Be sure to delete the workunit!
    delete Workunit;

    // Decide if we need to save the state.
    if ((this->ResumeSaveState) && ((time (NULL) - this->LastStateSave) > WORKUNIT_SAVE_INTERVAL)) {
        this->WriteSaveState();
        this->LastStateSave = time(NULL);
    }

#if USE_BOOST_THREADS
    this->mutex1Boost.unlock();
#else
    pthread_mutex_unlock(&this->mutex1);
#endif
    return 1;
}


void CHWorkunit::PrintStatusLine() {
    uint64_t  executionTimeMs;

    gettimeofday(&this->end, NULL);

    executionTimeMs = ((this->end.tv_sec  - this->start.tv_sec) * 1000 + (this->end.tv_usec - this->start.tv_usec)/1000.0) + 0.5;

    printf("Work units completed: %ld/%ld (%0.2f%% done)  %0.2fM/s\n",
            this->NumberOfWorkunitsCompleted,
            this->NumberOfWorkunits,
            (100.0 * ((float)this->NumberOfWorkunitsCompleted / (float)this->NumberOfWorkunits)),
            ((float)this->NumberOfWorkunitsCompleted * (float)this->ElementsPerWorkunit) / ((float)executionTimeMs * 1000.0)
            );

    fflush(stdout);
}

float CHWorkunit::GetAverageRate() {
    uint64_t  executionTimeMs;

    gettimeofday(&this->end, NULL);

    executionTimeMs = ((this->end.tv_sec  - this->start.tv_sec) * 1000 + (this->end.tv_usec - this->start.tv_usec)/1000.0) + 0.5;
    return ((float)this->NumberOfWorkunitsCompleted * (float)this->ElementsPerWorkunit) / ((float)executionTimeMs * 1000.0);
}

uint64_t CHWorkunit::GetNumberOfWorkunits() {
    return this->NumberOfWorkunits;
}
uint64_t CHWorkunit::GetNumberOfCompletedWorkunits() {
    return this->NumberOfWorkunitsCompleted;
}

int CHWorkunit::GetWorkunitBits() {
    return this->WorkunitBits;
}

// Write out the current save state.
void CHWorkunit::WriteSaveState() {
    // If the resume file exists, manipulate it.
    if (this->ResumeSaveState) {
        // Set the file pointer to the beginning of the file.
        fseek(this->ResumeSaveState, 0, SEEK_SET);
        // We do not need to truncate the file.  It will always be the same
        // size or larger for a given task.
        fwrite(this->CHWorkunits, sizeof(CHWorkunitElement), this->NumberOfWorkunits, this->ResumeSaveState);
#ifndef _WIN32 
		// fsync does not exist on Windows in the current state.
		fsync(fileno(this->ResumeSaveState));
#endif
    }
}

#if USE_NETWORK
void CHWorkunit::setNetworkClient(CHNetworkClient *newNetworkClient) {
    this->NetworkClient = newNetworkClient;
}

void CHWorkunit::setPasswordLength(int newPasswordLength) {
    this->currentPasswordLength = newPasswordLength;
}

#endif