/*
Cryptohaze GPU Rainbow Tables
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

#include "CH_Common/GRTWorkunit.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <exception>
#include <time.h>
//#include <sys/time.h>
//#include <unistd.h>

#ifdef _WIN32
#include <time.h>
#include <windows.h>
#include <iostream>

using namespace std;
 
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
 
#endif


using namespace std;


GRTWorkunit::GRTWorkunit() {
    // Set flag to indicate this class is created but not set up with a WU
    this->WorkunitInitialized = 0;

    this->NumberOfWorkunits = 0;
    this->NumberOfWorkunitsCompleted = 0;

    this->WorkunitBits = 0;

#if !USE_BOOST_THREADS
    pthread_mutexattr_init(&this->mutex1attr);
    pthread_mutex_init(&this->mutex1, &this->mutex1attr);
#endif
}

GRTWorkunit::~GRTWorkunit(){
    if (this->WorkunitInitialized) {
        delete [] this->CHWorkunits;
    }
}

// Given the size of the work to be done, create the workunits
// Returns 0 on failure, 1 on success.
int GRTWorkunit::CreateWorkunits(uint64_t NumberOfUnits, unsigned char BitsPerUnit) {
    uint64_t StartPoint = 0;
    uint64_t EndPoint = 0;
    uint64_t WorkunitId;
    uint64_t NumberOfWorkunits = 0;

    // Clear the workunit if it's being reinitialized.
    if (this->WorkunitInitialized) {
        delete [] this->CHWorkunits;
        this->NumberOfWorkunitsCompleted = 0;
        this->WorkunitInitialized = 0;
    }

    this->WorkunitBits = BitsPerUnit;

    // Calculate how many elements are needed per workunit
    this->ElementsPerWorkunit = pow((double)2.0, BitsPerUnit);

    //printf("Elements: %llu\n", NumberOfUnits);
    //printf("Elements per unit: %llu\n", this->ElementsPerWorkunit);

    NumberOfWorkunits = (NumberOfUnits / this->ElementsPerWorkunit);

    // If the number does not divide equally, add one.
    if (NumberOfUnits % this->ElementsPerWorkunit) {
        NumberOfWorkunits++;
    }

    this->NumberOfWorkunits = NumberOfWorkunits;



    //printf("Number of workunits: %ld\n", NumberOfWorkunits);

    // Catch 'new' exception should we have too much to allocate.
    try {
        this->CHWorkunits = new GRTWorkunitElement[NumberOfWorkunits];
    } catch (exception& e) {
        printf("Cannot allocate memory: %s.\nPlease use more bits per unit or a smaller problem size.\n", e.what());
        return 0;
    }

    // For each work unit, set things up.
    for (WorkunitId = 0; WorkunitId < NumberOfWorkunits; WorkunitId++) {
        // Calculate the endpoint.
        EndPoint = StartPoint + this->ElementsPerWorkunit - 1;
        // If the endpoint of this workunit would go past the number
        // of units, set it to however many is left.
        if (EndPoint > NumberOfUnits) {
            EndPoint = NumberOfUnits - 1;
        }
        //printf("WU %d: SP: %lu  EP: %lu\n", WorkunitId, StartPoint, EndPoint);

        this->CHWorkunits[WorkunitId].WorkUnitID = WorkunitId;
        this->CHWorkunits[WorkunitId].StartPoint = StartPoint;
        this->CHWorkunits[WorkunitId].EndPoint = EndPoint;
        this->CHWorkunits[WorkunitId].SecondsToFinish = 0;
        this->CHWorkunits[WorkunitId].TimeRequested = 0;
        this->CHWorkunits[WorkunitId].TimeReturned = 0;
        StartPoint += this->ElementsPerWorkunit;
    }

    // At this point, we should be done with creating the workunits.
    // Do some final cleanup and go.

    this->LastAllocatedWorkunitId = 0;
    this->NumberOfWorkunitsCompleted = 0;
    this->WorkunitInitialized = 1;
    this->TotalPasswordsFound = 0;
    //printf("WU init complete\n");


    this->ExecutionStartTimeMs = time(NULL);
    gettimeofday(&this->start, NULL);

    return 1;
}


struct GRTWorkunitElement* GRTWorkunit::GetNextWorkunit(){
    struct GRTWorkunitElement* WorkunitToReturn;

#if USE_BOOST_THREADS
    this->mutex1Boost.lock();
#else
    pthread_mutex_lock(&this->mutex1);
#endif

    // If there are workunits left, return one.
    if (this->LastAllocatedWorkunitId < this->NumberOfWorkunits) {
        this->CHWorkunits[this->LastAllocatedWorkunitId].TimeRequested = time(NULL);
        WorkunitToReturn = new GRTWorkunitElement;

        WorkunitToReturn->WorkUnitID = this->CHWorkunits[this->LastAllocatedWorkunitId].WorkUnitID;
        WorkunitToReturn->StartPoint = this->CHWorkunits[this->LastAllocatedWorkunitId].StartPoint;
        WorkunitToReturn->EndPoint = this->CHWorkunits[this->LastAllocatedWorkunitId].EndPoint;

        this->LastAllocatedWorkunitId++;
    } else {
        WorkunitToReturn = NULL;
    }

#if USE_BOOST_THREADS
    this->mutex1Boost.unlock();
#else
    pthread_mutex_unlock(&this->mutex1);
#endif
    return WorkunitToReturn;
}


int GRTWorkunit::SubmitWorkunit(struct GRTWorkunitElement* Workunit){
#if USE_BOOST_THREADS
    this->mutex1Boost.lock();
#else
    pthread_mutex_lock(&this->mutex1);
#endif

    this->CHWorkunits[Workunit->WorkUnitID].TimeReturned = time(NULL);
    this->CHWorkunits[Workunit->WorkUnitID].SecondsToFinish = Workunit->SecondsToFinish;

    // Increment the number back.
    this->NumberOfWorkunitsCompleted++;
    // Be sure to delete the workunit!
    delete Workunit;
#if USE_BOOST_THREADS
    this->mutex1Boost.unlock();
#else
    pthread_mutex_unlock(&this->mutex1);
#endif
    return 1;
}


void GRTWorkunit::PrintStatusLine() {
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

float GRTWorkunit::GetAverageRate() {
    uint64_t  executionTimeMs;

    gettimeofday(&this->end, NULL);

    executionTimeMs = ((this->end.tv_sec  - this->start.tv_sec) * 1000 + (this->end.tv_usec - this->start.tv_usec)/1000.0) + 0.5;
    return ((float)this->NumberOfWorkunitsCompleted * (float)this->ElementsPerWorkunit) / ((float)executionTimeMs * 1000.0);

}

uint64_t GRTWorkunit::GetNumberOfWorkunits() {
    return this->NumberOfWorkunits;
}
uint64_t GRTWorkunit::GetNumberOfCompletedWorkunits() {
    return this->NumberOfWorkunitsCompleted;
}

int GRTWorkunit::GetWorkunitBits() {
    return this->WorkunitBits;
}
