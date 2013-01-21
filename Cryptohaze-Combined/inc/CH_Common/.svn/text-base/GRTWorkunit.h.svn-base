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

// Include file for the GRTWorkunit class to handle workunit distribution.

#ifndef _GRTWORKUNIT_H
#define _GRTWORKUNIT_H

#include "GRT_Common/GRTCommon.h"
#include <limits.h>
#include <time.h>

#if USE_BOOST_THREADS
#include <boost/thread/mutex.hpp>
#else
#include <pthread.h>
#endif


#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

typedef struct GRTWorkunitElement {
    uint64_t WorkUnitID;     // Workunit unique ID
    uint64_t StartPoint;     // Starting offset ID
    uint64_t EndPoint;       // Ending offset ID
    float    SecondsToFinish;// How many seconds of work it took
    uint32_t TimeRequested;  // Seconds since start it was requested
    uint32_t TimeReturned;   // Seconds since start it was returned
} GRTWorkunitElement;


class GRTWorkunit {
private:
    char WorkunitInitialized;
    struct GRTWorkunitElement *CHWorkunits;
    uint64_t LastAllocatedWorkunitId;
    uint64_t ExecutionStartTimeMs; // Execution start in MS
    float CurrentRuntime; // Current runtime in seconds
#if USE_BOOST_THREADS
    boost::mutex mutex1Boost;
#else
    pthread_mutex_t mutex1;  // Mutex for important things.
    pthread_mutexattr_t mutex1attr;
#endif
    uint64_t NumberOfWorkunits;
    uint64_t NumberOfWorkunitsCompleted;
    uint64_t ElementsPerWorkunit;

    uint64_t TotalPasswordsFound;

    //timer values
    struct timeval start, end;

    // Number of bits per workunit, as needed
    int WorkunitBits;


public:
    GRTWorkunit();
    ~GRTWorkunit();
    // Create workunits: N workunits to fill NumberOfUnits, with BitsPerUnit as the size of each unit
    int CreateWorkunits(uint64_t NumberOfUnits, unsigned char BitsPerUnit);
    struct GRTWorkunitElement* GetNextWorkunit();
    int SubmitWorkunit(struct GRTWorkunitElement* Workunit);
    void PrintStatusLine();
    uint64_t GetNumberOfWorkunits();
    uint64_t GetNumberOfCompletedWorkunits();
    float GetAverageRate();
    int GetWorkunitBits();
};


#endif
