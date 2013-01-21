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

#ifndef __GRTCRACKDISPLAYDEBUG_H
#define __GRTCRACKDISPLAYDEBUG_H

#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTCrackDisplay.h"
// C headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
//#include <unistd.h>
#include <math.h>
#include <signal.h>
#include <limits.h>

// Display headers
#ifdef _WIN32
#include <curses.h>
#else
#include <ncurses.h>
#endif

#if USE_BOOST_THREADS
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#else
#include <pthread.h>
#endif


#define MAX_SUPPORTED_THREADS_DEBUG 16

class GRTCrackDisplayDebug : public GRTCrackDisplay {
private:
    // Mutexes for the display update and status line update.
#if USE_BOOST_THREADS
    boost::mutex displayUpdateMutexBoost;
    boost::mutex statusLineMutexBoost;
#else
    pthread_mutex_t displayUpdateMutex, statusLineMutex;
    pthread_mutexattr_t displayUpdateMutexAttr, statusLineMutexAttr;
#endif

    // Some debugging tests
    uint32_t WorkunitsTotal, WorkunitsCompleted;
    float PercentDone;
    float threadFractionDone[MAX_SUPPORTED_THREADS_DEBUG];


public:
    GRTCrackDisplayDebug();
    ~GRTCrackDisplayDebug();

    void Refresh();

    void setHashName(char *);
    void setTotalHashes(uint64_t);
    void setCrackedHashes(uint64_t);
    void setTotalTables(uint32_t);
    void setCurrentTableNumber(uint32_t);
    void setTableFilename(const char *);
    void setStagePercent(float);
    void addCrackedHashes(uint64_t);
    void setThreadCrackSpeed(unsigned char threadId, unsigned char threadType, float rateInM);
    void setWorkunitsTotal(uint32_t);
    void setWorkunitsCompleted(uint32_t);
    void addCrackedPassword(char *);
    void addStatusLine(char *);
    void setSystemStage(int);
    float getCurrentCrackRate();
    void setThreadFractionDone(unsigned char threadId, float fractionDone);

};

#endif