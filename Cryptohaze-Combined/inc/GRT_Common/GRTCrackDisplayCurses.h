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

#ifndef __GRTCRACKDISPLAYCURSES_H
#define __GRTCRACKDISPLAYCURSES_H

#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTCrackDisplay.h"
#include "GRT_Common/GRTHashes.h"
// C headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <math.h>
#include <signal.h>
#include <limits.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif


// C++ Headers
#include <iostream>
#include <valarray>
#include <new>
#include <exception>

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


const char programTitle[] = "Cryptohaze GRTCrack 1.40";

#define MAX_SUPPORTED_THREADS 16

#define MAX_FILENAME_LENGTH 1024
#define STAGE_LENGTH_CHARS 16
// Formerly 18
#define DISPLAY_PASSWORD_SCROLLER_HEIGHT 64
#define DISPLAY_PASSWORD_SCROLLER_WIDTH 64

// Formerly 13
#define STATUS_SCROLLER_HEIGHT 64
#define STATUS_SCROLLER_WIDTH 26

//#define MAX_HASH_STRING_LENGTH 64

class GRTCrackDisplayCurses : public GRTCrackDisplay {
private:
    // The current X and Y dimensions
    int currentMaxX, currentMaxY;

    // Various stats to display
    uint64_t HashesTotal, HashesCracked;
    uint32_t WorkunitsTotal, WorkunitsCompleted;
    uint32_t TablesTotal, TableCurrent;

    float PercentDone;
    
    uint64_t CrackingTimeStart;

    // Thread type: CPU thread, GPU thread, or unused.
    unsigned char threadType[MAX_SUPPORTED_THREADS];
    // Thread rate in M steps/sec
    float threadRate[MAX_SUPPORTED_THREADS];
    
    float threadFractionDone[MAX_SUPPORTED_THREADS];

    // Remote IP addresses for threads
    uint32_t threadRemoteIps[MAX_SUPPORTED_THREADS];

    // For the password scroller
    int PasswordScrollerIndex;
    char PasswordsToDisplay[DISPLAY_PASSWORD_SCROLLER_HEIGHT][DISPLAY_PASSWORD_SCROLLER_WIDTH];

    // Status scroller
    int StatusScrollerIndex;
    char StatusToDisplay[STATUS_SCROLLER_HEIGHT][STATUS_SCROLLER_WIDTH];
    
    char TableFilename[MAX_FILENAME_LENGTH];

    // Mutexes for the display update and status line update.
#if USE_BOOST_THREADS
    boost::mutex displayUpdateMutexBoost;
    boost::mutex statusLineMutexBoost;
#else
    pthread_mutex_t displayUpdateMutex, statusLineMutex;
    pthread_mutexattr_t displayUpdateMutexAttr, statusLineMutexAttr;
#endif

    // For pausing the program
    WINDOW *pause_win;

    char HashName[64];

    int systemStage;
    char systemStageString[STAGE_LENGTH_CHARS];
    
    // Draw the basic window framework based on current dimensions
    void DrawFramework();
    
    char cursesEnabled;


public:
    GRTCrackDisplayCurses();
    ~GRTCrackDisplayCurses();

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
    void endCursesMode();
};

#endif
