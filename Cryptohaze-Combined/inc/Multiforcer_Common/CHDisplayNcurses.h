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

#ifndef __CHDISPLAYNCURSES_H
#define __CHDISPLAYNCURSES_H

#include "Multiforcer_Common/CHDisplay.h"


const char programTitle[] = "Cryptohaze Multiforcer 1.31";

#define MAX_SUPPORTED_THREADS 16


#define UNUSED_THREAD 0
#define GPU_THREAD 1
#define CPU_THREAD 2
#define NETWORK_HOST 3

#define SYSTEM_MODE_STANDALONE 1
#define SYSTEM_MODE_SERVER 2
#define SYSTEM_MODE_CLIENT 3

// Formerly 18
#define DISPLAY_PASSWORD_SCROLLER_HEIGHT 64
#define DISPLAY_PASSWORD_SCROLLER_WIDTH 64

// Formerly 13
#define STATUS_SCROLLER_HEIGHT 64
#define STATUS_SCROLLER_WIDTH 26

// Data passed to each thread.
typedef struct displayRefreshData {
    CHDisplay *Display;
 }displayRefreshData;

class CHMultiforcerDisplay : public CHDisplay {
private:
    // The current X and Y dimensions
    int currentMaxX, currentMaxY;

    struct displayRefreshData displayRefreshData;

    // Various stats to display
    uint64_t TotalHashes;
    uint64_t CrackedHashes;
    int PasswordLen;
    uint32_t WorkunitsTotal;
    uint32_t WorkunitsCompleted;

    uint64_t CrackingTimeStart;

    // Thread type: CPU thread, GPU thread, or unused.
    unsigned char threadType[MAX_SUPPORTED_THREADS];
    // Thread rate in M steps/sec
    float threadRate[MAX_SUPPORTED_THREADS];

    // Remote IP addresses for threads
    uint32_t threadRemoteIps[MAX_SUPPORTED_THREADS];

    // For the password scroller
    int PasswordScrollerIndex;
    char PasswordsToDisplay[DISPLAY_PASSWORD_SCROLLER_HEIGHT][DISPLAY_PASSWORD_SCROLLER_WIDTH];

    // Status scroller
    int StatusScrollerIndex;
    char StatusToDisplay[STATUS_SCROLLER_HEIGHT][STATUS_SCROLLER_WIDTH];

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

    char HashName[MAX_HASH_STRING_LENGTH];
    char systemModeString[1024];
    char redrawModeString;

    int systemMode;
    int numberOfNetworkClients;

    // Draw the basic window framework based on current dimensions
    void DrawFramework();


public:
    CHMultiforcerDisplay();
    ~CHMultiforcerDisplay();

    void Refresh();

    void setHashName(char *);
    void setTotalHashes(uint64_t);
    void setCrackedHashes(uint64_t);
    void addCrackedHashes(uint64_t);
    void setPasswordLen(int);
    void setThreadCrackSpeed(unsigned char threadId, unsigned char threadType, float rateInM);
    void setWorkunitsTotal(uint32_t);
    void setWorkunitsCompleted(uint32_t);
    void addCrackedPassword(char *);
    void addStatusLine(char *);
    void setSystemMode(int, char *);
    void alterNetworkClientCount(int);
    int getFreeThreadId();
    float getCurrentCrackRate();

};

#endif
