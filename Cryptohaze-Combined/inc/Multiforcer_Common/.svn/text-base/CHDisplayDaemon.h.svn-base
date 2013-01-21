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

#ifndef __CHDISPLAYNCURSESDAEMON_H
#define __CHDISPLAYNCURSESDAEMON_H

#include "Multiforcer_Common/CHDisplay.h"



#define MAX_SUPPORTED_THREADS 16


#define UNUSED_THREAD 0
#define GPU_THREAD 1
#define CPU_THREAD 2

// Formerly 18
#define DISPLAY_PASSWORD_SCROLLER_HEIGHT 64
#define DISPLAY_PASSWORD_SCROLLER_WIDTH 64

// Formerly 13
#define STATUS_SCROLLER_HEIGHT 64
#define STATUS_SCROLLER_WIDTH 26

/*
class CHDisplay {
public:
    virtual void Refresh() = 0;
};
*/
class CHMultiforcerDisplayDaemon : public CHDisplay {
private:
    // Thread type: CPU thread, GPU thread, or unused.
    unsigned char threadType[MAX_SUPPORTED_THREADS];
    // Thread rate in M steps/sec
    float threadRate[MAX_SUPPORTED_THREADS];
public:
    CHMultiforcerDisplayDaemon();
    ~CHMultiforcerDisplayDaemon();

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