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

#ifndef __GRTCRACKDISPLAY_H
#define __GRTCRACKDISPLAY_H

#include "GRT_Common/GRTCommon.h"

// Define the stages
#define GRT_CRACK_CHGEN  1
#define GRT_CRACK_SEARCH 2
#define GRT_CRACK_REGEN  3

#define UNUSED_THREAD 0
#define GPU_THREAD 1
#define CPU_THREAD 2
#define NETWORK_HOST 3


class GRTCrackDisplay {
public:
    virtual void Refresh() = 0;

    virtual ~GRTCrackDisplay() {};

    // Set the hash name to be cracked.
    virtual void setHashName(char *) = 0;

    // Set the total hashes, and the number cracked.
    virtual void setTotalHashes(uint64_t) = 0;
    virtual void setCrackedHashes(uint64_t) = 0;

    // Set the total number of table files and the current one.
    virtual void setTotalTables(uint32_t) = 0;
    virtual void setCurrentTableNumber(uint32_t) = 0;

    // Set the current table filename for display.
    virtual void setTableFilename(const char *) = 0;

    // Set the percent done of this stage.
    virtual void setStagePercent(float) = 0;

    // Add a number of cracked hashes to the total
    virtual void addCrackedHashes(uint64_t) = 0;

    // Set the current thread crack speed.
    virtual void setThreadCrackSpeed(unsigned char threadId, unsigned char threadType, float rateInM) = 0;

    // Set the total and completed workunits for this stage.
    virtual void setWorkunitsTotal(uint32_t) = 0;
    virtual void setWorkunitsCompleted(uint32_t) = 0;

    // Get each threads status through its problem space
    virtual void setThreadFractionDone(unsigned char threadId, float fractionDone) = 0;

    // Display a cracked password.
    virtual void addCrackedPassword(char *) = 0;
    
    // Display an arbitrary status line in the scroller.
    virtual void addStatusLine(char *) = 0;

    // Set the stage: (CH Gen, Search, Chain Regen)
    virtual void setSystemStage(int) = 0;

    // Get the current total rate for this host
    virtual float getCurrentCrackRate() = 0;
    
    // End curses mode if needed.
    virtual void endCursesMode() {
        
    }
};

#endif