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

#ifndef __CHDISPLAY_H
#define __CHDISPLAY_H

#include "Multiforcer_Common/CHCommon.h"

class CHDisplay {
public:
    virtual void Refresh() = 0;

    virtual ~CHDisplay() {};

    virtual void setHashName(char *) = 0;
    virtual void setTotalHashes(uint64_t) = 0;
    virtual void setCrackedHashes(uint64_t) = 0;
    virtual void addCrackedHashes(uint64_t) = 0;
    virtual void setPasswordLen(int) = 0;
    virtual void setThreadCrackSpeed(unsigned char threadId, unsigned char threadType, float rateInM) = 0;
    virtual void setWorkunitsTotal(uint32_t) = 0;
    virtual void setWorkunitsCompleted(uint32_t) = 0;
    virtual void addCrackedPassword(char *) = 0;
    virtual void addStatusLine(char *) = 0;
    // Sets the system mode: Standalone, network server, network client.
    virtual void setSystemMode(int, char *) = 0;
    // Add or subtract from the number of connected clients
    virtual void alterNetworkClientCount(int) = 0;
    // Get the next free thread ID
    virtual int getFreeThreadId() = 0;
    // Get the current total rate for this host
    virtual float getCurrentCrackRate() = 0;
};

#endif