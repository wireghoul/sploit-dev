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

#include "MFN_Common/MFNDisplay.h"
#include <stdio.h>

uint16_t MFNDisplay::getFreeThreadId(uint8_t newThreadType) {
    //TODO: Rewrite in some sane manner with a *struct* or something...
    uint16_t threadId = 0;
    
    this->statusUpdateMutexBoost.lock();
    // Iterate through the thread IDs, and see if one is free.
    for (int i = 0; i < this->threadType.size(); i++) {
       // If there is an unused ID, return it.
       if (this->threadType[i] == UNUSED_THREAD) {
           // Store the thread type
           this->threadType[i] = newThreadType;
           threadId = i;
           this->statusUpdateMutexBoost.unlock();
           return threadId;
       }
    }
    
    // No unused IDs - add one, and extend the speed array.
    this->threadType.push_back(newThreadType);
    this->threadRate.push_back(0);
    
    threadId = this->threadType.size() - 1;
    
    this->statusUpdateMutexBoost.unlock();
    return threadId;
}

void MFNDisplay::releaseThreadId(uint16_t oldThreadId) {
    this->statusUpdateMutexBoost.lock();
    if (oldThreadId < this->threadRate.size()) {
        this->threadType[oldThreadId] = UNUSED_THREAD;
        this->threadRate[oldThreadId] = 0;
    }
    this->statusUpdateMutexBoost.unlock();
    this->Refresh();
}

void MFNDisplay::setThreadCrackSpeed(uint16_t threadId, float rate) {
    // Ensure that the thread ID is valid, then adjust the rate.
    this->statusUpdateMutexBoost.lock();
    
    if (this->printDebugOutput) {
        printf("MFND: Thread %d setting rate to %s\n", threadId, 
                this->getConvertedRateString(rate).c_str());
    }
    
    if (threadId < this->threadRate.size()) {
        this->threadRate[threadId] = rate;
    }
    this->statusUpdateMutexBoost.unlock();
    this->Refresh();
}


float MFNDisplay::getCurrentCrackRate() {
    float totalRate = 0;
   
    std::vector<float>::iterator rateIterator;
    
    this->statusUpdateMutexBoost.lock();
    for (rateIterator = this->threadRate.begin(); 
            rateIterator < this->threadRate.end(); rateIterator++) {
        totalRate += *rateIterator;
    }
    this->statusUpdateMutexBoost.unlock();
    
    if (this->printDebugOutput) {
        printf("MFND: Total crack rate %s/s\n", this->getConvertedRateString(totalRate).c_str());
    }

    return totalRate;
}

std::string MFNDisplay::getConvertedRateString(float rate) {
    // Suffixes for each division.
    const char suffixByThousands[] = {' ', 'K', 'M', 'B', 'T'};
    const int maxDivisions = 4;
    int divisionCount = 0;
    
    
    // Use sprintf, because it's easier for what I want to do.
    char outputBuffer[100];
    
    while ((rate > 1000.0) && (divisionCount < maxDivisions)) {
        rate /= 1000.0;
        divisionCount++;
    }
    
    sprintf(outputBuffer, "%0.2f%c", rate, suffixByThousands[divisionCount]);
    
    return std::string(outputBuffer);
}
