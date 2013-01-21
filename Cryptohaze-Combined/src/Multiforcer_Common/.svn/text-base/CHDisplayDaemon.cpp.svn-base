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

#include "Multiforcer_Common/CHDisplayDaemon.h"
#include "Multiforcer_Common/CHCommon.h"
#include "Multiforcer_Common/CHHashes.h"

extern struct global_commands global_interface;

void CHMultiforcerDisplayDaemon::setTotalHashes(uint64_t newTotalHashes) {
}
void CHMultiforcerDisplayDaemon::setCrackedHashes(uint64_t newCrackedHashes) {
}
void CHMultiforcerDisplayDaemon::addCrackedHashes(uint64_t newHashes) {
}

void CHMultiforcerDisplayDaemon::setPasswordLen(int setPasswordLen) {
}
void CHMultiforcerDisplayDaemon::setThreadCrackSpeed(unsigned char threadId, unsigned char threadType, float rateInM) {
    if (threadId < MAX_SUPPORTED_THREADS) {
        this->threadType[threadId] = threadType;
        this->threadRate[threadId] = rateInM;
    }

}
void CHMultiforcerDisplayDaemon::setWorkunitsTotal(uint32_t newWorkunitsTotal) {
}
void CHMultiforcerDisplayDaemon::setWorkunitsCompleted(uint32_t newWorkunitsCompleted) {
}

void CHMultiforcerDisplayDaemon::addCrackedPassword(char *newCrackedPassword) {
}

void CHMultiforcerDisplayDaemon::addStatusLine(char *newStatusLine) {
}

CHMultiforcerDisplayDaemon::CHMultiforcerDisplayDaemon() {
    memset(this->threadType, 0, MAX_SUPPORTED_THREADS * sizeof(unsigned char));
    memset(this->threadRate, 0, MAX_SUPPORTED_THREADS * sizeof(float));

}

CHMultiforcerDisplayDaemon::~CHMultiforcerDisplayDaemon() {
}


void CHMultiforcerDisplayDaemon::Refresh() {
}

void CHMultiforcerDisplayDaemon::setHashName(char * newHashName) {
}

void CHMultiforcerDisplayDaemon::setSystemMode(int systemMode, char * modeString) {
}
// Increment or decrement the number of network clients
void CHMultiforcerDisplayDaemon::alterNetworkClientCount(int alterBy) {
}
int CHMultiforcerDisplayDaemon::getFreeThreadId() {
    return MAX_SUPPORTED_THREADS - 1;
}
float CHMultiforcerDisplayDaemon::getCurrentCrackRate() {
    float totalSpeed = 0;
    int y;

    // Sum up the valid speeds
    for (y = 0; y < MAX_SUPPORTED_THREADS; y++) {
        if (this->threadType[y]) {
            totalSpeed += this->threadRate[y];
        }
    }
    return totalSpeed;
}

