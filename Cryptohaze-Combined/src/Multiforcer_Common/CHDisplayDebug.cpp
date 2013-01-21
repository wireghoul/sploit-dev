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

#include "Multiforcer_Common/CHDisplayDebug.h"
#include "Multiforcer_Common/CHCommon.h"
#include "Multiforcer_Common/CHHashes.h"

extern struct global_commands global_interface;

// Little utility to remove newlines...
void debugchomp(char *s) {
    while(*s && *s != '\n' && *s != '\r') s++;
    *s = 0;
}


void CHMultiforcerDisplayDebug::setTotalHashes(uint64_t newTotalHashes) {
    printf("setTotalHashes: %ld\n", newTotalHashes);
}
void CHMultiforcerDisplayDebug::setCrackedHashes(uint64_t newCrackedHashes) {
    printf("setCrackedHashes: %ld\n", newCrackedHashes);
}
void CHMultiforcerDisplayDebug::addCrackedHashes(uint64_t newHashes) {
    printf("addCrackedHashes: %ld\n", newHashes);
}

void CHMultiforcerDisplayDebug::setPasswordLen(int setPasswordLen) {
    printf("setPasswordLen: %ld\n", setPasswordLen);
}
void CHMultiforcerDisplayDebug::setThreadCrackSpeed(unsigned char threadId, unsigned char threadType, float rateInM) {
    printf("Thread ID %d, type %d, speed %f\n", threadId, threadType, rateInM);
}
void CHMultiforcerDisplayDebug::setWorkunitsTotal(uint32_t newWorkunitsTotal) {
    printf("setWorkunitsTotal: %ld\n", newWorkunitsTotal);
}
void CHMultiforcerDisplayDebug::setWorkunitsCompleted(uint32_t newWorkunitsCompleted) {
    printf("setWorkunitsCompleted: %ld\n", newWorkunitsCompleted);
}

void CHMultiforcerDisplayDebug::addCrackedPassword(char *newCrackedPassword) {
    printf("New cracked password!  '%s'\n", newCrackedPassword);
}

void CHMultiforcerDisplayDebug::addStatusLine(char *newStatusLine) {
#if USE_BOOST_THREADS
    this->mutex2Boost.lock();
#else
    pthread_mutex_lock(&this->mutex2);
#endif
    // Strip the newline if present.
    debugchomp(newStatusLine);
    printf("New status: '%s'\n", newStatusLine);
#if USE_BOOST_THREADS
    this->mutex2Boost.unlock();
#else
    pthread_mutex_unlock(&this->mutex2);
#endif
    this->Refresh();

}

CHMultiforcerDisplayDebug::CHMultiforcerDisplayDebug() {
#if !USE_BOOST_THREADS
    pthread_mutexattr_init(&this->mutex2attr);
    pthread_mutex_init(&this->mutex2, &this->mutex2attr);
#endif
}

CHMultiforcerDisplayDebug::~CHMultiforcerDisplayDebug() {
}


void CHMultiforcerDisplayDebug::Refresh() {
}

void CHMultiforcerDisplayDebug::setHashName(char * newHashName) {
    printf("setHashName: %s\n", newHashName);
}

void CHMultiforcerDisplayDebug::setSystemMode(int systemMode, char * modeString) {
    printf("Setting mode to %d (%s)\n", systemMode, modeString);
}
// Increment or decrement the number of network clients
void CHMultiforcerDisplayDebug::alterNetworkClientCount(int alterBy) {
    printf("Adding %d to network clients.\n", alterBy);
}
int CHMultiforcerDisplayDebug::getFreeThreadId() {
    printf("Get Free Thread ID...\n");
    return MAX_SUPPORTED_THREADS - 1;
}
float CHMultiforcerDisplayDebug::getCurrentCrackRate() {
    printf("Current crack rate request.\n");
    return 345.5;
}

