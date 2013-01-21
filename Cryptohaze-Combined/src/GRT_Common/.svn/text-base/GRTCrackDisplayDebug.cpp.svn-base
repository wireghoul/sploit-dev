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

#include "GRT_Common/GRTCrackDisplayDebug.h"
#include "GRT_Common/GRTCommon.h"

#define GRTDISPLAYDEBUG_UNIT_TEST 0

extern struct global_commands global_interface;


void GRTCrackDisplayDebug::setTotalHashes(uint64_t newTotalHashes) {
    printf("DisplayDebug: setTotalHashes(%lu)\n", newTotalHashes);
}
void GRTCrackDisplayDebug::setCrackedHashes(uint64_t newCrackedHashes) {
    printf("DisplayDebug: setCrackedHashes(%lu)\n", newCrackedHashes);
}
void GRTCrackDisplayDebug::addCrackedHashes(uint64_t newHashes) {
    printf("DisplayDebug: addCrackedHashes(%lu)\n", newHashes);
}

void GRTCrackDisplayDebug::setThreadCrackSpeed(unsigned char threadId, unsigned char threadType, float rateInM) {
    printf("DisplayDebug: setThreadCrackSpeed(threadId=%d, threadType=%d, rateInM=%0.2f)\n", threadId, threadType, rateInM);
}
void GRTCrackDisplayDebug::setWorkunitsTotal(uint32_t newWorkunitsTotal) {
    this->WorkunitsTotal = newWorkunitsTotal;
    printf("DisplayDebug: setWorkunitsTotal(%lu)\n", newWorkunitsTotal);
}
void GRTCrackDisplayDebug::setWorkunitsCompleted(uint32_t newWorkunitsCompleted) {
    this->WorkunitsCompleted = newWorkunitsCompleted;
    printf("DisplayDebug: setWorkunitsCompleted(%lu)\n", newWorkunitsCompleted);
}

void GRTCrackDisplayDebug::setTotalTables(uint32_t newTablesTotal) {
    printf("DisplayDebug: setTotalTables(%lu)\n", newTablesTotal);
}

void GRTCrackDisplayDebug::setCurrentTableNumber(uint32_t newTableCurrent) {
    printf("DisplayDebug: setCurrentTableNumber(%lu)\n", newTableCurrent);
}

void GRTCrackDisplayDebug::addCrackedPassword(char *newCrackedPassword) {
    printf("DisplayDebug: addCrackedPassword(%s)\n", newCrackedPassword);
}

void GRTCrackDisplayDebug::addStatusLine(char *addStatusLine) {
    printf("DisplayDebug: addStatusLine(%s)\n", addStatusLine);
}

GRTCrackDisplayDebug::GRTCrackDisplayDebug() {
    printf("DisplayDebug: GRTCrackDisplayDebug()\n");
    this->WorkunitsCompleted = 0;
    this->WorkunitsTotal = 0;
    this->PercentDone = 0.0;

    memset(this->threadFractionDone, 0, MAX_SUPPORTED_THREADS * sizeof(float));
}

GRTCrackDisplayDebug::~GRTCrackDisplayDebug() {
    printf("DisplayDebug: ~GRTCrackDisplayDebug()\n");
}

void GRTCrackDisplayDebug::Refresh() {
    printf("DisplayDebug: Refresh()\n");
}

void GRTCrackDisplayDebug::setHashName(char * newHashName) {
    printf("DisplayDebug: setHashName(%s)\n", newHashName);
}

float GRTCrackDisplayDebug::getCurrentCrackRate() {
    printf("DisplayDebug: getCurrentCrackRate()\n");
    return 1234.56;
}

void GRTCrackDisplayDebug::setSystemStage(int newSystemStage) {
    printf("DisplayDebug: setSystemStage(%d)\n", newSystemStage);
}

void GRTCrackDisplayDebug::setStagePercent(float newPercentDone) {
    printf("DisplayDebug: setStagePercent(%0.2f)\n", newPercentDone);
}

void GRTCrackDisplayDebug::setTableFilename(const char *newTableFilename) {
    printf("DisplayDebug: setTableFilename(%s)\n", newTableFilename);
}

void GRTCrackDisplayDebug::setThreadFractionDone(unsigned char threadId, float fractionDone) {
    int i;
    // For figuring out how many workunits are effectively done, counting fractions.
    float totalUnitsDone;
    
    printf("DisplayDebug: setThreadFractionDone(threadId=%d, fractionDone=%0.2f)\n", threadId, fractionDone);

    this->threadFractionDone[threadId] = fractionDone;

    // Some maffs!
    totalUnitsDone = (float)this->WorkunitsCompleted;
    for (i = 0; i < MAX_SUPPORTED_THREADS_DEBUG; i++) {
        totalUnitsDone += this->threadFractionDone[i];
    }
    printf("DisplayDebug: totalUnitsDone: %0.2f\n", totalUnitsDone);

    this->PercentDone = 100.0 * (totalUnitsDone / (float)this->WorkunitsTotal);
    printf("DisplayDebug: PercentDone: %0.2f\n", this->PercentDone);
}



#if GRTDISPLAYDEBUG_UNIT_TEST
int main() {
    GRTCrackDisplayDebug *Display;
    int i;

    char testStrings[1024];

    Display = new GRTCrackDisplayDebug();

    Display->setHashName("NTLM");
    Display->setTotalTables(999);
    Display->setCurrentTableNumber(4);

    Display->setWorkunitsTotal(17);
    Display->setWorkunitsCompleted(5);

    sprintf(testStrings, "NTLM-len8-idx0-chr95-cl1000-sd3351347057-0-v2.part");
    Display->setTableFilename(testStrings);



    for (i = 0; i < 16; i++) {
        Display->setThreadCrackSpeed(i, GPU_THREAD, 1234.5);
        sprintf(testStrings, "Status Test %d\n", i);
        Display->addStatusLine(testStrings);
        sprintf(testStrings, "Password%d", i);
        Display->addCrackedPassword(testStrings);
    }

    Display->setSystemStage(GRT_CRACK_CHGEN);
    for (i = 0; i <= 100; i++) {
        Display->setStagePercent((float)i);
        usleep(100000);
    }
    Display->setSystemStage(GRT_CRACK_SEARCH);
    for (i = 0; i <= 100; i++) {
        Display->setStagePercent((float)i);
        usleep(100000);
    }
    Display->setSystemStage(GRT_CRACK_REGEN);
    for (i = 0; i <= 100; i++) {
        Display->setStagePercent((float)i);
        usleep(100000);
    }


    sleep(15);

    delete Display;
}
#endif