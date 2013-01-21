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

#include "GRT_Common/GRTCrackDisplayCurses.h"
#include "GRT_Common/GRTCommon.h"

#define GRTDISPLAY_UNIT_TEST 0

// Little utility to remove newlines...
void chomp(char *s) {
    while(*s && *s != '\n' && *s != '\r') s++;
    *s = 0;
}


struct global_commands global_interface;


void GRTCrackDisplayCurses::setTotalHashes(uint64_t newTotalHashes) {
    this->HashesTotal = newTotalHashes;
    this->Refresh();
}
void GRTCrackDisplayCurses::setCrackedHashes(uint64_t newCrackedHashes) {
    this->HashesCracked = newCrackedHashes;
    this->Refresh();
}
void GRTCrackDisplayCurses::addCrackedHashes(uint64_t newHashes) {
    this->HashesCracked += newHashes;
    // If the total cracked hashes is greater than the number we need and
    // we are supposed to exit, do so.
    if (global_interface.exit_after_count &&
            (this->HashesCracked >= global_interface.exit_after_count)) {
        global_interface.exit = 1;
    }
}

void GRTCrackDisplayCurses::setThreadCrackSpeed(unsigned char threadId, unsigned char threadType, float rateInM) {
    // If this is a valid thread ID, set the type and rate.
    // No idea why someone would change the rate, but they can.
    if (threadId < MAX_SUPPORTED_THREADS) {
        this->threadType[threadId] = threadType;
        this->threadRate[threadId] = rateInM;
    }
    this->Refresh();
}
void GRTCrackDisplayCurses::setWorkunitsTotal(uint32_t newWorkunitsTotal) {
    this->WorkunitsTotal = newWorkunitsTotal;
    this->Refresh();
}
void GRTCrackDisplayCurses::setWorkunitsCompleted(uint32_t newWorkunitsCompleted) {
    this->WorkunitsCompleted = newWorkunitsCompleted;
    this->Refresh();
}

void GRTCrackDisplayCurses::setTotalTables(uint32_t newTablesTotal) {
    this->TablesTotal = newTablesTotal;
}

void GRTCrackDisplayCurses::setCurrentTableNumber(uint32_t newTableCurrent) {
    this->TableCurrent = newTableCurrent;
}

void GRTCrackDisplayCurses::addCrackedPassword(char *newCrackedPassword) {
    strncpy(this->PasswordsToDisplay[this->PasswordScrollerIndex],
            newCrackedPassword, DISPLAY_PASSWORD_SCROLLER_WIDTH - 1);
    // Increment the index, wrap if needed
    this->PasswordScrollerIndex++;
    this->PasswordScrollerIndex %= DISPLAY_PASSWORD_SCROLLER_HEIGHT;
    this->Refresh();
}

void GRTCrackDisplayCurses::addStatusLine(char *newStatusLine) {
#if USE_BOOST_THREADS
    this->statusLineMutexBoost.lock();
#else
    pthread_mutex_lock(&this->statusLineMutex);
#endif
    // Strip the newline if present.
    chomp(newStatusLine);
    strncpy(this->StatusToDisplay[this->StatusScrollerIndex],
            newStatusLine, STATUS_SCROLLER_WIDTH - 1);
    // Increment the index, wrap if needed
    this->StatusScrollerIndex++;
    this->StatusScrollerIndex %= STATUS_SCROLLER_HEIGHT;
#if USE_BOOST_THREADS
    this->statusLineMutexBoost.unlock();
#else
    pthread_mutex_unlock(&this->statusLineMutex);
#endif
    this->Refresh();
}

GRTCrackDisplayCurses::GRTCrackDisplayCurses() {

    initscr();			/* Start curses mode 		  */
    this->cursesEnabled = 1;
    curs_set(0); // Turn off cursor

    // Get the current X, Y dimensions
    getmaxyx(stdscr, this->currentMaxY, this->currentMaxX);

    // No delay on inputs - just continue if no key is pressed.
    nodelay(stdscr,true);
    // Do not echo characters to the screen.
    noecho();

    this->pause_win = NULL;

#if !USE_BOOST_THREADS
    pthread_mutexattr_init(&this->displayUpdateMutexAttr);
    pthread_mutexattr_init(&this->statusLineMutexAttr);
    pthread_mutex_init(&this->displayUpdateMutex, &this->displayUpdateMutexAttr);
    pthread_mutex_init(&this->statusLineMutex, &this->statusLineMutexAttr);
#endif

    this->DrawFramework();

    this->HashesTotal = 0;
    this->HashesCracked = 0;
    this->TablesTotal = 0;
    this->TableCurrent = 0;

    this->PercentDone = 0;

    this->WorkunitsCompleted = 0;
    this->WorkunitsTotal = 0;

    this->CrackingTimeStart = time(NULL);

    // Disable and zero all threads
    memset(this->threadType, 0, MAX_SUPPORTED_THREADS * sizeof(unsigned char));
    memset(this->threadRate, 0, MAX_SUPPORTED_THREADS * sizeof(float));
    memset(this->PasswordsToDisplay, 0, DISPLAY_PASSWORD_SCROLLER_WIDTH * DISPLAY_PASSWORD_SCROLLER_HEIGHT);
    this->PasswordScrollerIndex = 0;
    memset(this->StatusToDisplay, 0, STATUS_SCROLLER_WIDTH * STATUS_SCROLLER_HEIGHT);
    this->StatusScrollerIndex = 0;
    memset(this->HashName, 0, MAX_HASH_STRING_LENGTH);
    memset(this->systemStageString, 0, STAGE_LENGTH_CHARS);
    memset(this->TableFilename, 0, MAX_FILENAME_LENGTH);
    memset(this->threadFractionDone, 0, MAX_SUPPORTED_THREADS * sizeof(float));

    this->Refresh();
}

void GRTCrackDisplayCurses::DrawFramework() {
    int x, y;

    clear();

    for (x = 0; x < this->currentMaxX; x++) {
        mvaddch(0,x,'-');
        mvaddch((this->currentMaxY - 1),x,'-');
    }
    for (y = 0; y < this->currentMaxY; y++) {
        mvaddch(y, 0, '|');
        mvaddch(y,(this->currentMaxX - 1),'|');
    }

    for (y = 5; y < (this->currentMaxY - 2); y++) {
        mvaddch(y, 27, '|');
        mvaddch(y, (this->currentMaxX - 25), '|');

    }

    mvaddch(0, 0, '+');
    mvaddch(0,this->currentMaxX - 1, '+');
    mvaddch(this->currentMaxY - 1,0,'+');
    mvaddch(this->currentMaxY - 1,this->currentMaxX - 1,'+');

    mvprintw(1, ((this->currentMaxX / 2) - (strlen(programTitle) / 2)), "%s", programTitle);

    mvprintw(1, 1, "'p' to pause");
    mvprintw(1, (this->currentMaxX - 13), "'q' to quit");

    mvprintw(5, 1, "Hash type     :");
    mvprintw(6, 1, "Tables        :");
    mvprintw(7, 1, "Total hashes  :");
    mvprintw(8, 1, "Cracked hashes:");
    mvprintw(9, 1, "Total time    :");
    mvprintw(10, 1, "WUs: ");


    mvprintw(11, 1, "--------------------------");

    mvprintw(5, ((this->currentMaxX / 2) - (15 / 2)), "Passwords Found");

}

void GRTCrackDisplayCurses::endCursesMode() {
    if (this->cursesEnabled) {
        endwin();
        this->cursesEnabled = 0;
    }
}

GRTCrackDisplayCurses::~GRTCrackDisplayCurses() {

    int StatusScrollerPrint, y;
    int hours, minutes, seconds;
    uint64_t CrackingTimeSeconds;
    
    if (this->cursesEnabled) {
        endwin();
        this->cursesEnabled = 0;
    }

    // print the last few status lines
    StatusScrollerPrint = this->StatusScrollerIndex - 1;
    for (y = 0; y < STATUS_SCROLLER_HEIGHT; y++) {
        if (StatusScrollerPrint < 0) {
            StatusScrollerPrint += STATUS_SCROLLER_HEIGHT;
        }
        if (strlen(this->StatusToDisplay[StatusScrollerPrint])) {
            printf("%s\n", this->StatusToDisplay[StatusScrollerPrint]);
        }
        StatusScrollerPrint--;
    }
    
    // Print total execution time.
    CrackingTimeSeconds = time(NULL) - this->CrackingTimeStart;

    seconds = CrackingTimeSeconds % 60;
    minutes = (CrackingTimeSeconds / 60) % 60;
    hours = CrackingTimeSeconds / 3600;
    
    printf("\nTotal execution time: %02d:%02d:%02d\n\n", hours, minutes, seconds);
}

void GRTCrackDisplayCurses::Refresh() {
    int x, y, PasswordScrollerPrint, StatusScrollerPrint;
    float totalSpeed = 0;
    char keyPressed;
    int currentY, currentX;

    int hours, minutes, seconds;

    uint64_t CrackingTimeSeconds;
#if USE_BOOST_THREADS
    this->displayUpdateMutexBoost.lock();
#else
    pthread_mutex_lock(&this->displayUpdateMutex);
#endif

    // Check for window resize
    getmaxyx(stdscr, currentY, currentX);
    if ((currentY != this->currentMaxY) || (currentX != this->currentMaxX)) {
        this->currentMaxX = currentX;
        this->currentMaxY = currentY;
        clear();
        this->DrawFramework();
    }

    CrackingTimeSeconds = time(NULL) - this->CrackingTimeStart;

    seconds = CrackingTimeSeconds % 60;
    minutes = (CrackingTimeSeconds / 60) % 60;
    hours = CrackingTimeSeconds / 3600;

    // Current table filename: Clear, then print.
    for (x = 1; x < this->currentMaxX - 1; x++) {
        mvprintw(3, x, " ");
    }
    mvprintw(3, 1, "Current file: %s", this->TableFilename);

    // Print the second status line
    mvprintw(2, 1, "(%01d/3) %s: %0.1f%%     ", this->systemStage, this->systemStageString, this->PercentDone);

    // Percentage bar
    mvprintw(2, 26, "|");
    mvprintw(2, 77, "|");
    for (x = 27; x < 77; x++) {
        mvprintw(2, x, " ");
    }
    for (x = 0; x < (int)(this->PercentDone / 2); x++) {
        mvprintw(2, x + 27, "=");
    }

    mvprintw(5, 17, "%9s", this->HashName);
    mvprintw(6, 17, "%3d / %3d", this->TableCurrent, this->TablesTotal);
    mvprintw(7, 17, "%9lu", this->HashesTotal);
    mvprintw(8, 17, "%9lu", this->HashesCracked);
    mvprintw(9, 17, "%02d:%02d:%02d", hours, minutes, seconds);
    mvprintw(10, 6, "%u/%u (%0.1f%%)   ", this->WorkunitsCompleted, this->WorkunitsTotal,
        ((100.0 * (float)this->WorkunitsCompleted) / (float)this->WorkunitsTotal));

    // Print out the speed statuses
    for (y = 0; y < MAX_SUPPORTED_THREADS; y++) {
        // Clear the thread display
        mvprintw(5 + y, (this->currentMaxX - 23), "                    ");

        // If the thread is set, print it.
        if (this->threadType[y] == GPU_THREAD) {
            mvprintw(5 + y, (this->currentMaxX - 23), "%2d: GPU: %0.1f M/s", y,  this->threadRate[y]);
            totalSpeed += this->threadRate[y];
        } else if (this->threadType[y] == CPU_THREAD && this->systemStage != GRT_CRACK_SEARCH) {
            mvprintw(5 + y, (this->currentMaxX - 23), "%2d: CPU: %0.1f M/s", y,  this->threadRate[y]);
            totalSpeed += this->threadRate[y];
        } else if (this->threadType[y] == CPU_THREAD && this->systemStage == GRT_CRACK_SEARCH) {
            mvprintw(5 + y, (this->currentMaxX - 23), "%2d: DSK: %0.1f H/s", y,  this->threadRate[y]);
            totalSpeed += this->threadRate[y];
        } else if (this->threadType[y] == NETWORK_HOST) {
            // Network thread - should be from the remote host.
            mvprintw(5 + y, (this->currentMaxX - 23), "%2d: NET: %0.1f M/s", y,  this->threadRate[y]);
            totalSpeed += this->threadRate[y];
        }
    }
    
    if (this->systemStage == GRT_CRACK_SEARCH) {
        mvprintw((this->currentMaxY - 3), (this->currentMaxX - 23), "TOTAL: %0.1f H/s  ", totalSpeed);
    } else {
        mvprintw((this->currentMaxY - 3), (this->currentMaxX - 23), "TOTAL: %0.1f M/s  ", totalSpeed);
    }

    // Print the password scroller
    // print height - 7
    PasswordScrollerPrint = this->PasswordScrollerIndex - 1;
    for (y = 0; y < (this->currentMaxY - 8); y++) {
        if (PasswordScrollerPrint < 0) {
            PasswordScrollerPrint += DISPLAY_PASSWORD_SCROLLER_HEIGHT;
        }
        // Clear the current row...
        for (x = 0; x < (this->currentMaxX - 55); x++) {
            mvprintw(y + 6, x + 30, " ");
        }
        mvprintw(y + 6, 30, "%s", this->PasswordsToDisplay[PasswordScrollerPrint]);
        PasswordScrollerPrint--;

    }

    // Status scroller
    // Print height - 12 options.
    StatusScrollerPrint = this->StatusScrollerIndex - 1;

    for (y = 0; y < (this->currentMaxY - 14); y++) {
        if (StatusScrollerPrint < 0) {
            StatusScrollerPrint += STATUS_SCROLLER_HEIGHT;
        }
        mvprintw(y + 12, 2, "                         ");
        mvprintw(y + 12, 2, "%s", this->StatusToDisplay[StatusScrollerPrint]);
        StatusScrollerPrint--;
    }

    // Input handling.  Get the current key.
    keyPressed = getch();
    // If 'p', pause until another key is pressed.
    if (keyPressed == 'p') {
        global_interface.pause = 1;
        // Create a new window.
        // dimY, dimX, startY, startX
        this->pause_win = newwin(4, 20, (this->currentMaxY / 2) - 2, (this->currentMaxX / 2) - 10);
        // Box it so it's visible.
        box(this->pause_win, 0 , 0);
        mvwprintw(this->pause_win, 1, 7, "PAUSED");
        mvwprintw(this->pause_win, 2, 3, "Press any key");

        // Display it.
        wrefresh(this->pause_win);
        nodelay(stdscr,false);
        getch();
        nodelay(stdscr,true);
        global_interface.pause = 0;
    }
    // If 'q', quit.
    if (keyPressed == 'q') {
        global_interface.exit = 1;
    }

    refresh();
#if USE_BOOST_THREADS
    this->displayUpdateMutexBoost.unlock();
#else
    pthread_mutex_unlock(&this->displayUpdateMutex);
#endif
}

void GRTCrackDisplayCurses::setHashName(char * newHashName) {
    strcpy(this->HashName, newHashName);
}



float GRTCrackDisplayCurses::getCurrentCrackRate() {
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

void GRTCrackDisplayCurses::setSystemStage(int newSystemStage) {
    this->systemStage = newSystemStage;
    this->PercentDone = 0;
    switch(newSystemStage) {
        case GRT_CRACK_CHGEN:
            sprintf(this->systemStageString, "CH Gen");
            break;
        case GRT_CRACK_SEARCH:
            sprintf(this->systemStageString, "Search");
            break;
        case GRT_CRACK_REGEN:
            sprintf(this->systemStageString, "Regen ");
            break;
        default:
            sprintf(this->systemStageString, "UNKOWN");
            break;
    }
    // Clear out any old display cruft.
    this->DrawFramework();
    this->Refresh();
}

void GRTCrackDisplayCurses::setStagePercent(float newPercentDone) {
    this->PercentDone = newPercentDone;
    this->Refresh();
}

void GRTCrackDisplayCurses::setTableFilename(const char *newTableFilename) {
    sprintf(this->TableFilename, "%s", newTableFilename);
}

void GRTCrackDisplayCurses::setThreadFractionDone(unsigned char threadId, float fractionDone) {
    int i;
    // For figuring out how many workunits are effectively done, counting fractions.
    float totalUnitsDone;

    this->threadFractionDone[threadId] = fractionDone;

    // Some maffs!
    totalUnitsDone = (float)this->WorkunitsCompleted;
    for (i = 0; i < MAX_SUPPORTED_THREADS; i++) {
        totalUnitsDone += this->threadFractionDone[i];
    }
    this->PercentDone = 100.0 * (totalUnitsDone / (float)this->WorkunitsTotal);
    this->Refresh();
}




#if GRTDISPLAY_UNIT_TEST
int main() {
    GRTCrackDisplayCurses *Display;
    int i;

    char testStrings[1024];

    Display = new GRTCrackDisplayCurses();

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