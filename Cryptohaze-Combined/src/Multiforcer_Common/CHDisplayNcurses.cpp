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

#include "Multiforcer_Common/CHCommon.h"
#include "Multiforcer_Common/CHHashes.h"
#include "Multiforcer_Common/CHDisplayNcurses.h"

extern struct global_commands global_interface;


// Little utility to remove newlines...
void curseschomp(char *s) {
    while(*s && *s != '\n' && *s != '\r') s++;
    *s = 0;
}


void CHMultiforcerDisplay::setTotalHashes(uint64_t newTotalHashes) {
    this->TotalHashes = newTotalHashes;
    this->Refresh();
}
void CHMultiforcerDisplay::setCrackedHashes(uint64_t newCrackedHashes) {
    this->CrackedHashes = newCrackedHashes;
    this->Refresh();
}
void CHMultiforcerDisplay::addCrackedHashes(uint64_t newHashes) {
    this->CrackedHashes += newHashes;
    // If the total cracked hashes is greater than the number we need and
    // we are supposed to exit, do so.
    if (global_interface.exit_after_count && 
            (this->CrackedHashes >= global_interface.exit_after_count)) {
        global_interface.exit = 1;
    }
}

void CHMultiforcerDisplay::setPasswordLen(int newPasswordLen) {
    this->PasswordLen = newPasswordLen;
    this->Refresh();
}
void CHMultiforcerDisplay::setThreadCrackSpeed(unsigned char threadId, unsigned char threadType, float rateInM) {
    // If this is a valid thread ID, set the type and rate.
    // No idea why someone would change the rate, but they can.
    if (threadId < MAX_SUPPORTED_THREADS) {
        this->threadType[threadId] = threadType;
        this->threadRate[threadId] = rateInM;
    }
    this->Refresh();
}
void CHMultiforcerDisplay::setWorkunitsTotal(uint32_t newWorkunitsTotal) {
    this->WorkunitsTotal = newWorkunitsTotal;
    this->Refresh();
}
void CHMultiforcerDisplay::setWorkunitsCompleted(uint32_t newWorkunitsCompleted) {
    this->WorkunitsCompleted = newWorkunitsCompleted;
    this->Refresh();
}

void CHMultiforcerDisplay::addCrackedPassword(char *newCrackedPassword) {
    strncpy(this->PasswordsToDisplay[this->PasswordScrollerIndex],
            newCrackedPassword, DISPLAY_PASSWORD_SCROLLER_WIDTH - 1);
    // Increment the index, wrap if needed
    this->PasswordScrollerIndex++;
    this->PasswordScrollerIndex %= DISPLAY_PASSWORD_SCROLLER_HEIGHT;
    this->Refresh();
}

void CHMultiforcerDisplay::addStatusLine(char *newStatusLine) {
#if USE_BOOST_THREADS
    this->statusLineMutexBoost.lock();
#else
    pthread_mutex_lock(&this->statusLineMutex);
#endif
    // Strip the newline if present.
    curseschomp(newStatusLine);
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

CHMultiforcerDisplay::CHMultiforcerDisplay() {

    initscr();			/* Start curses mode 		  */
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

    memset(this->systemModeString, 0, sizeof(this->systemModeString));
    sprintf(this->systemModeString, "Mode: Standalone");
    this->systemMode = SYSTEM_MODE_STANDALONE;
    this->redrawModeString = 0;

    this->DrawFramework();
    
    this->TotalHashes = 0;
    this->CrackedHashes = 0;
    this->PasswordLen = 0;

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
    this->numberOfNetworkClients = 0;


    this->Refresh();
}

void CHMultiforcerDisplay::DrawFramework() {
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

    for (y = 3; y < (this->currentMaxY - 3); y++) {
        mvaddch(y, 27, '|');
        mvaddch(y, (this->currentMaxX - 25), '|');

    }

    mvaddch(0, 0, '+');
    mvaddch(0,this->currentMaxX - 1, '+');
    mvaddch(this->currentMaxY - 1,0,'+');
    mvaddch(this->currentMaxY - 1,this->currentMaxX - 1,'+');

    mvprintw(1, ((this->currentMaxX / 2) - (strlen(programTitle) / 2)), "%s", programTitle);
    mvprintw(2, ((this->currentMaxX / 2) - (strlen(this->systemModeString) / 2)), this->systemModeString);

    mvprintw(1, 1, "'p' to pause");
    mvprintw(1, (this->currentMaxX - 13), "'q' to quit");

    // Offset 17 to start text.
    mvprintw(3, 1, "Hash type     :");
    mvprintw(4, 1, "Current PW len:");
    mvprintw(5, 1, "Total hashes  :");
    mvprintw(6, 1, "Cracked hashes:");
    mvprintw(7, 1, "Total time    :");
    if (this->systemMode != SYSTEM_MODE_CLIENT) {
        mvprintw(8, 1, "WUs: ");
    }
    if (this->systemMode == SYSTEM_MODE_SERVER) {
        mvprintw(9, 1, "Net Clients   :");
    }


    mvprintw(10, 1, "--------------------------");

    //mvprintw(9, 11, "Status");


    mvprintw(3, ((this->currentMaxX / 2) - (15 / 2)), "Passwords Found");

}

CHMultiforcerDisplay::~CHMultiforcerDisplay() {

    int StatusScrollerPrint, y;
    endwin();			/* End curses mode		  */

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
}

void CHMultiforcerDisplay::Refresh() {
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

    mvprintw(3, 17, "%9s", this->HashName);
    mvprintw(4, 17, "%9d", this->PasswordLen);
    mvprintw(5, 17, "%9lu", this->TotalHashes);
    mvprintw(6, 17, "%9lu", this->CrackedHashes);
    mvprintw(7, 17, "%02d:%02d:%02d", hours, minutes, seconds);
    if (this->systemMode != SYSTEM_MODE_CLIENT) {
        mvprintw(8, 6, "%u/%u (%0.1f%%)   ", this->WorkunitsCompleted, this->WorkunitsTotal,
            ((100.0 * (float)this->WorkunitsCompleted) / (float)this->WorkunitsTotal));
    }
    if (this->systemMode == SYSTEM_MODE_SERVER) {
        mvprintw(9, 17, "%9d", this->numberOfNetworkClients);
    }

    // Print out the speed statuses
    for (y = 0; y < MAX_SUPPORTED_THREADS; y++) {
        // Clear the thread display
        mvprintw(4 + y, (this->currentMaxX - 23), "                    ");
        
        // If the thread is set, print it.
        if (this->threadType[y] == GPU_THREAD) {
            mvprintw(4 + y, (this->currentMaxX - 23), "%2d: GPU: %0.2fM/s", y,  this->threadRate[y]);
            totalSpeed += this->threadRate[y];
        } else if (this->threadType[y] == CPU_THREAD) {
            mvprintw(4 + y, (this->currentMaxX - 23), "%2d: CPU: %0.2fM/s", y,  this->threadRate[y]);
            totalSpeed += this->threadRate[y];
        } else if (this->threadType[y] == NETWORK_HOST) {
            // Network thread - should be from the remote host.
            mvprintw(4 + y, (this->currentMaxX - 23), "%2d: NET: %0.2fM/s", y,  this->threadRate[y]);
            totalSpeed += this->threadRate[y];
        }
    }
    mvprintw((this->currentMaxY - 3), (this->currentMaxX - 23), "TOTAL: %0.2fM/s  ", totalSpeed);

    // Print the password scroller
    // print height - 7
    PasswordScrollerPrint = this->PasswordScrollerIndex - 1;
    for (y = 0; y < (this->currentMaxY - 7); y++) {
        if (PasswordScrollerPrint < 0) {
            PasswordScrollerPrint += DISPLAY_PASSWORD_SCROLLER_HEIGHT;
        }
        // Clear the current row... 
        for (x = 0; x < (this->currentMaxX - 55); x++) {
            mvprintw(y + 5, x + 30, " ");
        }
        mvprintw(y + 5, 30, "%s", this->PasswordsToDisplay[PasswordScrollerPrint]);
        PasswordScrollerPrint--;

    }
    
    // Status scroller
    // Print height - 12 options.
    StatusScrollerPrint = this->StatusScrollerIndex - 1;

    for (y = 0; y < (this->currentMaxY - 13); y++) {
        if (StatusScrollerPrint < 0) {
            StatusScrollerPrint += STATUS_SCROLLER_HEIGHT;
        }
        mvprintw(y + 11, 2, "                         ");
        mvprintw(y + 11, 2, "%s", this->StatusToDisplay[StatusScrollerPrint]);
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
        global_interface.user_exit = 1;
    }

    
    
    if (this->redrawModeString) {
        mvprintw(2, ((this->currentMaxX / 2) - (strlen(this->systemModeString) / 2)), this->systemModeString);
    }
    refresh();
#if USE_BOOST_THREADS
    this->displayUpdateMutexBoost.unlock();
#else
    pthread_mutex_unlock(&this->displayUpdateMutex);
#endif
}

void CHMultiforcerDisplay::setHashName(char * newHashName) {
    strcpy(this->HashName, newHashName);
}

void CHMultiforcerDisplay::setSystemMode(int systemMode, char *modeString) {
    switch(systemMode) {
        case SYSTEM_MODE_STANDALONE:
            sprintf(this->systemModeString, "Mode: Standalone");
            break;
        case SYSTEM_MODE_SERVER:
            sprintf(this->systemModeString, "Mode: Server (Port %s)", modeString);
            break;
        case SYSTEM_MODE_CLIENT:
            sprintf(this->systemModeString, "Mode: Client (Server %s)", modeString);
            break;
        default:
            break; // Do nothing
    }
    this->systemMode = systemMode;
    this->DrawFramework();
    this->redrawModeString = 1;

}


// Increment or decrement the number of network clients
void CHMultiforcerDisplay::alterNetworkClientCount(int alterBy) {
    this->numberOfNetworkClients += alterBy;
    this->Refresh();
}

// Figure out the next free thread ID, or -1 for "We can't do this."
int CHMultiforcerDisplay::getFreeThreadId() {
    int i;

    for (i = 0; i < MAX_SUPPORTED_THREADS; i++) {
        if (this->threadType[i] == UNUSED_THREAD) {
            return i;
        }
    }
    return -1;
}

float CHMultiforcerDisplay::getCurrentCrackRate() {
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
