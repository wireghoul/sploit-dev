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

#include "MFN_Common/MFNDisplayCurses.h"
#include "CH_Common/CHHiresTimer.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "MFN_Common/MFNWorkunitBase.h"
#include "CH_HashFiles/CHHashFileV.h"
#include <curses.h>
#include "Multiforcer_Common/CHCommon.h"

extern struct global_commands global_interface;
extern MFNClassFactory MultiforcerGlobalClassFactory;


MFNDisplayCurses::MFNDisplayCurses() {

    // Start curses mode on init.
    initscr();
    // Turn off the cursor.  No point for it.
    curs_set(0);

    // Get the current X, Y dimensions
    getmaxyx(stdscr, this->consoleDimY, this->consoleDimX);

    // No delay on inputs - just continue if no key is pressed.
    nodelay(stdscr,true);
    
    // Do not echo characters to the screen.
    noecho();

    this->pause_win = NULL;

    this->systemModeString.clear();
    this->systemModeString = "Mode: Standalone";
    this->systemMode = SYSTEM_MODE_STANDALONE;
    this->frameworkRedrawNeeded = 0;

    this->DrawFramework();
    
    this->TotalHashes = 0;
    this->CrackedHashes = 0;
    this->PasswordLen = 0;

    this->WorkunitsCompleted = 0;
    this->WorkunitsTotal = 0;
    
    this->DisplayTimer.start();
    this->lastTimeUpdated = 0;

    this->numberOfNetworkClients = 0;
    
    this->HashFileClass = MultiforcerGlobalClassFactory.getHashfileClass();
    this->WorkunitClass = MultiforcerGlobalClassFactory.getWorkunitClass();

    this->Refresh();
}

void MFNDisplayCurses::DrawFramework() {
    int x, y;

    clear();
    
    for (x = 0; x < this->consoleDimX; x++) {
        mvaddch(0,x,'-');
        mvaddch((this->consoleDimY - 1),x,'-');
    }
    for (y = 0; y < this->consoleDimY; y++) {
        mvaddch(y, 0, '|');
        mvaddch(y,(this->consoleDimX - 1),'|');
    }

    for (y = 3; y < (this->consoleDimY - 3); y++) {
        mvaddch(y, 27, '|');
        mvaddch(y, (this->consoleDimX - 25), '|');

    }

    mvaddch(0, 0, '+');
    mvaddch(0,this->consoleDimX - 1, '+');
    mvaddch(this->consoleDimY - 1,0,'+');
    mvaddch(this->consoleDimY - 1,this->consoleDimX - 1,'+');

    mvprintw(1, ((this->consoleDimX / 2) - (strlen(MFNProgramTitle) / 2)), "%s", MFNProgramTitle);
    mvprintw(2, ((this->consoleDimX / 2) - (this->systemModeString.length() / 2)), this->systemModeString.c_str());

    mvprintw(1, 1, "'p' to pause");
    mvprintw(1, (this->consoleDimX - 13), "'q' to quit");

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

    mvprintw(3, ((this->consoleDimX / 2) - (15 / 2)), "Passwords Found");
    
}

MFNDisplayCurses::~MFNDisplayCurses() {

    int StatusScrollerPrint, y;
    endwin();			/* End curses mode		  */
/*/
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
    }*/
}



void MFNDisplayCurses::Refresh() {
    int x, y;
    float totalSpeed = 0;
    char keyPressed;
    int currentY, currentX;

    int hours, minutes, seconds;

    uint64_t CrackingTimeSeconds;
    
    // Check to see if it's been long enough that we can update.
    if ((this->DisplayTimer.getElapsedTime() - this->lastTimeUpdated) < 0.5) {
        return;
    }
    
    // Attempt to lock these in order.
    this->statusUpdateMutexBoost.lock();
    this->displayUpdateMutex.lock();

    // Update some status variables.
    this->WorkunitsCompleted = this->WorkunitClass->GetNumberOfCompletedWorkunits();
    this->WorkunitsTotal = this->WorkunitClass->GetNumberOfWorkunits();
    this->CrackedHashes = this->HashFileClass->GetCrackedHashCount();
    this->TotalHashes = this->HashFileClass->GetTotalHashCount();
    
    // Check for window resize or a console redraw request.
    getmaxyx(stdscr, currentY, currentX);
    if ((currentY != this->consoleDimY) || (currentX != this->consoleDimX) || (this->frameworkRedrawNeeded)) {
        this->consoleDimX = currentX;
        this->consoleDimY = currentY;
        clear();
        this->DrawFramework();
    }

    CrackingTimeSeconds = (uint64_t)this->DisplayTimer.getElapsedTime();

    seconds = CrackingTimeSeconds % 60;
    minutes = (CrackingTimeSeconds / 60) % 60;
    hours = CrackingTimeSeconds / 3600;

    mvprintw(3, 17, "%9s", this->HashName.c_str());
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
    for (y = 0; y < this->threadType.size(); y++) {
        // Clear the thread display
        mvprintw(4 + y, (this->consoleDimX - 23), "                    ");
        
        // If the thread is set, print it.
        if (this->threadType[y] == GPU_THREAD) {
            mvprintw(4 + y, (this->consoleDimX - 23), "%2d: GPU: %s/s", y, this->getConvertedRateString(this->threadRate[y]).c_str());
            totalSpeed += this->threadRate[y];
        } else if (this->threadType[y] == CPU_THREAD) {
            mvprintw(4 + y, (this->consoleDimX - 23), "%2d: CPU: %s/s", y, this->getConvertedRateString(this->threadRate[y]).c_str());
            totalSpeed += this->threadRate[y];
        } else if (this->threadType[y] == NETWORK_HOST) {
            // Network thread - should be from the remote host.
            mvprintw(4 + y, (this->consoleDimX - 23), "%2d: NET: %s/s", y, this->getConvertedRateString(this->threadRate[y]).c_str());
            totalSpeed += this->threadRate[y];
        }
    }
    mvprintw((this->consoleDimY - 3), (this->consoleDimX - 23), "TOTAL: %s/s  ", this->getConvertedRateString(totalSpeed).c_str());

    // Print the password scroller
    {
        std::list<std::string>::iterator passwordIt;
        int passwordsPrinted = 0;
        for (passwordIt = this->FoundPasswords.begin(); 
                passwordIt != this->FoundPasswords.end(); passwordIt++) {
            // If we have printed enough, break out of the loop.
            if (passwordsPrinted >= (this->consoleDimY - 7)) {
                break;
            }
            // Clear the old row.
            for (x = 0; x < (this->consoleDimX - 55); x++) {
                mvprintw(passwordsPrinted + 5, x + 30, " ");
            }
            mvprintw(passwordsPrinted + 5, 30, "%s", passwordIt->substr(0, (this->consoleDimX - 55)).c_str());
            passwordsPrinted++;
        }
    }
    
    // Status scroller
    {
        std::list<std::string>::iterator statusIt;
        int statusesPrinted = 0;
        for (statusIt = this->StatusScroller.begin(); 
                statusIt != this->StatusScroller.end(); statusIt++) {
            // If we have printed enough, break out of the loop.
            if (statusesPrinted >= (this->consoleDimY - 13)) {
                break;
            }
            // Clear the old row.
            mvprintw(statusesPrinted + 11, 2, "                         ");
            
            mvprintw(statusesPrinted + 11, 2, "%s", statusIt->substr(0, 25).c_str());
            statusesPrinted++;
        }
    }

    // Input handling.  Get the current key.
    keyPressed = getch();
    // If 'p', pause until another key is pressed.
    if (keyPressed == 'p') {
        global_interface.pause = 1;
        // Create a new window.
        // dimY, dimX, startY, startX
        this->pause_win = newwin(4, 20, (this->consoleDimY / 2) - 2, (this->consoleDimX / 2) - 10);
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
    // Actually redraw things.
    refresh();
    
    this->lastTimeUpdated = this->DisplayTimer.getElapsedTime();
    
    this->statusUpdateMutexBoost.unlock();
    this->displayUpdateMutex.unlock();
}
void MFNDisplayCurses::setHashName(std::string newHashName) {
    this->statusUpdateMutexBoost.lock();
    this->HashName = newHashName;
    this->statusUpdateMutexBoost.unlock();
    this->Refresh();
}

void MFNDisplayCurses::setPasswordLen(uint16_t newPasswordLength) {
    this->PasswordLen = newPasswordLength;
    this->Refresh();
}

void MFNDisplayCurses::addCrackedPassword(std::vector<uint8_t> newFoundPassword) {
    this->statusUpdateMutexBoost.lock();
    // Convert the password into a string.
    char buffer[256];
    memset(buffer, 0, sizeof(buffer));
    memcpy(buffer, &newFoundPassword[0], newFoundPassword.size());
    
    std::string crackedPassword(buffer);
    this->FoundPasswords.push_front(crackedPassword);
    this->statusUpdateMutexBoost.unlock();
    this->Refresh();
}

void MFNDisplayCurses::addStatusLine(std::string newStatusLine) {
    this->statusUpdateMutexBoost.lock();
    this->StatusScroller.push_front(newStatusLine);
    this->frameworkRedrawNeeded = 1;
    this->statusUpdateMutexBoost.unlock();
    this->Refresh();
}

void MFNDisplayCurses::addStatusLine(char * newStatusLine) {
    std::string statusLine(newStatusLine);
    this->addStatusLine(statusLine);
}

void MFNDisplayCurses::setSystemMode(int systemMode, std::string modeString) {
    this->statusUpdateMutexBoost.lock();
    this->systemModeString = modeString;
    this->statusUpdateMutexBoost.unlock();
    this->Refresh();
}

void MFNDisplayCurses::alterNetworkClientCount(int networkClientCount) {
    this->statusUpdateMutexBoost.lock();
    this->numberOfNetworkClients += networkClientCount;
    this->statusUpdateMutexBoost.unlock();
    this->Refresh();
}

#if UNIT_TEST

#include <unistd.h>

int main() {
    MFNDisplayCurses Display;
    
    Display.Refresh();
    
    sleep(3);
    
    
}
#endif