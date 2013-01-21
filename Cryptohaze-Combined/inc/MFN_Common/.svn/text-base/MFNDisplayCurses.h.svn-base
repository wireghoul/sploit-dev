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

#ifndef __MFNDISPLAY_CURSES_H
#define __MFNDISPLAY_CURSES_H

const char MFNProgramTitle[] = "Cryptohaze MFN 1.31";

// Number of passwords and statuses to keep around.
#define PASSWORD_SCROLLER_MAX_SIZE 256
#define STATUS_SCROLLER_MAX_SIZE 256

#include "MFN_Common/MFNDisplay.h"
#include <CH_Common/CHHiresTimer.h>
#include <string>
#include <list>
#include <curses.h>

class CHHashFileV;
class MFNWorkunitBase;

class MFNDisplayCurses : public MFNDisplay {
    protected:
        
        /**
         * Pointers to the hashfile class (for cracked/total count), and the
         * workunit class (for percent done status).
         */
        CHHashFileV *HashFileClass;
        MFNWorkunitBase *WorkunitClass;
        
        /**
         * Timer class for measuring execution time.
         */
        CHHiresTimer DisplayTimer;
        
        /**
         * Last display update time.  We update twice a second, tops.
         */
        double lastTimeUpdated;
        
        /**
         * Contains the current console dimensions.  If these change, a full
         * redraw is triggered.
         */
        int consoleDimX, consoleDimY;

        /**
         * Assorted stats.
         */
        uint64_t TotalHashes;
        uint64_t CrackedHashes;
        int PasswordLen;
        uint32_t WorkunitsTotal;
        uint32_t WorkunitsCompleted;

        /**
         * Lists are used for the found passwords and the status scrollers.
         * This allows us to efficiently add new entries to the front
         * and trim entries off the back if it is too long.  Accesses will
         * be adding new entries to the head, removing old entries beyond
         * the storage limit from the tail, and printing out entries from the
         * head back until we have filled the needed space.  A list should be
         * the most efficient STL structure for this.
         */
        std::list<std::string> FoundPasswords;
        std::list<std::string> StatusScroller;


        // For pausing the program - a sub-window.
        WINDOW *pause_win;

        std::string HashName;
        std::string systemModeString;

        int systemMode;
        int numberOfNetworkClients;
        
        int frameworkRedrawNeeded;
        
        /**
         * Clears the display and draws the entire framework again.
         */
        void DrawFramework();

        
    public:
        MFNDisplayCurses();
        ~MFNDisplayCurses();
        /**
         * Called to refresh the normal display regions.
         */
        virtual void Refresh();

        /**
         * Sets the currently-active hash name.
         * @param newHashName String of the hash name.
         */
        virtual void setHashName(std::string newHashName);
        
        /**
         * Sets the current password length being cracked.
         * @param newPasswordLength New length.
         */
        virtual void setPasswordLen(uint16_t newPasswordLength);
        
        /**
         * Adds a new cracked password.  This is a vector, as that is how the
         * passwords are handled internally.
         * @param newFoundPassword A vector containing the password string.
         */
        virtual void addCrackedPassword(std::vector<uint8_t> newFoundPassword);
        
        /**
         * Adds a new status line to the system.
         * @param newStatusLine std::string or char* status line.
         */
        virtual void addStatusLine(std::string newStatusLine);
        virtual void addStatusLine(char * newStatusLine);
        
        // Sets the system mode: Standalone, network server, network client.
        virtual void setSystemMode(int systemMode, std::string modeString);
        
        // Add or subtract from the number of connected clients
        virtual void alterNetworkClientCount(int networkClientCount);
};


#endif