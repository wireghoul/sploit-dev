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

#ifndef __MFNDISPLAY_DEBUG_H
#define __MFNDISPLAY_DEBUG_H

#include "MFN_Common/MFNDisplay.h"

class CHHashFileV;
class MFNWorkunitBase;

class MFNDisplayDebug : public MFNDisplay {
    protected:
        CHHashFileV *HashFileClass;
        MFNWorkunitBase *WorkunitClass;
    public:
        MFNDisplayDebug();
        virtual void Refresh();

        virtual void setHashName(std::string newHashName);
        virtual void setPasswordLen(uint16_t newPasswordLength);
        virtual void addCrackedPassword(std::vector<uint8_t> newFoundPassword);
        virtual void addStatusLine(std::string newStatusLine);
        virtual void addStatusLine(char * newStatusLine);
        // Sets the system mode: Standalone, network server, network client.
        virtual void setSystemMode(int systemMode, std::string modeString);
        // Add or subtract from the number of connected clients
        virtual void alterNetworkClientCount(int networkClientCount);
};


#endif