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

// Network client class for the CH Multiforcer

#ifndef __CHNETWORKCLIENT_H__
#define __CHNETWORKCLIENT_H__


// Forward define
class CHNetworkClient;


// For the various structures
#include "Multiforcer_Common/CHNetworkCommon.h"
#include "CH_Common/CHCharset.h"
#include "CH_Common/CHWorkunitBase.h"
#include "CH_Common/CHWorkunitNetwork.h"
#include "Multiforcer_Common/CHDisplayNcurses.h"

using boost::asio::ip::tcp;

class CHNetworkClient {
public:
    // Attempts to create an instance connecting to the given host.
    // Exits if this does not happen.  Then attempts to get the general
    // info and store this locally.
    CHNetworkClient(char *hostname, uint16_t port);

    void loadCharsetWithData(CHCharset *Charset);
    void loadHashlistWithData(CHHashFileTypes *Hashlist);

    // Updates the general information about the remote tasks.
    void updateGeneralInfo();

    // Return a struct with the general remote info.
    void provideGeneralInfo(CHMultiforcerNetworkGeneral *generalInfo);

    struct CHWorkunitRobustElement getNextNetworkWorkunit();
    int submitNetworkWorkunit(struct CHWorkunitRobustElement Workunit, uint32_t FoundPasswords);

    int reportNetworkFoundPassword(unsigned char *Hash, unsigned char *Password);

    void setDisplay(CHDisplay *);


protected:
    CHMultiforcerNetworkGeneral GeneralInfo;

    int CurrentPasswordLength;


    // Network related things
    boost::asio::io_service io_service;
    tcp::socket *socket;

    CHDisplay *Display;



};


#endif