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

#ifndef __CHNETWORKCOMMON_H__
#define __CHNETWORKCOMMON_H__

#include "Multiforcer_Common/CHCommon.h"
#include "Multiforcer_CUDA_host/CHCommandLineData.h"
#include "CH_Common/CHWorkunit.h"
#include "CH_Common/CHHashFileTypes.h"
#include "CH_Common/CHCharset.h"
#include "CH_Common/CHWorkunitBase.h"
#include "Multiforcer_Common/CHDisplayNcurses.h"

#define NETWORK_SERVER_VERSION 2


// Actions for the network clients -
// this may involve some cleanup on complete.
// No action: Nothing has been done.
#define NETWORK_ACTION_NONE 0
// General data: The basic "overview structure"
#define NETWORK_ACTION_GENERAL 1
// Send a workunit struct
#define NETWORK_ACTION_WORKUNIT 2
// Send a hash list
#define NETWORK_ACTION_HASHLIST 3
// Send a charest
#define NETWORK_ACTION_CHARSET 4



#define MAX_IO_THREADS 5

class CHNetworkServerInstance;
class CHNetworkServer;

typedef struct classContainerStruct {
        CHCommandLineData *CommandLineData;
        CHCharset *Charset;
        CHWorkunitBase *Workunit;
        CHHashFileTypes *HashFile;
        CHNetworkServer *NetworkServer;
        CHDisplay *Display;
} classContainerStruct;

typedef union CHMultiforcerNetworkGeneral {
    struct {
      uint32_t version; // Version of the protocol/configuration structure
      uint32_t hash_id; // Hash ID value
      uint32_t password_length; // Length of current password being worked on
      uint32_t workunit_size_bits; // Size of the workunits in bits
      uint32_t number_hashes; // Number of hashes in the workunit
      uint32_t charset_max_characters; // Max size of the each charset element
      uint32_t charset_max_length; // Total number of charset fields
    } structure;
    // Reading the structure out as needed.
    unsigned char readbuffer[7 * sizeof(uint32_t)];
}CHMultiforcerNetworkGeneral;

typedef union CHMultiforcerNeworkWorkunitRobust {
    struct CHWorkunitRobustElement WorkunitStruct;
    unsigned char readbuffer[sizeof(struct CHWorkunitRobustElement)];
}CHMultiforcerNeworkWorkunitRobust;


typedef struct CHMultiforcerNetworkUnsaltedHash {
    unsigned char hash[32];
    unsigned char password[MAX_PASSWORD_LEN];
} CHMultiforcerNetworkUnsaltedHash;

typedef union CHMultiforcerNetworkUnsaltedHashTransfer {
    struct CHMultiforcerNetworkUnsaltedHash Hash;
    unsigned char readbuffer[sizeof(struct CHMultiforcerNetworkUnsaltedHash)];
} CHMultiforcerNetworkUnsaltedHashTransfer;

#endif