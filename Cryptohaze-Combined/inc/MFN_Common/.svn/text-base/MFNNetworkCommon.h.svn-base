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

/**
 * This is a set of common defines for the MFN network classes.  These specify
 * the RPC request and response actions/codes.
 */

#ifndef __MFNNETWORKCOMMON_H__
#define __MFNNETWORKCOMMON_H__

#include <string>
#include <stdint.h>

/**
 * Primary network protocol version.  This will be changed on major changes
 * to the network protocol that break existing functionality.
 */
#define MFN_NETWORK_SERVER_VERSION 4

/**
 * RPC request and respone codes.  These are uint32 values.
 *
 * Requests are prefixed with 0x1000, responses with 0x2000.
 * 
 * Note that the system_cracking_rate field is always optional in RPC requests,
 * but should be set on all RPC requests for best communication with the user.
 */

/**
 * General information request for a client that has just connected to the
 * server or needs new information.
 */
#define RPC_REQUEST_GENERAL_INFO 0x1001
#define RPC_RESPONSE_GENERAL_INFO 0x2001

/**
 * Request a list of the uncracked hashes.  This returns a protobuf from the
 * hash file class and packs it as-is into the data stream.
 */
#define RPC_REQUEST_UNCRACKED_HASHES 0x1002
#define RPC_RESPONSE_UNCRACKED_HASHES 0x2002

/**
 * Request a list of salts for uncracked hashes.  This is less likely to be
 * used frequently, but can be used to reduce effort when a large list is being
 * cracked.  This ONLY updates the salts - though in most cases, requesting the
 * uncracked hashes again would work nicely.
 */
#define RPC_REQUEST_UNCRACKED_SALTS 0x1003
#define RPC_RESPONSE_UNCRACKED_SALTS 0x2003

/**
 * Request the current charset in use.  This returns a protobuf from the charset
 * class file.
 */
#define RPC_REQUEST_CHARSET 0x1004
#define RPC_RESPONSE_CHARSET 0x2004

/**
 * Request a workunit (or number of workunits) from the server.  If this field
 * is set, the number_workunits_requested field MUST be set.
 */
#define RPC_REQUEST_GET_WORKUNITS 0x1005
#define RPC_RESPONSE_GET_WORKUNITS 0x2005

/**
 * Submit a found password/hash pair to the server.  If this is used, the 
 * found_password_value and found_password_hash fields MUST be set.
 */
#define RPC_REQUEST_SUBMIT_PASSWORD 0x1006
#define RPC_RESPONSE_SUBMIT_PASSWORD 0x2006

/**
 * Submit a completed workunit back to the server.  The submitted_workunit_id
 * field must be set.  Optionally, the workunit can be packed up in the 
 * data field as a protobuf.
 */
#define RPC_REQUEST_SUBMIT_WORKUNITS 0x1007
#define RPC_RESPONSE_SUBMIT_WORKUNITS 0x2007

/**
 * Submit the system cracking rate.  This is to be used if another message has
 * not been sent in a while.  system_cracking_rate must be set!
 */
#define RPC_REQUEST_SUBMIT_RATE 0x1008
#define RPC_RESPONSE_SUBMIT_RATE 0x2008

/**
 * Cancel a workunit that has not been completed.  This may be needed if a list
 * of workunits is for the wrong password length.
 */
#define RPC_REQUEST_CANCEL_WORKUNITS 0x1009
#define RPC_RESPONSE_CANCEL_WORKUNITS 0x2009


/**
 * Error RPC responses.  These come from the server back to the client if
 * something is badly wrong.  They all have an 0x4000 prefix.
 */
#define RPC_ERROR_MISSING_REQUIRED_FIELDS 0x4001
#define RPC_ERROR_INVALID_PROTOBUF_DATA 0x4002
#define RPC_ERROR_UNKNOWN_ERROR 0x4003
#define RPC_ERROR_FUNCTION_NOT_IMPLEMENTED 0x4004
#define RPC_ERROR_HASH_NOT_PRESENT 0x4005
#define RPC_ERROR_SERVER_DISCONNECT 0x4006


/**
 * Default information for the network classes
 */

// <3
#define MFN_NETWORK_DEFAULT_PORT 12410

// Number of IO threads to run for the network server
#define MFN_MAX_IO_THREADS 10

/**
 * Given the error code, return a string corresponding to the error.
 * 
 * @param errorCode The error code reported
 * @return A string consisting of a description of the error code.
 */
std::string getMFNNetworkErrorString(uint32_t errorCode);

#endif
