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

#include "MFN_Common/MFNNetworkCommon.h"

std::string getMFNNetworkErrorString(uint32_t errorCode) {
    switch(errorCode) {
        case RPC_ERROR_MISSING_REQUIRED_FIELDS:
            return std::string("Error: RPC missing required fields.");
            break;
        case RPC_ERROR_INVALID_PROTOBUF_DATA:
            return std::string("Error: Invalid protobuf data.");
            break;
        case RPC_ERROR_UNKNOWN_ERROR:
            return std::string("Error: Unknown/other error.");
            break;
        case RPC_ERROR_FUNCTION_NOT_IMPLEMENTED:
            return std::string("Error: Function not implemented.");
            break;
        case RPC_ERROR_HASH_NOT_PRESENT:
            return std::string("Error: Hash not present.");
            break;
        default:
            return std::string("Unknown error code.");
            break;
    }
}

