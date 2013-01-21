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
 * @section DESCRIPTION
 * CHHashFileVSSHA is an implementation of CHHashFileVSalted for the {SSHA} LDAP
 * hash type.  This hash is the string '{SSHA}' followed by base64 encoded data
 * that consists of the binary SHA1 hash, followed by some number of bytes of
 * salt data (typically 8? unsure).  This data gets loaded into the salted
 * hash class and is treated as a generic salted hash type for the rest of the
 * functions and network communication.  This class will treat all data after
 * the initial 20 bytes as a salt.
 */


#ifndef _CHHASHFILEVSSHA_H
#define _CHHASHFILEVSSHA_H

#include "CH_HashFiles/CHHashFileVSalted.h"

class CHHashFileVSSHA : public CHHashFileVSalted {
protected:
    virtual int OutputFoundHashesToFile();

    virtual void PrintHashToHandle(HashSalted &Hash, FILE *stream);

public:
    CHHashFileVSSHA();

    virtual int OpenHashFile(std::string filename);

    virtual void PrintAllFoundHashes();

    virtual int OutputUnfoundHashesToFile(std::string filename);
};


#endif
