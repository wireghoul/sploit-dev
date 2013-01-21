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
 * This file implements the hash loading for phpass hashes - base64 encoded
 * hashes with the '$H$ or '$P' prefix.  The prefix is '$X$N', where N is the 
 * number of iterations, encoded in the funky base64 encoding they use.
 * 
 * Salt is 8 bytes.
 * 
 */

#ifndef _CHHASHFILEPHPASS_H
#define _CHHASHFILEPHPASS_H


#define PHPBB_MAGIC_BYTES "$H$"
#define PHPASS_MAGIC_BYTES "$P$"

#include "CH_HashFiles/CHHashFileSalted.h"

class CHHashFilePhpass : public CHHashFileSalted {
protected:

    virtual void parseFileLine(std::string fileLine, size_t lineNumber);

public:
    /**
     * Calls the salted constructor with the 16 byte length after decoding.
     */
    CHHashFilePhpass();
};


#endif