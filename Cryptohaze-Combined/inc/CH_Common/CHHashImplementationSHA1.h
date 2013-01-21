/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
Copyright (C) 2012  Bitweasil (http://www.cryptohaze.com/)

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
 *
 * @section DESCRIPTION
 *
 * This class implements basic SHA1 operations, used in a variety of places
 * throughout the projects.
 * 
 * Most of the SHA1 code can be pulled from elsewhere in the project.  Please
 * ensure that the forward hash function works for all lengths!
 */

#ifndef __CHHASHIMPLEMENTATIONSHA1_H__
#define __CHHASHIMPLEMENTATIONSHA1_H__

#include <CH_Common/CHHashImplementation.h>

class CHHashImplementationSHA1 : public CHHashImplementation {
public:
    std::vector<uint8_t> hashData(const std::vector<uint8_t> &rawData);
    void prepareHash(int passLength, std::vector<uint8_t> &rawHash);
    void postProcessHash(int passLength, std::vector<uint8_t> &rawHash);
};

#endif