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

#include <CH_Common/CHHashImplementationNTLM.h>

std::vector<uint8_t> CHHashImplementationNTLM::hashData(
    const std::vector<uint8_t> &rawData) {
    
    // Do a forward NTLM hash.
    return std::vector<uint8_t>();   
}

void CHHashImplementationNTLM::prepareHash(int passLength,
        std::vector<uint8_t> &rawHash) {
    
}

void CHHashImplementationNTLM::postProcessHash(int passLength,
        std::vector<uint8_t> &rawHash) {
    
}

    
    