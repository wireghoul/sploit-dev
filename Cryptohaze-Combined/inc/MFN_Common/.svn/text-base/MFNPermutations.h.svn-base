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
 * This class handles loading permutations for the Multiforcer.  This is used to
 * enable a "table attack" on the GPUs, for trying all possible mutations of a
 * word provided to the system.
 *
 * This class is mostly designed for handling standard 8-bit ASCII permutations,
 * but could be extended to support unicode if needed.
 */

#ifndef __MFNPERMUTATIONS_H__
#define __MFNPERMUTATIONS_H__

#include <string>
#include <vector>
#include <stdint.h>


class MFNPermutations {
public:
    MFNPermutations();
    ~MFNPermutations();

    /**
     * Loads a permutation from a file, one per line, based on the initial
     * character.  Returns true if success, false on failure.
     *
     * @param permutationFilename The path to the permutation file to load.
     */
    int LoadPermutations(std::string permutationFilename);

    /**
     * Returns the character set (permutations) for a specified character.  If
     * there are no permutations, simply return the character.
     *
     * @param wordCharacter The character to seek the permutation for.
     * @return A vector containing all valid permutations for this character.
     */
    std::vector<uint8_t> GetPermutationsForCharacter(uint8_t wordCharacter) {
        return this->loadedPermutations[wordCharacter];
    }

    /**
     * Returns the number of permutations for a given character.
     */
    uint32_t GetPermutationCountForCharacter(uint8_t wordCharacter) {
        return this->loadedPermutations[wordCharacter].size();
    }

    /**
     * Returns a vector of vectors containing all loaded permutations.
     */
    std::vector<std::vector<uint8_t> > GetAllPermutations() {
        return this->loadedPermutations;
    }

private:
    std::vector<std::vector<uint8_t> > loadedPermutations;
};

#endif