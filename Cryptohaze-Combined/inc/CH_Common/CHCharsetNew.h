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

/*
 * This file defines a charset class that will be used in all the Cryptohaze
 * tools going forward.  It implements the charset as a vector of vectors, not
 * a big long ugly character array with harcoded constant offsets into it.
 *
 * The charset class also supports adding charset values as hex values,
 * including null bytes.  This will probably require new network versions
 * to handle, but for now the class will have the ability to export the "old"
 * charset array of bytes to functions as need it.  When receiving this data,
 * assume that the data is a null terminated string.  Other algorithms such as
 * SL3 will handle this by defining charsets as needed.
 */

#ifndef _CHCHARSETNEW_H
#define _CHCHARSETNEW_H

// Legacy size for multiforcer network support.
#define MAX_CHARSET_LENGTH 128
#define MAX_PASSWORD_LEN 48


#include <stdint.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "MFN_Common/MFNCharsetNew.pb.h"

class CHCharsetNew {
private:
    // This contains the current charset to use.
    std::vector<std::vector<uint8_t> > Charset;
    
    MFNCharsetNewProtobuf Protobuf;

public:
    /**
     * Reads a charset from a newline-deliniated file.
     *
     * This function takes a string of a filename and will read the charset
     * from the file.  The resulting charset is stored in the internal variable.
     * This reads the full charset from the file, including blank lines as
     * present, though they are likely errors.  Returns false if there is an
     * error such as a not-present file or an empty file, else returns true.
     * There is no line length limit - this will be handled by calling code
     * as needed.
     *
     * @param filename The charset file to read
     * @return True if the read is successful, false if there is a failure.
     */
    uint8_t readCharsetFromFile(std::string charsetFilename);
    
    /**
     * Returns the currently loaded charset.
     *
     * @return The currently loaded charset, which may be empty.
     */
    std::vector<std::vector<uint8_t> > getCharset() {
        return this->Charset;
    }

    /**
     * Clears the currently loaded charset.
     */
    void clearCharset();

    /**
     * Returns the number of positions loaded into the charset.
     *
     * This function returns the number of character positions loaded into the
     * current charset.
     *
     * @return Number of character positions loaded into the charset.
     */
    uint16_t getCharsetNumberElements() {
        return (uint16_t)this->Charset.size();
    }

    /**
     * Returns the length of the charset in the specified position.
     *
     * This function returns the number of elements in the specified character
     * position.  If beyond the current length of the charset, return 0.
     *
     * @param position The character position to return the length of.
     * @return The length of the charset in the specified position, or 0.
     */
    uint16_t getCharsetLength(uint16_t position);

    /**
     * Returns the password space size for the given passwordLength.
     *
     * This function returns the number of possible passwords for the given
     * password length, based on the currently loaded charset.  This is
     * determined by the standard password space algorithm of
     * (passspace = charsetlength[0] * charsetlength[1] * charsetlength[2]...)
     * If it is bigger than a uint64, it doesn't matter, because the user
     * is insane.  For now.  Give it a decade and this will matter!
     * If one of the positions is of length 0, the password space is 0.
     *
     * @param passwordLength The password length to determine size for.
     * @return The size of the password space.
     */
    uint64_t getPasswordSpaceSize(uint16_t passwordLength);

    /**
     * Export the charset as a char array, null filled.
     * 
     * This is a legacy function to export the charset as a character array
     * of MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN bytes.  This is used for network
     * support among other things.  The array MUST be null-filled where not
     * filled with charset data.  This function uses 'new' to create the array.
     *
     * @return A new'd uint8_t array containing the full charset.
     */
    uint8_t *exportLegacyCharset();

    /**
     * Loads the charset file with a legacy charset array from the network.
     *
     * This function takes a legacy charset as created above and loads it into
     * the current charset.  The array passed in will be of size
     * MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN bytes and will get loaded in as
     * though it is null terminated strings.  Non-null bytes will get put
     * into the appropriate position.
     *
     * @param remoteCharset The remote charset to be loaded.
     */
    void loadRemoteCharsetIntoCharset(uint8_t *remoteCharset);

    /**
     * Adds a hex character of the specified value at the given position.
     *
     * This function allows the user to add an arbitrary hex character into
     * the specified charset offset.  This will be added to the end of whatever
     * is present at that location (push_back ftw here).  If the position is
     * beyond the current size of the charset, the charset vector is increased
     * with null length positions to the currently specified position before
     * adding the character.
     *
     * @param charsetPosition The charset position to add a character to.
     * @param characterValue The value of the character to add.
     */
    void addCharacterAtPosition(uint16_t charsetPosition, uint8_t characterValue);
    
    /**
     * Export a charset to a remote system using a protobuf.
     * 
     * This function exports the currently loaded charset data to a remote
     * system as a protobuf.  The data is stored in the provided structure,
     * and is passed over the network.
     * 
     * @param exportData Pointer to the string to load with data... 
     */
    void ExportCharsetToRemoteSystem(std::string * exportData);
    
    /**
     * Import a charset from a remote system using a protobuf.
     * 
     * This function imports the specified protobuf into the charset class,
     * replacing all other data (it clears the class entirely).
     * 
     * @param remoteData A string containing the protobuf to import.
     */
    void ImportCharsetFromRemoteSystem(std::string &remoteData);

};


#endif
