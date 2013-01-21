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

#include <MFN_OpenCL_host/MFNOpenCLMetaprograms.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

/**
 * The incrementors here are vector focused, and work around the fact that the
 * ATI cards take *forever* entering and exiting new clauses.  So, we do this
 * without clauses except where absolutely needed.
 * 
 * The first position will ALWAYS be incremented, so is done slightly differently
 * from the others - we don't need to check, we always do it.  Things are 
 * hardcoded to be a bit faster.
 * 
 * They look like this (commented):
 *  ================= Start Incrementor =============
 * // Get the first character out of b0, and mask it to a byte.                    
 * lookupIndex = (b0 >> 0) & 0xff;
 * // Look up the position of this character in the reverse lookup for all 
 * // vector offsets (expand as needed).
 * passwordOffsetVector.s0 = sharedReverseCharsetPlainMD5[lookupIndex.s0];
 * // Mask off the first character in b0 to zero it.  We will fill it later.                 
 * b0 &= 0xffffff00;
 * // Increment all values in the offset vector by one.
 * passwordOffsetVector += 1;
 * // If they are >= than the charset length for this position, set them to zero.
 * // This is done by a ternary operator to avoid branching - turns into a 
 * // cmov_logical operator, in theory.
 * passwordOffsetVector = (passwordOffsetVector >= sharedCharsetLengthsPlainMD5[0]) ? 0 : passwordOffsetVector;
 * // Given the new offsets, look up the new character position and build a vector.
 * newPasswordCharacters.s0 = (unsigned int)sharedCharsetPlainMD5[passwordOffsetVector.s0];
 * // Shift the password characters to the right position, if needed.
 * newPasswordCharacters = newPasswordCharacters << 0; // >> Fix formatting in Netbeans
 * // Or the new vector with the existing password to set the new character.
 * b0 |= newPasswordCharacters;
 * 
 * // If the offset for any position was 0, this means that we wrapped at least
 * // one position, and need to continue on to the next position and figure out
 * // the enable mask so we don't increment the wrong positions.
 * if (!passwordOffsetVector.s0 || !passwordOffsetVector.s1 || !passwordOffsetVector.s2 || !passwordOffsetVector.s3) {
 *   // New vector types: maskVector is the mask we use for setting/resetting characters,
 *   // and enableMask is set if the particular vector position is still being
 *   // updated - this is chained so we don't update more than needed.
 *   vector_type maskVector;
 *   vector_type enableMask;
 *   // For this second position, set enableMask based on the previous
 *   // passwordOffsetVector being zero.  Future checks will require checking this
 *   // as well as the current state of enableMask.
 *   enableMask = (!passwordOffsetVector) ? (uint)0x01 : (uint)0x00;
 *   // maskVector is set the the correct position if the vector element is being
 *   // updated, else set to all 1s, so we don't nuke the character in a password
 *   // not being updated.
 *   maskVector = (enableMask > 0) ? 0xffff00ff : 0xffffffff;
 *   // As before, get the character of interest into the lookupIndex.
 *   lookupIndex = (b0 >> 8) & 0xff;
 *   // And, look it up for all vectors.  If this is a per-position charset, this
 *   // will involve adding the correct static offset.
 *   passwordOffsetVector.s0 = sharedReverseCharsetPlainMD5[lookupIndex.s0];
 *   // Increment the positions for all loaded offsets.
 *   passwordOffsetVector++;
 *   // And, check to see if they're beyond the limit, resetting if needed.
 *   passwordOffsetVector = (passwordOffsetVector >= sharedCharsetLengthsPlainMD5[0]) ? 0 : passwordOffsetVector;
 *   // Load the new character for all positions.  Shared memory lookups are less
 *   // expensive than branching/clauses, so we just load it for all of them.
 *   newPasswordCharacters.s0 = (unsigned int)sharedCharsetPlainMD5[passwordOffsetVector.s0];
 *   // Shift by the correct amount to put the character in the right position.
 *   newPasswordCharacters = newPasswordCharacters << 8; // >> format fix
 *   // Mask out characters that are not to be updated.  This will and the 
 *   // unused characters with all 1s, and be a noop.
 *   b0 &= maskVector;
 *   // Or the new characters with the password, after anding the password with
 *   // the bitwise inverse of the maskVector.  The positions that need to be
 *   // updated will be masked out then updated, and the positions that do not
 *   // need to be updated will not get updated.
 *   b0 |= (~maskVector & newPasswordCharacters);
 *   // Determine if the carry needs to be continued.  If the enableMask for
 *   // the position is true, and the passwordOffsetVector was zero (indicating
 *   // that the character wrapped), set the enableMask to 1 and continue, 
 *   // otherwise set it to 0.  The clamp function simply sets a zero
 *   // offsetVector value to 0, and anything else to 1, allowing the use of
 *   // the (1 - clamp) construct.  Otherwise, the compiler complains really 
 *   // hard about signed/unsigned values being mixed and is a pain. 
 *   enableMask = (enableMask && (1 - clamp(passwordOffsetVector, 0, 1))) ? (uint)1 : (uint)0;
 *   // Continue until we're done...                        

 */

/**
 * Set of characters for vector indexing.  This is used for vector width
 * 16 - it's s{0-9,A-F} and this is the easiest way to do this.
 */
static char vectorIndexes[16] = {'0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

/**
    * This struct is used for tracking the "skips" in a charset.  It will be
    * filled with the value that should be checked for (one past the end of
    * the previous character) and the value of the next point to skip to. 
    */
typedef struct charsetSkips {
    uint8_t skipFrom;
    uint8_t skipTo;
} charsetSkips;

/**
 * Gets the bit mask based on the character position.
 * 
 * 0xffffff00 - pos 0
 * 0xffff00ff - pos 1
 * 0xff00ffff - pos 2
 * 0x00ffffff - pos 3
 * 
 */
static std::string getPasswordMaskLE(int characterPosition) {
    switch (characterPosition % 4) {
        case 0:
            return std::string("0xffffff00");
        case 1:
            return std::string("0xffff00ff");
        case 2:
            return std::string("0xff00ffff");
        case 3:
            return std::string("0x00ffffff");
    }
    return "0xffffffff";
}

static std::string getPasswordMaskBE(int characterPosition) {
    switch (characterPosition % 4) {
        case 3:
            return std::string("0xffffff00");
        case 2:
            return std::string("0xffff00ff");
        case 1:
            return std::string("0xff00ffff");
        case 0:
            return std::string("0x00ffffff");
    }
    return "0xffffffff";
}

static inline int getShiftAmountLE(int characterPosition) {
    return (characterPosition % 4) * 8;
}

static inline int getShiftAmountBE(int characterPosition) {
    return (3 - (characterPosition % 4)) * 8;
}

// return b0, b1, b2, ...
static inline std::string getPasswordVariableLE(int characterPosition) {
    char buffer[256];
    sprintf(buffer, "b%d", (characterPosition / 4));
    return std::string(buffer);
}

// I think this is identical - is there any reason to keep it?
static inline std::string getPasswordVariableBE(int characterPosition) {
    char buffer[256];
    sprintf(buffer, "b%d", (characterPosition / 4));
    return std::string(buffer);
}

// Build an "if (variable.s0 || variable.s1 || variable.s2 ... ) {" string.
static std::string buildIfCheck(std::string variableName, int vectorWidth) {
    std::string ifString;
    char buffer[4096];
    
    ifString += "if (";
    
    if (vectorWidth == 1) {
        // No fancy stuff.  Just the variable name.
        sprintf(buffer, "%s", variableName.c_str());
        ifString += buffer;
    } else {
        for (int i = 0; i < vectorWidth; i++) {
            sprintf(buffer, "%s.s%c", variableName.c_str(), vectorIndexes[i]);
            ifString += buffer;
            if (i != (vectorWidth - 1)) {
                ifString += " || ";
            }
        }
    }
    ifString += ") {\\\n";
    
    return ifString;
}

static inline std::string getVectorType(int vectorWidth) {
    char buffer[256];
    if (vectorWidth > 1) {
        sprintf(buffer, "uint%d", vectorWidth);
        return std::string(buffer);
    } else {
        return std::string("uint");
    }
}

static inline std::string getSignedVectorType(int vectorWidth) {
    char buffer[256];
    if (vectorWidth > 1) {
        sprintf(buffer, "int%d", vectorWidth);
        return std::string(buffer);
    } else {
        return std::string("int");
    }
}

// Return a multiplication that handles charset buffer size + character position
static inline std::string getCharsetBufferOffset(int characterPosition, int charsetBufferSize) {
    char buffer[256];
    sprintf(buffer, " (%d * %d) ", characterPosition, charsetBufferSize);
    return std::string(buffer);
}

// Because I don't want to deal with stringstream
static inline std::string getInt(int integer) {
    char buffer[256];
    sprintf(buffer, "%d", integer);
    return std::string(buffer);
}

// Returns the .s0, .s1, ... (or nothing if vectorWidth == 1)
static inline std::string getVectorIdentifier(int element, int vectorWidth) {
    if (vectorWidth == 1) {
        return std::string("");
    } else {
        char buffer[256];
        sprintf(buffer, ".s%c", vectorIndexes[element]);
        return std::string(buffer);
    }
}

// Shorten the function name so it's easier to insert.
#define VI(pos) getVectorIdentifier(pos, vectorWidth).c_str()

std::string MFNOpenCLMetaprograms::makePasswordSingleIncrementorsLE(int passwordLength, 
        int vectorWidth) {
    
    std::string incrementorDefine;
    int vectorPosition;
    int passwordPosition;
    char buffer[1024];
    std::string indent;
    
    // Build the define macro.
    
    incrementorDefine += "#define OpenCLPasswordIncrementorLE(charsetForward, charsetReverse, charsetLengths) { \\\n";
    
    // Put in the code to increment the first position.
    incrementorDefine += "lookupIndex = (b0 >> 0) & 0xff;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[lookupIndex.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "b0 &= 0xffffff00;\\\n";
    incrementorDefine += "passwordOffsetVector += 1;\\\n";
    incrementorDefine += "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[0]) ? 0 : passwordOffsetVector;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[passwordOffsetVector.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "b0 |= newPasswordCharacters;\\\n";
    
    // For each new character position, add the appropriate code.
    for (passwordPosition = 1; passwordPosition < passwordLength; passwordPosition++) {
        
        if (passwordPosition == 1) {
            incrementorDefine += buildIfCheck("!passwordOffsetVector", vectorWidth);
            incrementorDefine += "  " + getVectorType(vectorWidth) + " maskVector;\\\n";
            incrementorDefine += "  " + getVectorType(vectorWidth) + " enableMask;\\\n";
        } else {
            incrementorDefine += (indent + buildIfCheck("enableMask", vectorWidth));
        }
        indent += "  ";
        if (passwordPosition == 1) {
            incrementorDefine += (indent + "enableMask = (!passwordOffsetVector) ? (" + getVectorType(vectorWidth) + ")0x01 : (" + getVectorType(vectorWidth) + ")0x00;\\\n");
        }
        // Set the mask vector appropriately.
        incrementorDefine += (indent + "maskVector = (enableMask > 0) ? (" + getVectorType(vectorWidth) + ")");
        incrementorDefine += getPasswordMaskLE(passwordPosition);
        incrementorDefine += " : (" + getVectorType(vectorWidth) + ")0xffffffff;\\\n";
        
        // Shift the needed character into the low byte.
        sprintf(buffer, "lookupIndex = (%s >> %d) & 0xff;\\\n", 
                getPasswordVariableLE(passwordPosition).c_str(), 
                getShiftAmountLE(passwordPosition));
        incrementorDefine += (indent + buffer);
        
        // Do the reverse lookup, reset to 0, forward lookup
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[lookupIndex.s%c];\\\n",
                    vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        incrementorDefine += (indent + "passwordOffsetVector += 1;\\\n");
        incrementorDefine += (indent + "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[0]) ? 0 : passwordOffsetVector;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[passwordOffsetVector.s%c];\\\n",
                    vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        sprintf(buffer, "newPasswordCharacters = newPasswordCharacters << %d;\\\n", getShiftAmountLE(passwordPosition));
        incrementorDefine += (indent + buffer);
        
        // And with the mask vector to clear the position.
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition) + " &= maskVector;\\\n");
        // And with the negatated mask vector to add the new characters in
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition) + " |= (~maskVector & newPasswordCharacters);\\\n");
        // Set the enableMask.  The compiler *should* remove this on the last loop...
        incrementorDefine += (indent + "enableMask = (enableMask && (1 - clamp(passwordOffsetVector, (uint)0, (uint)1))) ? (" + getVectorType(vectorWidth) + ")1 : (" + getVectorType(vectorWidth) + ")0;\\\n");
    }
    
    for (passwordPosition = 0; passwordPosition < passwordLength; passwordPosition++) {
        incrementorDefine += "} ";
    }
    incrementorDefine += "\n";
    return incrementorDefine;
}

std::string MFNOpenCLMetaprograms::makePasswordSingleIncrementorsBE(int passwordLength, 
        int vectorWidth) {
    
    std::string incrementorDefine;
    int vectorPosition;
    int passwordPosition;
    char buffer[1024];
    std::string indent;
    
    // Build the define macro.
    
    incrementorDefine += "#define OpenCLPasswordIncrementorBE(charsetForward, charsetReverse, charsetLengths) { \\\n";
    
    // Put in the code to increment the first position.
    incrementorDefine += "lookupIndex = (b0 >> 24) & 0xff;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[lookupIndex.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "b0 &= 0x00ffffff;\\\n";
    incrementorDefine += "passwordOffsetVector += 1;\\\n";
    incrementorDefine += "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[0]) ? 0 : passwordOffsetVector;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[passwordOffsetVector.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "newPasswordCharacters = newPasswordCharacters << 24;\\\n";
    incrementorDefine += "b0 |= newPasswordCharacters;\\\n";
    
    // For each new character position, add the appropriate code.
    for (passwordPosition = 1; passwordPosition < passwordLength; passwordPosition++) {
        
        if (passwordPosition == 1) {
            incrementorDefine += buildIfCheck("!passwordOffsetVector", vectorWidth);
            incrementorDefine += "  " + getVectorType(vectorWidth) + " maskVector;\\\n";
            incrementorDefine += "  " + getVectorType(vectorWidth) + " enableMask;\\\n";
        } else {
            incrementorDefine += (indent + buildIfCheck("enableMask", vectorWidth));
        }
        indent += "  ";
        if (passwordPosition == 1) {
            incrementorDefine += (indent + "enableMask = (!passwordOffsetVector) ? (" + getVectorType(vectorWidth) + ")0x01 : (" + getVectorType(vectorWidth) + ")0x00;\\\n");
        }
        // Set the mask vector appropriately.
        incrementorDefine += (indent + "maskVector = (enableMask > 0) ? (" + getVectorType(vectorWidth) + ")");
        incrementorDefine += getPasswordMaskBE(passwordPosition);
        incrementorDefine += " : (" + getVectorType(vectorWidth) + ")0xffffffff;\\\n";
        
        // Shift the needed character into the low byte.
        sprintf(buffer, "lookupIndex = (%s >> %d) & 0xff;\\\n", 
                getPasswordVariableLE(passwordPosition).c_str(), 
                getShiftAmountBE(passwordPosition));
        incrementorDefine += (indent + buffer);
        
        // Do the reverse lookup, reset to 0, forward lookup
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[lookupIndex.s%c];\\\n",
                    vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        incrementorDefine += (indent + "passwordOffsetVector += 1;\\\n");
        incrementorDefine += (indent + "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[0]) ? 0 : passwordOffsetVector;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[passwordOffsetVector.s%c];\\\n",
                    vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        sprintf(buffer, "newPasswordCharacters = newPasswordCharacters << %d;\\\n", getShiftAmountBE(passwordPosition));
        incrementorDefine += (indent + buffer);
        
        // And with the mask vector to clear the position.
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition) + " &= maskVector;\\\n");
        // And with the negatated mask vector to add the new characters in
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition) + " |= (~maskVector & newPasswordCharacters);\\\n");
        // Set the enableMask.  The compiler *should* remove this on the last loop...
        incrementorDefine += (indent + "enableMask = (enableMask && (1 - clamp(passwordOffsetVector, (uint)0, (uint)1))) ? (" + getVectorType(vectorWidth) + ")1 : (" + getVectorType(vectorWidth) + ")0;\\\n");
    }
    
    for (passwordPosition = 0; passwordPosition < passwordLength; passwordPosition++) {
        incrementorDefine += "} ";
    }
    incrementorDefine += "\n";
    return incrementorDefine;
}

std::string MFNOpenCLMetaprograms::makePasswordSingleIncrementorsNTLM(int passwordLength, 
        int vectorWidth) {
    
    std::string incrementorDefine;
    int vectorPosition;
    int passwordPosition;
    char buffer[1024];
    std::string indent;
    
    // Build the define macro.
    
    incrementorDefine += "#define OpenCLPasswordIncrementorNTLM(charsetForward, charsetReverse, charsetLengths) { \\\n";
    
    // Put in the code to increment the first position.
    incrementorDefine += "lookupIndex = (b0 >> 0) & 0xff;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[lookupIndex.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "b0 &= 0xffffff00;\\\n";
    incrementorDefine += "passwordOffsetVector += 1;\\\n";
    incrementorDefine += "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[0]) ? 0 : passwordOffsetVector;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[passwordOffsetVector.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "b0 |= newPasswordCharacters;\\\n";
    
    // For each new character position, add the appropriate code.
    for (passwordPosition = 1; passwordPosition < passwordLength; passwordPosition++) {
        
        if (passwordPosition == 1) {
            incrementorDefine += buildIfCheck("!passwordOffsetVector", vectorWidth);
            incrementorDefine += "  " + getVectorType(vectorWidth) + " maskVector;\\\n";
            incrementorDefine += "  " + getVectorType(vectorWidth) + " enableMask;\\\n";
        } else {
            incrementorDefine += (indent + buildIfCheck("enableMask", vectorWidth));
        }
        indent += "  ";
        if (passwordPosition == 1) {
            incrementorDefine += (indent + "enableMask = (!passwordOffsetVector) ? (" + getVectorType(vectorWidth) + ")0x01 : (" + getVectorType(vectorWidth) + ")0x00;\\\n");
        }
        // Set the mask vector appropriately.
        incrementorDefine += (indent + "maskVector = (enableMask > 0) ? (" + getVectorType(vectorWidth) + ")");
        incrementorDefine += getPasswordMaskLE(passwordPosition * 2);
        incrementorDefine += " : (" + getVectorType(vectorWidth) + ")0xffffffff;\\\n";
        
        // Shift the needed character into the low byte.
        sprintf(buffer, "lookupIndex = (%s >> %d) & 0xff;\\\n", 
                getPasswordVariableLE(passwordPosition * 2).c_str(), 
                getShiftAmountLE(passwordPosition * 2));
        incrementorDefine += (indent + buffer);
        
        // Do the reverse lookup, reset to 0, forward lookup
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[lookupIndex.s%c];\\\n",
                    vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        incrementorDefine += (indent + "passwordOffsetVector += 1;\\\n");
        incrementorDefine += (indent + "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[0]) ? 0 : passwordOffsetVector;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[passwordOffsetVector.s%c];\\\n",
                    vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        sprintf(buffer, "newPasswordCharacters = newPasswordCharacters << %d;\\\n", getShiftAmountLE(passwordPosition * 2));
        incrementorDefine += (indent + buffer);
        
        // And with the mask vector to clear the position.
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition * 2) + " &= maskVector;\\\n");
        // And with the negatated mask vector to add the new characters in
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition * 2) + " |= (~maskVector & newPasswordCharacters);\\\n");
        // Set the enableMask.  The compiler *should* remove this on the last loop...
        incrementorDefine += (indent + "enableMask = (enableMask && (1 - clamp(passwordOffsetVector, (uint)0, (uint)1))) ? (" + getVectorType(vectorWidth) + ")1 : (" + getVectorType(vectorWidth) + ")0;\\\n");
    }
    
    for (passwordPosition = 0; passwordPosition < passwordLength; passwordPosition++) {
        incrementorDefine += "} ";
    }
    incrementorDefine += "\n";
    return incrementorDefine;
}

std::string MFNOpenCLMetaprograms::makePasswordNoMemSingleIncrementorsLE(
        int passwordLength, int vectorWidth, 
        std::vector<std::vector<uint8_t> > charset, int passStride,
        int reverseMacroStep) {
    
    std::vector<uint8_t> cs = charset[0];
    
    
    std::vector<charsetSkips> skips;
    charsetSkips currentSkip;
    
    //printf("Received single charset of length %d\n", cs.size());
    
    //std::sort(cs.begin(), cs.end());
    
    //printf("Looking for skips.\n");
    for (int i = 0; i < (cs.size() - 1); i++) {
        if (cs[i+1] != (cs[i] + 1)) {
            //printf("Skip detected from %d to %d\n", cs[i], cs[i+1]);
            currentSkip.skipFrom = cs[i] + 1;
            currentSkip.skipTo = cs[i+1];
            skips.push_back(currentSkip);
        }
    }
    // Add the final skip in.
    currentSkip.skipFrom = cs[cs.size() - 1] + 1;
    currentSkip.skipTo = cs[0];
    skips.push_back(currentSkip);
    
    /*printf("Resulting skips: \n");
    for (int i = 0; i < skips.size(); i++) {
        printf("%d => %d\n", skips[i].skipFrom, skips[i].skipTo);
    }*/
    
    std::string incrementorDefine;
    int passwordPosition;
    char buffer[1024];
    std::string indent;
    
    // Build the define macro.
    
    incrementorDefine += "#define OpenCLNoMemPasswordIncrementorLE() { \\\n";
    // Variables needed
    incrementorDefine += getVectorType(vectorWidth) + " currentChar;\\\n";
    incrementorDefine += getSignedVectorType(vectorWidth) + " enableMask;\\\n";
    
    // Put in the code to increment the first position.
    incrementorDefine += "currentChar = (b0 >> 0) & 0xff;\\\n";
    incrementorDefine += "currentChar++;\\\n";
    incrementorDefine += "b0 &= 0xffffff00;\\\n";
    for (int i = 0; i < skips.size(); i++) {
        incrementorDefine += "currentChar = (currentChar == " + 
                getInt(skips[i].skipFrom) + ") ? " + 
                getInt(skips[i].skipTo) + " : currentChar;\\\n";
    }
    incrementorDefine += "b0 |= currentChar;\\\n";
    incrementorDefine += "enableMask = (currentChar == " + getInt(cs[0]) + ");\\\n";
    // For each new character position, add the appropriate code.
    for (passwordPosition = 1; passwordPosition < passwordLength; passwordPosition++) {
        if (vectorWidth == 1) {
            // The "any" operator, for whatever reasons, does not work on uint types.
            incrementorDefine += indent + "if (enableMask) {\\\n";
        } else {
            incrementorDefine += indent + "if (any(enableMask)) {\\\n";
        }
        indent += "  ";
        if (passwordPosition == 1) {
            incrementorDefine += "  " + getVectorType(vectorWidth) + " maskVector;\\\n";
        }
        // Set the mask vector appropriately.
        incrementorDefine += (indent + "maskVector = (enableMask) ? (" + getVectorType(vectorWidth) + ")");
        incrementorDefine += getPasswordMaskLE(passwordPosition * passStride);
        incrementorDefine += " : (" + getVectorType(vectorWidth) + ")0xffffffff;\\\n";
        
        // Shift the needed character into the low byte.
        sprintf(buffer, "currentChar = (%s >> %d) & 0xff;\\\n", 
                getPasswordVariableLE(passwordPosition * passStride).c_str(), 
                getShiftAmountLE(passwordPosition * passStride));
        incrementorDefine += (indent + buffer);
        incrementorDefine += (indent + "currentChar++;\\\n");
        for (int i = 0; i < skips.size(); i++) {
            incrementorDefine += indent + "currentChar = (currentChar == " + 
                    getInt(skips[i].skipFrom) + ") ? " + 
                    getInt(skips[i].skipTo) + " : currentChar;\\\n";
        }
        
        // And with the mask vector to clear the position.
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition * passStride) + " &= maskVector;\\\n");
        // And with the negatated mask vector to add the new characters in
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition * passStride) + 
                " |= (~maskVector & (currentChar << " + getInt(getShiftAmountLE(passwordPosition * passStride)) + "));\\\n");
        // If the reverse macro is to be called, call it here.
        if (passwordPosition == reverseMacroStep) {
            incrementorDefine += (indent + "REVERSE();\\\n");
        }
        // Set the enableMask.  The compiler *should* remove this on the last loop...
        incrementorDefine += (indent + "enableMask = (enableMask && (currentChar == " + getInt(cs[0]) + "));\\\n");
    }
    
    for (passwordPosition = 0; passwordPosition < passwordLength; passwordPosition++) {
        incrementorDefine += "} ";
    }
    incrementorDefine += "\n";
    return incrementorDefine;
}


std::string MFNOpenCLMetaprograms::makePasswordNoMemMultipleIncrementorsLE(
        int passwordLength, int vectorWidth, 
        std::vector<std::vector<uint8_t> > charset, int passStride,
        int reverseMacroStep) {
    
    std::vector<std::vector<uint8_t> > cs = charset;
    
    // One for each position.
    std::vector<std::vector<charsetSkips> > skips;

    charsetSkips currentSkip;
    
    //printf("Received multiple charset of length %d\n", cs.size());
    skips.resize(cs.size());
    
    //printf("Looking for skips.\n");
    for (int pos = 0; pos < cs.size(); pos++) {
        //printf("pos %d length: %d\n", pos, cs[pos].size());
        for (int i = 0; i < (cs[pos].size() - 1); i++) {
            
            if (cs[pos][i+1] != (cs[pos][i] + 1)) {
                //printf("Skip detected from %d to %d pos %d\n", cs[pos][i], cs[pos][i+1], pos);
                currentSkip.skipFrom = cs[pos][i] + 1;
                currentSkip.skipTo = cs[pos][i+1];
                skips[pos].push_back(currentSkip);
            }
        }
        // Add the final skip in.
        currentSkip.skipFrom = cs[pos][cs[pos].size() - 1] + 1;
        currentSkip.skipTo = cs[pos][0];
        skips[pos].push_back(currentSkip);
    }
    
    /*printf("Resulting skips: \n");
    for (int pos = 0; pos < cs.size(); pos++) {
        for (int i = 0; i < skips[pos].size(); i++) {
            printf("%d: %d => %d\n", pos, skips[pos][i].skipFrom, skips[pos][i].skipTo);
        }
    }*/
    
    std::string incrementorDefine;
    int passwordPosition;
    char buffer[1024];
    std::string indent;
    
    // Build the define macro.
    
    incrementorDefine += "#define OpenCLNoMemPasswordIncrementorLE() { \\\n";
    // Variables needed
    incrementorDefine += getVectorType(vectorWidth) + " currentChar;\\\n";
    incrementorDefine += getSignedVectorType(vectorWidth) + " enableMask;\\\n";
    
    // Put in the code to increment the first position.
    incrementorDefine += "currentChar = (b0 >> 0) & 0xff;\\\n";
    incrementorDefine += "currentChar++;\\\n";
    incrementorDefine += "b0 &= 0xffffff00;\\\n";
    for (int i = 0; i < skips[0].size(); i++) {
        incrementorDefine += "currentChar = (currentChar == " + 
                getInt(skips[0][i].skipFrom) + ") ? " + 
                getInt(skips[0][i].skipTo) + " : currentChar;\\\n";
    }
    incrementorDefine += "b0 |= currentChar;\\\n";
    incrementorDefine += "enableMask = (currentChar == " + getInt(cs[0][0]) + ");\\\n";
    // For each new character position, add the appropriate code.
    for (passwordPosition = 1; passwordPosition < passwordLength; passwordPosition++) {
        if (vectorWidth == 1) {
            // The "any" operator, for whatever reasons, does not work on uint types.
            incrementorDefine += indent + "if (enableMask) {\\\n";
        } else {
            incrementorDefine += indent + "if (any(enableMask)) {\\\n";
        }
        indent += "  ";
        if (passwordPosition == 1) {
            incrementorDefine += "  " + getVectorType(vectorWidth) + " maskVector;\\\n";
        }
        // Set the mask vector appropriately.
        incrementorDefine += (indent + "maskVector = (enableMask) ? (" + getVectorType(vectorWidth) + ")");
        incrementorDefine += getPasswordMaskLE(passwordPosition * passStride);
        incrementorDefine += " : (" + getVectorType(vectorWidth) + ")0xffffffff;\\\n";
        
        // Shift the needed character into the low byte.
        sprintf(buffer, "currentChar = (%s >> %d) & 0xff;\\\n", 
                getPasswordVariableLE(passwordPosition * passStride).c_str(), 
                getShiftAmountLE(passwordPosition * passStride));
        incrementorDefine += (indent + buffer);
        incrementorDefine += (indent + "currentChar++;\\\n");
        
        for (int i = 0; i < skips[passwordPosition].size(); i++) {
            incrementorDefine += indent + "currentChar = (currentChar == " + 
                    getInt(skips[passwordPosition][i].skipFrom) + ") ? " + 
                    getInt(skips[passwordPosition][i].skipTo) + " : currentChar;\\\n";
        }
        
        // And with the mask vector to clear the position.
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition * passStride) + " &= maskVector;\\\n");
        // And with the negatated mask vector to add the new characters in
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition * passStride) + 
                " |= (~maskVector & (currentChar << " + getInt(getShiftAmountLE(passwordPosition * passStride)) + "));\\\n");
        // If the reverse macro is to be called, call it here.
        if (passwordPosition == reverseMacroStep) {
            incrementorDefine += (indent + "REVERSE();\\\n");
        }
        // Set the enableMask.  The compiler *should* remove this on the last loop...
        incrementorDefine += (indent + "enableMask = (enableMask && (currentChar == " + getInt(cs[passwordPosition][0]) + "));\\\n");
    }
    
    for (passwordPosition = 0; passwordPosition < passwordLength; passwordPosition++) {
        incrementorDefine += "} ";
    }
    incrementorDefine += "\n";
    return incrementorDefine;
}

std::string MFNOpenCLMetaprograms::makePasswordMultipleIncrementorsLE(int passwordLength, int vectorWidth,
        int charsetBufferLength) {
    
    std::string incrementorDefine;
    int vectorPosition;
    int passwordPosition = 0;
    char buffer[1024];
    std::string indent;
    
    // Build the define macro.
    incrementorDefine += "#define OpenCLPasswordIncrementorLE(charsetForward, charsetReverse, charsetLengths) { \\\n";
    
    // Put in the code to increment the first position.
    incrementorDefine += "lookupIndex = (b0 >> 0) & 0xff;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        // Position 0 - no need for an additional offset.
        sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[lookupIndex.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "b0 &= 0xffffff00;\\\n";
    incrementorDefine += "passwordOffsetVector += 1;\\\n";
    // Always position 0 here.
    incrementorDefine += "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[0]) ? 0 : passwordOffsetVector;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        // And again, no offset needed for position 0.
        sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[passwordOffsetVector.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "b0 |= newPasswordCharacters;\\\n";
    
    // For each new character position, add the appropriate code.
    for (passwordPosition = 1; passwordPosition < passwordLength; passwordPosition++) {
        
        if (passwordPosition == 1) {
            incrementorDefine += buildIfCheck("!passwordOffsetVector", vectorWidth);
            incrementorDefine += "  " + getVectorType(vectorWidth) + " maskVector;\\\n";
            incrementorDefine += "  " + getVectorType(vectorWidth) + " enableMask;\\\n";
        } else {
            incrementorDefine += (indent + buildIfCheck("enableMask", vectorWidth));
        }
        indent += "  ";
        if (passwordPosition == 1) {
            incrementorDefine += (indent + "enableMask = (!passwordOffsetVector) ? (" + getVectorType(vectorWidth) + ")0x01 : (" + getVectorType(vectorWidth) + ")0x00;\\\n");
        }
        // Set the mask vector appropriately.
        incrementorDefine += (indent + "maskVector = (enableMask > 0) ? (" + getVectorType(vectorWidth) + ")");
        incrementorDefine += getPasswordMaskLE(passwordPosition);
        incrementorDefine += " : (" + getVectorType(vectorWidth) + ")0xffffffff;\\\n";
        
        // Shift the needed character into the low byte.
        sprintf(buffer, "lookupIndex = (%s >> %d) & 0xff;\\\n", 
                getPasswordVariableLE(passwordPosition).c_str(), 
                getShiftAmountLE(passwordPosition));
        incrementorDefine += (indent + buffer);
        
        // Do the reverse lookup, reset to 0, forward lookup
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[%d + lookupIndex.s%c];\\\n",
                    vectorIndexes[vectorPosition], (passwordPosition * charsetBufferLength), vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        incrementorDefine += (indent + "passwordOffsetVector += 1;\\\n");
        incrementorDefine += (indent + "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[" 
                + getInt(passwordPosition) + "]) ? 0 : passwordOffsetVector;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[%d + passwordOffsetVector.s%c];\\\n",
                    vectorIndexes[vectorPosition], (passwordPosition * charsetBufferLength), vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        sprintf(buffer, "newPasswordCharacters = newPasswordCharacters << %d;\\\n", getShiftAmountLE(passwordPosition));
        incrementorDefine += (indent + buffer);
        
        // And with the mask vector to clear the position.
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition) + " &= maskVector;\\\n");
        // And with the negatated mask vector to add the new characters in
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition) + " |= (~maskVector & newPasswordCharacters);\\\n");
        // Set the enableMask.  The compiler *should* remove this on the last loop...
        incrementorDefine += (indent + "enableMask = (enableMask && (1 - clamp(passwordOffsetVector, (uint)0, (uint)1))) ? (" + getVectorType(vectorWidth) + ")1 : (" + getVectorType(vectorWidth) + ")0;\\\n");
    }
    
    for (passwordPosition = 0; passwordPosition < passwordLength; passwordPosition++) {
        incrementorDefine += "} ";
    }
    incrementorDefine += "\n";
    return incrementorDefine;
}

std::string MFNOpenCLMetaprograms::makePasswordMultipleIncrementorsBE(int passwordLength, int vectorWidth,
        int charsetBufferLength) {
    
    std::string incrementorDefine;
    int vectorPosition;
    int passwordPosition = 0;
    char buffer[1024];
    std::string indent;
    
    // Build the define macro.
    incrementorDefine += "#define OpenCLPasswordIncrementorBE(charsetForward, charsetReverse, charsetLengths) { \\\n";
    
    // Put in the code to increment the first position.
    incrementorDefine += "lookupIndex = (b0 >> 24) & 0xff;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        // Position 0 - no need for an additional offset.
        sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[lookupIndex.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "b0 &= 0x00ffffff;\\\n";
    incrementorDefine += "passwordOffsetVector += 1;\\\n";
    // Always position 0 here.
    incrementorDefine += "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[0]) ? 0 : passwordOffsetVector;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        // And again, no offset needed for position 0.
        sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[passwordOffsetVector.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "newPasswordCharacters = newPasswordCharacters << 24;\\\n";
    incrementorDefine += "b0 |= newPasswordCharacters;\\\n";
    
    // For each new character position, add the appropriate code.
    for (passwordPosition = 1; passwordPosition < passwordLength; passwordPosition++) {
        
        if (passwordPosition == 1) {
            incrementorDefine += buildIfCheck("!passwordOffsetVector", vectorWidth);
            incrementorDefine += "  " + getVectorType(vectorWidth) + " maskVector;\\\n";
            incrementorDefine += "  " + getVectorType(vectorWidth) + " enableMask;\\\n";
        } else {
            incrementorDefine += (indent + buildIfCheck("enableMask", vectorWidth));
        }
        indent += "  ";
        if (passwordPosition == 1) {
            incrementorDefine += (indent + "enableMask = (!passwordOffsetVector) ? (" + getVectorType(vectorWidth) + ")0x01 : (" + getVectorType(vectorWidth) + ")0x00;\\\n");
        }
        // Set the mask vector appropriately.
        incrementorDefine += (indent + "maskVector = (enableMask > 0) ? (" + getVectorType(vectorWidth) + ")");
        incrementorDefine += getPasswordMaskBE(passwordPosition);
        incrementorDefine += " : (" + getVectorType(vectorWidth) + ")0xffffffff;\\\n";
        
        // Shift the needed character into the low byte.
        sprintf(buffer, "lookupIndex = (%s >> %d) & 0xff;\\\n", 
                getPasswordVariableLE(passwordPosition).c_str(), 
                getShiftAmountBE(passwordPosition));
        incrementorDefine += (indent + buffer);
        
        // Do the reverse lookup, reset to 0, forward lookup
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[%d + lookupIndex.s%c];\\\n",
                    vectorIndexes[vectorPosition], (passwordPosition * charsetBufferLength), vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        incrementorDefine += (indent + "passwordOffsetVector += 1;\\\n");
        incrementorDefine += (indent + "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[" 
                + getInt(passwordPosition) + "]) ? 0 : passwordOffsetVector;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[%d + passwordOffsetVector.s%c];\\\n",
                    vectorIndexes[vectorPosition], (passwordPosition * charsetBufferLength), vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        sprintf(buffer, "newPasswordCharacters = newPasswordCharacters << %d;\\\n", getShiftAmountBE(passwordPosition));
        incrementorDefine += (indent + buffer);
        
        // And with the mask vector to clear the position.
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition) + " &= maskVector;\\\n");
        // And with the negatated mask vector to add the new characters in
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition) + " |= (~maskVector & newPasswordCharacters);\\\n");
        // Set the enableMask.  The compiler *should* remove this on the last loop...
        incrementorDefine += (indent + "enableMask = (enableMask && (1 - clamp(passwordOffsetVector, (uint)0, (uint)1))) ? (" + getVectorType(vectorWidth) + ")1 : (" + getVectorType(vectorWidth) + ")0;\\\n");
    }
    
    for (passwordPosition = 0; passwordPosition < passwordLength; passwordPosition++) {
        incrementorDefine += "} ";
    }
    incrementorDefine += "\n";
    return incrementorDefine;
}

std::string MFNOpenCLMetaprograms::makePasswordMultipleIncrementorsNTLM(int passwordLength, int vectorWidth,
        int charsetBufferLength) {
    
    std::string incrementorDefine;
    int vectorPosition;
    int passwordPosition = 0;
    char buffer[1024];
    std::string indent;
    
    // Build the define macro.
    incrementorDefine += "#define OpenCLPasswordIncrementorNTLM(charsetForward, charsetReverse, charsetLengths) { \\\n";
    
    // Put in the code to increment the first position.
    incrementorDefine += "lookupIndex = (b0 >> 0) & 0xff;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        // Position 0 - no need for an additional offset.
        sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[lookupIndex.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "b0 &= 0xffffff00;\\\n";
    incrementorDefine += "passwordOffsetVector += 1;\\\n";
    // Always position 0 here.
    incrementorDefine += "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[0]) ? 0 : passwordOffsetVector;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        // And again, no offset needed for position 0.
        sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[passwordOffsetVector.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        incrementorDefine += buffer;
    }
    incrementorDefine += "b0 |= newPasswordCharacters;\\\n";
    
    // For each new character position, add the appropriate code.
    for (passwordPosition = 1; passwordPosition < passwordLength; passwordPosition++) {
        
        if (passwordPosition == 1) {
            incrementorDefine += buildIfCheck("!passwordOffsetVector", vectorWidth);
            incrementorDefine += "  " + getVectorType(vectorWidth) + " maskVector;\\\n";
            incrementorDefine += "  " + getVectorType(vectorWidth) + " enableMask;\\\n";
        } else {
            incrementorDefine += (indent + buildIfCheck("enableMask", vectorWidth));
        }
        indent += "  ";
        if (passwordPosition == 1) {
            incrementorDefine += (indent + "enableMask = (!passwordOffsetVector) ? (" + getVectorType(vectorWidth) + ")0x01 : (" + getVectorType(vectorWidth) + ")0x00;\\\n");
        }
        // Set the mask vector appropriately.
        incrementorDefine += (indent + "maskVector = (enableMask > 0) ? (" + getVectorType(vectorWidth) + ")");
        incrementorDefine += getPasswordMaskLE(passwordPosition * 2);
        incrementorDefine += " : (" + getVectorType(vectorWidth) + ")0xffffffff;\\\n";
        
        // Shift the needed character into the low byte.
        sprintf(buffer, "lookupIndex = (%s >> %d) & 0xff;\\\n", 
                getPasswordVariableLE(passwordPosition * 2).c_str(), 
                getShiftAmountLE(passwordPosition * 2));
        incrementorDefine += (indent + buffer);
        
        // Do the reverse lookup, reset to 0, forward lookup
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "passwordOffsetVector.s%c = charsetReverse[%d + lookupIndex.s%c];\\\n",
                    vectorIndexes[vectorPosition], (passwordPosition * charsetBufferLength), vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        incrementorDefine += (indent + "passwordOffsetVector += 1;\\\n");
        incrementorDefine += (indent + "passwordOffsetVector = (passwordOffsetVector >= charsetLengths[" 
                + getInt(passwordPosition) + "]) ? 0 : passwordOffsetVector;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "newPasswordCharacters.s%c = (unsigned int)charsetForward[%d + passwordOffsetVector.s%c];\\\n",
                    vectorIndexes[vectorPosition], (passwordPosition * charsetBufferLength), vectorIndexes[vectorPosition]);
            incrementorDefine += (indent + buffer);
        }
        sprintf(buffer, "newPasswordCharacters = newPasswordCharacters << %d;\\\n", getShiftAmountLE(passwordPosition * 2));
        incrementorDefine += (indent + buffer);
        
        // And with the mask vector to clear the position.
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition * 2) + " &= maskVector;\\\n");
        // And with the negatated mask vector to add the new characters in
        incrementorDefine += (indent + getPasswordVariableLE(passwordPosition * 2) + " |= (~maskVector & newPasswordCharacters);\\\n");
        // Set the enableMask.  The compiler *should* remove this on the last loop...
        incrementorDefine += (indent + "enableMask = (enableMask && (1 - clamp(passwordOffsetVector, (uint)0, (uint)1))) ? (" + getVectorType(vectorWidth) + ")1 : (" + getVectorType(vectorWidth) + ")0;\\\n");
    }
    
    for (passwordPosition = 0; passwordPosition < passwordLength; passwordPosition++) {
        incrementorDefine += "} ";
    }
    incrementorDefine += "\n";
    return incrementorDefine;
}

/**
 * Onto the new defines for the password lookup bitmap.  This does as much
 * as possible in parallel, for improved speed and performance.  It, much as
 * the password incrementors does, makes a huge difference.
 * 
 * Looks an awful lot like this:
 * 
 * ========================
 * 
 * // Calculate the lookup index for the 8kb shared bitmap
 * lookupIndex = (a & 0x0000ffff) >> 3;
 * 
 * // Lookup for each vector element the value in the bitmap.
 * lookupResult.s0 = sharedBitmap[lookupIndex.s0];
 * // And shift them to the right bit in the byte.
 * lookupResult = (lookupResult >> (a & 0x00000007)) & 0x00000001;
 * 
 * // Check the bitmap results.  If any are positive, move to the next stage.
 * if (lookupResult.s0 || lookupResult.s1 || lookupResult.s2 || lookupResult.s3) {
 *   // Same thing, different values.
 *   lookupIndex = (a >> 3) & 0x07FFFFFF;
 *   // Lookup in the global bitmap for the "A" variables.  Add the result to the
 *   // existing variable so false hits later don't cause more work.
 *   lookupResult.s0 = lookupResult.s0 & (deviceGlobalBitmapAPlainMD5[lookupIndex.s0] >> (a.s0 & 0x00000007)) & 0x00000001;
 *   if (lookupResult.s0 || lookupResult.s1 || lookupResult.s2 || lookupResult.s3) {
 *     lookupIndex = (b >> 3) & 0x07FFFFFF;
 *     lookupResult.s0 = lookupResult.s0 & (deviceGlobalBitmapBPlainMD5[lookupIndex.s0] >> (b.s0 & 0x00000007)) & 0x00000001;
 *     if (lookupResult.s0 || lookupResult.s1 || lookupResult.s2 || lookupResult.s3) {
 *       lookupIndex = (c >> 3) & 0x07FFFFFF;
 *       lookupResult.s0 = lookupResult.s0 & (deviceGlobalBitmapCPlainMD5[lookupIndex.s0] >> (c.s0 & 0x00000007)) & 0x00000001;
 *       if (lookupResult.s0 || lookupResult.s1 || lookupResult.s2 || lookupResult.s3) {
 *         lookupIndex = (d >> 3) & 0x07FFFFFF;
 *         lookupResult.s0 = lookupResult.s0 & (deviceGlobalBitmapDPlainMD5[lookupIndex.s0] >> (d.s0 & 0x00000007)) & 0x00000001;
 *         // Finally, if a result is still positive, search the list and report
 *         // the found password if it is discovered.
 *         if (lookupResult.s0 > 0) {
 *           CheckPassword128LE(deviceGlobalHashlistAddressPlainMD5, deviceGlobalFoundPasswordsPlainMD5, deviceGlobalFoundPasswordFlagsPlainMD5, numberOfHashesPlainMD5, 0);
 * } } } } }
 */

std::string MFNOpenCLMetaprograms::makeBitmapLookup(int vectorWidth, int numberBigBitmapsFilled, 
        std::string passwordCheckFunction) {

    std::string passcheckDefine;
    int vectorPosition;
    char buffer[1024];
    std::string indent = "  ";
    int i;
    
    // sb: shared bitmap a
    // gb{a-d}: global bitmap a-d
    // dgh: Device Global hashlist
    // dfp: Device Found Passwords
    // dfpf: Device Found Passwords Flags
    // dnh: Device number hashes

    passcheckDefine += "#define OpenCLPasswordCheck128(sb, gba, gbb, gbc, gbd, dgh, dfp, dfpf, dnh) { \\\n";
    
    passcheckDefine += "  lookupIndex = (a & SHARED_BITMAP_MASK) >> 3;\\\n";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        sprintf(buffer, "  lookupResult.s%c = sb[lookupIndex.s%c];\\\n",
                vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
        passcheckDefine += buffer;
    }
    passcheckDefine += "  lookupResult = (lookupResult >> (a & 0x00000007)) & 0x00000001;\\\n";
    if (numberBigBitmapsFilled >= 1) {
        passcheckDefine += (indent + buildIfCheck("lookupResult", vectorWidth));
        indent += "  ";
        passcheckDefine += (indent + "lookupIndex = (a >> 3) & 0x07FFFFFF;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "lookupResult.s%c = lookupResult.s%c & (gba[lookupIndex.s%c] >> (a.s%c & 0x00000007)) & 0x00000001;\\\n",
                    vectorIndexes[vectorPosition], vectorIndexes[vectorPosition], vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
            passcheckDefine += (indent + buffer);
        }
    }
    if (numberBigBitmapsFilled >= 2) {
        passcheckDefine += (indent + buildIfCheck("lookupResult", vectorWidth));
        indent += "  ";
        passcheckDefine += (indent + "lookupIndex = (b >> 3) & 0x07FFFFFF;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "lookupResult.s%c = lookupResult.s%c & (gbb[lookupIndex.s%c] >> (b.s%c & 0x00000007)) & 0x00000001;\\\n",
                    vectorIndexes[vectorPosition], vectorIndexes[vectorPosition], vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
            passcheckDefine += (indent + buffer);
        }
    }
    if (numberBigBitmapsFilled >= 3) {
        passcheckDefine += (indent + buildIfCheck("lookupResult", vectorWidth));
        indent += "  ";
        passcheckDefine += (indent + "lookupIndex = (c >> 3) & 0x07FFFFFF;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "lookupResult.s%c = lookupResult.s%c & (gbc[lookupIndex.s%c] >> (c.s%c & 0x00000007)) & 0x00000001;\\\n",
                    vectorIndexes[vectorPosition], vectorIndexes[vectorPosition], vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
            passcheckDefine += (indent + buffer);
        }
    }
    if (numberBigBitmapsFilled >= 4) {
        passcheckDefine += (indent + buildIfCheck("lookupResult", vectorWidth));
        indent += "  ";
        passcheckDefine += (indent + "lookupIndex = (d >> 3) & 0x07FFFFFF;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "lookupResult.s%c = lookupResult.s%c & (gbd[lookupIndex.s%c] >> (d.s%c & 0x00000007)) & 0x00000001;\\\n",
                    vectorIndexes[vectorPosition], vectorIndexes[vectorPosition], vectorIndexes[vectorPosition], vectorIndexes[vectorPosition]);
            passcheckDefine += (indent + buffer);
        }
    }
    // Put in the actual useful checks if we've passed everything.
    passcheckDefine += (indent + buildIfCheck("lookupResult", vectorWidth));
    indent += "  ";
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        sprintf(buffer, "if (lookupResult.s%c > 0) {\\\n",
                vectorIndexes[vectorPosition]);
        passcheckDefine += (indent + buffer);
        sprintf(buffer, "  %s(dgh, dfp, dfpf, dnh, %c);\\\n", 
                passwordCheckFunction.c_str(), vectorIndexes[vectorPosition]);
        passcheckDefine += (indent + buffer);
        passcheckDefine += (indent + "}\\\n");
    }
    
    // Close out the parens.
    for (i = 0; i < (numberBigBitmapsFilled + 2); i++) {
        passcheckDefine += "} ";
    }
    passcheckDefine += "\n";
        
    return passcheckDefine;
}



std::string MFNOpenCLMetaprograms::makeBitmapLookupEarlyOut(int vectorWidth,  std::string passwordCheckFunction,
    char bitmap_0_letter, int bitmap_0_is_present, 
    char bitmap_1_letter, int bitmap_1_is_present, std::string bitmap_1_calculation_string,
    char bitmap_2_letter, int bitmap_2_is_present, std::string bitmap_2_calculation_string,
    char bitmap_3_letter, int bitmap_3_is_present, std::string bitmap_3_calculation_string,
    char use_l2_bitmap, std::string lookupFunctionName, std::string ifFoundRunMacro) {
        
    std::string passcheckDefine;
    int vectorPosition;
    char buffer[1024];
    std::string indent = "  ";

    // sb: shared bitmap a
    // gb{a-d}: global bitmap a-d
    // dgh: Device Global hashlist
    // dfp: Device Found Passwords
    // dfpf: Device Found Passwords Flags
    // dnh: Device number hashes
    // gbl2: Global bitmap L2 cache sized

    
    // First bitmap will always be provided.
    passcheckDefine += "#define " + lookupFunctionName + "(sb, gba, gbb, gbc, gbd, dgh, dfp, dfpf, dnh, gbl2) { \\\n";
    
    // Perform the check for the 8kb shared bitmap
    passcheckDefine += ("  lookupIndex = (" + std::string(&bitmap_0_letter, 1) + " & SHARED_BITMAP_MASK) >> 3;\\\n");
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        sprintf(buffer, "  lookupResult%s = sb[lookupIndex%s];\\\n",
                VI(vectorPosition), VI(vectorPosition));
        passcheckDefine += buffer;
    }
    passcheckDefine += "  lookupResult = (lookupResult >> (" + std::string(&bitmap_0_letter, 1) + " & 0x00000007)) & 0x00000001;\\\n";

    // Check to see if the 8kb bitmap passed.
    passcheckDefine += (indent + buildIfCheck("lookupResult", vectorWidth));
    indent += "  ";
    
    // If the 256kb global bitmap is present, use it.
    if (use_l2_bitmap) {
        passcheckDefine += (indent + "lookupIndex = (" + std::string(&bitmap_0_letter, 1) + " >> 3) & 0x0003FFFF;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "lookupResult%s = lookupResult%s & (gbl2[lookupIndex%s] >> (%c%s & 0x00000007)) & 0x00000001;\\\n",
                    VI(vectorPosition), VI(vectorPosition),VI(vectorPosition), bitmap_0_letter, VI(vectorPosition) );
            passcheckDefine += (indent + buffer);
        }
        passcheckDefine += (indent + buildIfCheck("lookupResult", vectorWidth));
        indent += "  ";
    }
    
    // If the large bitmap0 is present, compare with it.
    if (bitmap_0_is_present) {
        passcheckDefine += (indent + "lookupIndex = (" + std::string(&bitmap_0_letter, 1) + " >> 3) & 0x07FFFFFF;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "if (lookupResult%s) {lookupResult%s = lookupResult%s & (gb%c[lookupIndex%s] >> (%c%s & 0x00000007)) & 0x00000001;}\\\n",
                    VI(vectorPosition), VI(vectorPosition), VI(vectorPosition), bitmap_0_letter, VI(vectorPosition), bitmap_0_letter, VI(vectorPosition) );
            passcheckDefine += (indent + buffer);
        }
    }

    // Check to see if we got past large bitmap 0.
    passcheckDefine += (indent + buildIfCheck("lookupResult", vectorWidth));
    indent += "  ";
    // If so, we need to calculate the step for bitmap 1 (present or not).
    passcheckDefine += (indent + bitmap_1_calculation_string + "\\\n");
    // If bitmap 1 is present, check against it.
    if (bitmap_1_is_present) {
        passcheckDefine += (indent + "lookupIndex = (" + std::string(&bitmap_1_letter, 1) + " >> 3) & 0x07FFFFFF;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "if (lookupResult%s) {lookupResult%s = lookupResult%s & (gb%c[lookupIndex%s] >> (%c%s & 0x00000007)) & 0x00000001;}\\\n",
                    VI(vectorPosition), VI(vectorPosition), VI(vectorPosition), bitmap_1_letter, VI(vectorPosition), bitmap_1_letter, VI(vectorPosition) );
            passcheckDefine += (indent + buffer);
        }
    }

    passcheckDefine += (indent + buildIfCheck("lookupResult", vectorWidth));
    indent += "  ";
    passcheckDefine += (indent + bitmap_2_calculation_string + "\\\n");
    if (bitmap_2_is_present) {
        passcheckDefine += (indent + "lookupIndex = (" + std::string(&bitmap_2_letter, 1) + " >> 3) & 0x07FFFFFF;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "if (lookupResult%s) {lookupResult%s = lookupResult%s & (gb%c[lookupIndex%s] >> (%c%s & 0x00000007)) & 0x00000001;}\\\n",
                    VI(vectorPosition), VI(vectorPosition), VI(vectorPosition), bitmap_2_letter, VI(vectorPosition), bitmap_2_letter, VI(vectorPosition) );
            passcheckDefine += (indent + buffer);
        }
    }

    passcheckDefine += (indent + buildIfCheck("lookupResult", vectorWidth));
    indent += "  ";
    passcheckDefine += (indent + bitmap_3_calculation_string + "\\\n");
    if (bitmap_3_is_present) {
        passcheckDefine += (indent + "lookupIndex = (" + std::string(&bitmap_3_letter, 1) + " >> 3) & 0x07FFFFFF;\\\n");
        for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
            sprintf(buffer, "if (lookupResult%s) {lookupResult%s = lookupResult%s & (gb%c[lookupIndex%s] >> (%c%s & 0x00000007)) & 0x00000001;}\\\n",
                    VI(vectorPosition), VI(vectorPosition), VI(vectorPosition), bitmap_3_letter, VI(vectorPosition), bitmap_3_letter, VI(vectorPosition) );
            passcheckDefine += (indent + buffer);
        }
    }
    
    // Put in the actual useful checks if we've passed everything.
    passcheckDefine += (indent + buildIfCheck("lookupResult", vectorWidth));
    indent += "  ";
    // Insert the pre-check macro if present.
    if (ifFoundRunMacro.length()) {
        passcheckDefine += (indent + ifFoundRunMacro + ";\\\n");
    }
    for (vectorPosition = 0; vectorPosition < vectorWidth; vectorPosition++) {
        sprintf(buffer, "if (lookupResult%s > 0) {\\\n",
                VI(vectorPosition));
        passcheckDefine += (indent + buffer);
        sprintf(buffer, "  %s(dgh, dfp, dfpf, dnh, %c);\\\n", 
                passwordCheckFunction.c_str(), vectorIndexes[vectorPosition]);
        passcheckDefine += (indent + buffer);
        passcheckDefine += (indent + "}\\\n");
    }
    if (use_l2_bitmap) {
        passcheckDefine += "} ";
    }
    passcheckDefine += "} } } } } }\n";
    
    return passcheckDefine;
}

std::string MFNOpenCLMetaprograms::makeCopyOperator(std::string copierName, int numberElements) {
    std::string copyOperator;
    int i;
    
    copyOperator += "#define " + copierName + "(src, dst) {\\\n";
    for (i = 0; i < numberElements; i++) {
        copyOperator += "dst[" + getInt(i) + "] = src[" + getInt(i) + "];\\\n";
    }
    copyOperator += "}\n\n";
    return copyOperator;
}


// Same as the little endian one, but big endian.  So less commented.
std::string MFNOpenCLMetaprograms::makePasswordNoMemMultipleIncrementorsBE(
        int passwordLength, int vectorWidth, 
        std::vector<std::vector<uint8_t> > charset, int passStride,
        int reverseMacroStep) {
    
    std::vector<std::vector<uint8_t> > cs = charset;
    
    // One for each position.
    std::vector<std::vector<charsetSkips> > skips;

    charsetSkips currentSkip;
    
    skips.resize(cs.size());
    
    for (int pos = 0; pos < cs.size(); pos++) {
        for (int i = 0; i < (cs[pos].size() - 1); i++) {
            
            if (cs[pos][i+1] != (cs[pos][i] + 1)) {
                currentSkip.skipFrom = cs[pos][i] + 1;
                currentSkip.skipTo = cs[pos][i+1];
                skips[pos].push_back(currentSkip);
            }
        }
        // Add the final skip in.
        currentSkip.skipFrom = cs[pos][cs[pos].size() - 1] + 1;
        currentSkip.skipTo = cs[pos][0];
        skips[pos].push_back(currentSkip);
    }
    
    std::string incrementorDefine;
    int passwordPosition;
    char buffer[1024];
    std::string indent;
    
    // Build the define macro.
    
    incrementorDefine += "#define OpenCLNoMemPasswordIncrementorBE() { \\\n";
    // Variables needed
    incrementorDefine += getVectorType(vectorWidth) + " currentChar;\\\n";
    incrementorDefine += getSignedVectorType(vectorWidth) + " enableMask;\\\n";
    
    // Put in the code to increment the first position.
    incrementorDefine += "currentChar = (b0 >> 24) & 0xff;\\\n";
    incrementorDefine += "currentChar++;\\\n";
    incrementorDefine += "b0 &= 0x00ffffff;\\\n";
    for (int i = 0; i < skips[0].size(); i++) {
        incrementorDefine += "currentChar = (currentChar == " + 
                getInt(skips[0][i].skipFrom) + ") ? " + 
                getInt(skips[0][i].skipTo) + " : currentChar;\\\n";
    }
    incrementorDefine += "b0 |= (currentChar << 24);\\\n";
    incrementorDefine += "enableMask = (currentChar == " + getInt(cs[0][0]) + ");\\\n";
    // For each new character position, add the appropriate code.
    for (passwordPosition = 1; passwordPosition < passwordLength; passwordPosition++) {
        if (vectorWidth == 1) {
            // The "any" operator, for whatever reasons, does not work on uint types.
            incrementorDefine += indent + "if (enableMask) {\\\n";
        } else {
            incrementorDefine += indent + "if (any(enableMask)) {\\\n";
        }
        indent += "  ";
        if (passwordPosition == 1) {
            incrementorDefine += "  " + getVectorType(vectorWidth) + " maskVector;\\\n";
        }
        // Set the mask vector appropriately.
        incrementorDefine += (indent + "maskVector = (enableMask) ? (" + getVectorType(vectorWidth) + ")");
        incrementorDefine += getPasswordMaskBE(passwordPosition * passStride);
        incrementorDefine += " : (" + getVectorType(vectorWidth) + ")0xffffffff;\\\n";
        
        // Shift the needed character into the low byte.
        sprintf(buffer, "currentChar = (%s >> %d) & 0xff;\\\n", 
                getPasswordVariableBE(passwordPosition * passStride).c_str(), 
                getShiftAmountBE(passwordPosition * passStride));
        incrementorDefine += (indent + buffer);
        incrementorDefine += (indent + "currentChar++;\\\n");
        
        for (int i = 0; i < skips[passwordPosition].size(); i++) {
            incrementorDefine += indent + "currentChar = (currentChar == " + 
                    getInt(skips[passwordPosition][i].skipFrom) + ") ? " + 
                    getInt(skips[passwordPosition][i].skipTo) + " : currentChar;\\\n";
        }
        
        // And with the mask vector to clear the position.
        incrementorDefine += (indent + getPasswordVariableBE(passwordPosition * passStride) + " &= maskVector;\\\n");
        // And with the negatated mask vector to add the new characters in
        incrementorDefine += (indent + getPasswordVariableBE(passwordPosition * passStride) + 
                " |= (~maskVector & (currentChar << " + getInt(getShiftAmountBE(passwordPosition * passStride)) + "));\\\n");
        // If the reverse macro is to be called, call it here.
        if (passwordPosition == reverseMacroStep) {
            incrementorDefine += (indent + "REVERSE();\\\n");
        }
        // Set the enableMask.  The compiler *should* remove this on the last loop...
        incrementorDefine += (indent + "enableMask = (enableMask && (currentChar == " + getInt(cs[passwordPosition][0]) + "));\\\n");
    }
    
    for (passwordPosition = 0; passwordPosition < passwordLength; passwordPosition++) {
        incrementorDefine += "} ";
    }
    incrementorDefine += "\n";
    return incrementorDefine;
}

std::string MFNOpenCLMetaprograms::makePasswordNoMemSingleIncrementorsBE(
        int passwordLength, int vectorWidth, 
        std::vector<std::vector<uint8_t> > charset, int passStride,
        int reverseMacroStep) {
    std::vector<uint8_t> cs = charset[0];
    std::vector<charsetSkips> skips;
    charsetSkips currentSkip;

    for (int i = 0; i < (cs.size() - 1); i++) {
        if (cs[i+1] != (cs[i] + 1)) {
            //printf("Skip detected from %d to %d\n", cs[i], cs[i+1]);
            currentSkip.skipFrom = cs[i] + 1;
            currentSkip.skipTo = cs[i+1];
            skips.push_back(currentSkip);
        }
    }
    // Add the final skip in.
    currentSkip.skipFrom = cs[cs.size() - 1] + 1;
    currentSkip.skipTo = cs[0];
    skips.push_back(currentSkip);
    
    std::string incrementorDefine;
    int passwordPosition;
    char buffer[1024];
    std::string indent;
    
    // Build the define macro.
    incrementorDefine += "#define OpenCLNoMemPasswordIncrementorBE() { \\\n";
    // Variables needed
    incrementorDefine += getVectorType(vectorWidth) + " currentChar;\\\n";
    incrementorDefine += getSignedVectorType(vectorWidth) + " enableMask;\\\n";
    
    // Put in the code to increment the first position.
    incrementorDefine += "currentChar = (b0 >> 24) & 0xff;\\\n";
    incrementorDefine += "currentChar++;\\\n";
    incrementorDefine += "b0 &= 0x00ffffff;\\\n";
    for (int i = 0; i < skips.size(); i++) {
        incrementorDefine += "currentChar = (currentChar == " + 
                getInt(skips[i].skipFrom) + ") ? " + 
                getInt(skips[i].skipTo) + " : currentChar;\\\n";
    }
    incrementorDefine += "b0 |= (currentChar << 24);\\\n";
    incrementorDefine += "enableMask = (currentChar == " + getInt(cs[0]) + ");\\\n";
    // For each new character position, add the appropriate code.
    for (passwordPosition = 1; passwordPosition < passwordLength; passwordPosition++) {
        if (vectorWidth == 1) {
            // The "any" operator, for whatever reasons, does not work on uint types.
            incrementorDefine += indent + "if (enableMask) {\\\n";
        } else {
            incrementorDefine += indent + "if (any(enableMask)) {\\\n";
        }
        indent += "  ";
        if (passwordPosition == 1) {
            incrementorDefine += "  " + getVectorType(vectorWidth) + " maskVector;\\\n";
        }
        // Set the mask vector appropriately.
        incrementorDefine += (indent + "maskVector = (enableMask) ? (" + getVectorType(vectorWidth) + ")");
        incrementorDefine += getPasswordMaskBE(passwordPosition * passStride);
        incrementorDefine += " : (" + getVectorType(vectorWidth) + ")0xffffffff;\\\n";
        
        // Shift the needed character into the low byte.
        sprintf(buffer, "currentChar = (%s >> %d) & 0xff;\\\n", 
                getPasswordVariableBE(passwordPosition * passStride).c_str(), 
                getShiftAmountBE(passwordPosition * passStride));
        incrementorDefine += (indent + buffer);
        incrementorDefine += (indent + "currentChar++;\\\n");
        for (int i = 0; i < skips.size(); i++) {
            incrementorDefine += indent + "currentChar = (currentChar == " + 
                    getInt(skips[i].skipFrom) + ") ? " + 
                    getInt(skips[i].skipTo) + " : currentChar;\\\n";
        }
        
        // And with the mask vector to clear the position.
        incrementorDefine += (indent + getPasswordVariableBE(passwordPosition * passStride) + " &= maskVector;\\\n");
        // And with the negatated mask vector to add the new characters in
        incrementorDefine += (indent + getPasswordVariableBE(passwordPosition * passStride) + 
                " |= (~maskVector & (currentChar << " + getInt(getShiftAmountBE(passwordPosition * passStride)) + "));\\\n");
        // If the reverse macro is to be called, call it here.
        if (passwordPosition == reverseMacroStep) {
            incrementorDefine += (indent + "REVERSE();\\\n");
        }
        // Set the enableMask.  The compiler *should* remove this on the last loop...
        incrementorDefine += (indent + "enableMask = (enableMask && (currentChar == " + getInt(cs[0]) + "));\\\n");
    }
    
    for (passwordPosition = 0; passwordPosition < passwordLength; passwordPosition++) {
        incrementorDefine += "} ";
    }
    incrementorDefine += "\n";
    return incrementorDefine;
}


#if UNIT_TEST
int main() {
    MFNOpenCLMetaprograms CodeClass;
    
    std::string result;
    
    //result = CodeClass.makePasswordMultipleIncrementorsLE(5, 4, 128);
    //result = CodeClass.makePasswordSingleIncrementorsLE(5, 1);
    //result = CodeClass.makePasswordMultipleIncrementorsNTLM(5, 4, 128);
    //result = CodeClass.makeBitmapLookup(4, 4, "CheckPassword128BE");
//    result = CodeClass.makeBitmapLookupEarlyOut(4,  "CheckPassword128LE",
//    'a', 1, 
//    'd', 1, "MD5II(d, a, b, c, b3, MD5S42, 0x8f0ccc92);",
//    'c', 1, "MD5II(c, d, a, b, b10, MD5S43, 0xffeff47d);",
//    'b', 1, "MD5II(b, c, d, a, b1, MD5S44, 0x85845dd1);", 
//    1);
    
    //result = CodeClass.makeCopyOperator("copyMe", 32);
    //result = CodeClass.makePasswordSingleIncrementorsBE(5, 4);
    //result = CodeClass.makePasswordMultipleIncrementorsBE(5, 4, 128);
//    result = 
//            CodeClass.makeBitmapLookupEarlyOut(4,  "CheckPassword160",
//        'd', 1, 
//        'c', 1, "b13 = SHA1_ROTL((b10^b5^b15^b13), 1); one_cycle_par2(d,e,a,b,c,b13);",
//        'b', 1, "b14 = SHA1_ROTL((b11^b6^b0^b14), 1); one_cycle_par2(c,d,e,a,b,b14);",
//        'a', 1, "b15 = SHA1_ROTL((b12^b7^b1^b15), 1); one_cycle_par2(b,c,d,e,a,b15);",
//        1);
    std::vector<std::vector<uint8_t> > charset;
    charset.push_back(std::vector<uint8_t>());
    for (int i = 0x20; i <= 0x7E; i++) {
        charset[0].push_back(i);
    }
    /*for (int i = 0x30; i <= 0x39; i++) {
        charset[0].push_back(i);
    }
    for (int i = 0x40; i <= 0x5A; i++) {
        charset[0].push_back(i);
    }
    for (int i = 0x60; i <= 0x6A; i++) {
        charset[0].push_back(i);
    }*/
    result = CodeClass.makePasswordSingleIncrementorsLEMemoryless(4, 
        4, charset);
    
    printf("%s\n", result.c_str());
}
#endif