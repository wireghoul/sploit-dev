<?php

/**
 * Make the incrementers for the new MFN device class.  This will be separate
 * for little endian & big endian , as the positions are swapped in the hash.
 */


$maxPasswordLength = 16;

// MUST MATCH the build settings!
$maxCharsetLength = 128;


//========================================================================

/**
 * Returns the register name for the given position.
 * @param int $position The position to get the register name for.
 * @return The register name (b0, b1, b2...)
 */
function getRegisterName($position) {
    $registerIndex = floor($position / 4);
    return "w" . $registerIndex;
}

/**
 * Determines the register shift amount in bits for a given position.
 * @param int $position The password position to get the shift for
 * @return int The number of bits to shift things around.
 */
function getRegisterShiftBits($position) {
    $registerShiftBits = ($position % 4) * 8;
    return $registerShiftBits;
}


function getRegisterMask($position) {
    switch ($position % 4) {
        case 0:
            return "0xffffff00";
        case 1:
            return "0xffff00ff";
        case 2:
            return "0xff00ffff";
        case 3:
            return "0x00ffffff";
    }
}




// Start main loop
for ($passwordLength = 0; $passwordLength < $maxPasswordLength; $passwordLength++) {
    // Reset the tab offset.
    $tabOffset = "";
    
    print "#define makeMFNSingleIncrementorsCPU_NTLM" . ($passwordLength + 1) . "(charsetForward, charsetReverse, charsetLengths) {\\\n";

    for ($passPos = 0; $passPos <= $passwordLength; $passPos++) {
        print $tabOffset . "passOffset = charsetReverse[("
            . getRegisterName($passPos * 2) . " >> " . getRegisterShiftBits($passPos * 2) . ") & 0xff]; \\\n";
        print $tabOffset . getRegisterName($passPos * 2) . " &= " . getRegisterMask($passPos * 2) . ";\\\n";
        print $tabOffset . "passOffset++;\\\n";
        print $tabOffset . getRegisterName($passPos * 2) .
            " |= (uint32_t)(charsetForward[passOffset] << " . getRegisterShiftBits($passPos * 2) .
            ");\\\n";
        print $tabOffset . "if (passOffset >= charsetLengths[0]) { \\\n";
        $tabOffset .= "  ";
        print $tabOffset . getRegisterName($passPos * 2) . " &= " . getRegisterMask($passPos * 2). ";\\\n";
        print $tabOffset . getRegisterName($passPos * 2) .
            " |= (uint32_t)(charsetForward[0] << " . getRegisterShiftBits($passPos * 2) .
            ");\\\n";

    }

    for ($passPos = 0; $passPos <= ($passwordLength + 1); $passPos++) {
        print "} ";
    }
    print "\n\n\n";
}


// Start main loop
for ($passwordLength = 0; $passwordLength < $maxPasswordLength; $passwordLength++) {
    // Reset the tab offset.
    $tabOffset = "";

    print "#define makeMFNMultipleIncrementorsCPU_NTLM" . ($passwordLength + 1) . "(charsetForward, charsetReverse, charsetLengths) {\\\n";

    for ($passPos = 0; $passPos <= $passwordLength; $passPos++) {
        print $tabOffset . "passOffset = charsetReverse[(($maxCharsetLength * $passPos) + (("
            . getRegisterName($passPos * 2) . " >> " . getRegisterShiftBits($passPos * 2) . ") & 0xff))]; \\\n";
        print $tabOffset . getRegisterName($passPos * 2) . " &= " . getRegisterMask($passPos * 2) . ";\\\n";
        print $tabOffset . "passOffset++;\\\n";
        print $tabOffset . getRegisterName($passPos * 2) .
            " |= (uint32_t)(charsetForward[($maxCharsetLength * $passPos) + passOffset] << " . getRegisterShiftBits($passPos * 2) .
            ");\\\n";
        print $tabOffset . "if (passOffset >= charsetLengths[$passPos]) { \\\n";
        $tabOffset .= "  ";
        print $tabOffset . getRegisterName($passPos * 2) . " &= " . getRegisterMask($passPos * 2). ";\\\n";
        print $tabOffset . getRegisterName($passPos * 2) .
            " |= (uint32_t)(charsetForward[($maxCharsetLength * $passPos)] << " . getRegisterShiftBits($passPos * 2) .
            ");\\\n";

    }

    for ($passPos = 0; $passPos <= ($passwordLength + 1); $passPos++) {
        print "} ";
    }
    print "\n\n\n";

}










?>