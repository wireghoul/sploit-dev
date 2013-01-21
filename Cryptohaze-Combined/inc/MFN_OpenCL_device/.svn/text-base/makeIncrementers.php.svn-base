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
    return "b" . $registerIndex . ".s##suffix";
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
    
    print "#define MFNSingleIncrementorsOpenCL" . ($passwordLength + 1) . "(charsetForward, charsetReverse, charsetLengths, suffix) {\\\n";

    for ($passPos = 0; $passPos <= $passwordLength; $passPos++) {
        print $tabOffset . "passOffset = charsetReverse[("
            . getRegisterName($passPos) . " >> " . getRegisterShiftBits($passPos) . ") & 0xff]; \\\n";
        print $tabOffset . getRegisterName($passPos) . " &= " . getRegisterMask($passPos) . ";\\\n";
        print $tabOffset . "passOffset++;\\\n";
        print $tabOffset . getRegisterName($passPos) .
            " |= (unsigned int)(charsetForward[passOffset] << " . getRegisterShiftBits($passPos) .
            ");\\\n";
        print $tabOffset . "if (passOffset >= charsetLengths[0]) { \\\n";
        $tabOffset .= "  ";
        print $tabOffset . getRegisterName($passPos) . " &= " . getRegisterMask($passPos). ";\\\n";
        print $tabOffset . getRegisterName($passPos) .
            " |= (unsigned int)(charsetForward[0] << " . getRegisterShiftBits($passPos) .
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

    print "#define MFNMultipleIncrementorsOpenCL" . ($passwordLength + 1) . "(charsetForward, charsetReverse, charsetLengths, suffix) {\\\n";

    for ($passPos = 0; $passPos <= $passwordLength; $passPos++) {
        print $tabOffset . "passOffset = charsetReverse[(($maxCharsetLength * $passPos) + ("
            . getRegisterName($passPos) . " >> " . getRegisterShiftBits($passPos) . ") & 0xff)]; \\\n";
        print $tabOffset . getRegisterName($passPos) . " &= " . getRegisterMask($passPos) . ";\\\n";
        print $tabOffset . "passOffset++;\\\n";
        print $tabOffset . getRegisterName($passPos) .
            " |= (unsigned int)(charsetForward[($maxCharsetLength * $passPos) + passOffset] << " . getRegisterShiftBits($passPos) .
            ");\\\n";
        print $tabOffset . "if (passOffset >= charsetLengths[$passPos]) { \\\n";
        $tabOffset .= "  ";
        print $tabOffset . getRegisterName($passPos) . " &= " . getRegisterMask($passPos). ";\\\n";
        print $tabOffset . getRegisterName($passPos) .
            " |= (unsigned int)(charsetForward[($maxCharsetLength * $passPos)] << " . getRegisterShiftBits($passPos) .
            ");\\\n";

    }

    for ($passPos = 0; $passPos <= ($passwordLength + 1); $passPos++) {
        print "} ";
    }
    print "\n\n\n";

}




/**
 * Looks like this:

#define makeMFNSingleIncrementors5(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } } } } }
 */





?>