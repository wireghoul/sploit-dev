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
    
    print "#define MFNSingleIncrementorsNTLMOpenCL" . ($passwordLength + 1) . "(charsetForward, charsetReverse, charsetLengths, suffix) {\\\n";

    for ($passPos = 0; $passPos <= $passwordLength; $passPos++) {
        print $tabOffset . "passOffset = charsetReverse[("
            . getRegisterName($passPos * 2) . " >> " . getRegisterShiftBits($passPos * 2) . ") & 0xff]; \\\n";
        print $tabOffset . getRegisterName($passPos * 2) . " &= " . getRegisterMask($passPos * 2) . ";\\\n";
        print $tabOffset . "passOffset++;\\\n";
        print $tabOffset . getRegisterName($passPos * 2) .
            " |= (unsigned int)(charsetForward[passOffset] << " . getRegisterShiftBits($passPos * 2) .
            ");\\\n";
        print $tabOffset . "if (passOffset >= charsetLengths[0]) { \\\n";
        $tabOffset .= "  ";
        print $tabOffset . getRegisterName($passPos * 2) . " &= " . getRegisterMask($passPos * 2). ";\\\n";
        print $tabOffset . getRegisterName($passPos * 2) .
            " |= (unsigned int)(charsetForward[0] << " . getRegisterShiftBits($passPos * 2) .
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

    print "#define MFNMultipleIncrementorsNTLMOpenCL" . ($passwordLength + 1) . "(charsetForward, charsetReverse, charsetLengths, suffix) {\\\n";

    for ($passPos = 0; $passPos <= $passwordLength; $passPos++) {
        print $tabOffset . "passOffset = charsetReverse[(($maxCharsetLength * $passPos) + ("
            . getRegisterName($passPos * 2) . " >> " . getRegisterShiftBits($passPos * 2) . ") & 0xff)]; \\\n";
        print $tabOffset . getRegisterName($passPos * 2) . " &= " . getRegisterMask($passPos * 2) . ";\\\n";
        print $tabOffset . "passOffset++;\\\n";
        print $tabOffset . getRegisterName($passPos * 2) .
            " |= (unsigned int)(charsetForward[($maxCharsetLength * $passPos) + passOffset] << " . getRegisterShiftBits($passPos * 2) .
            ");\\\n";
        print $tabOffset . "if (passOffset >= charsetLengths[$passPos]) { \\\n";
        $tabOffset .= "  ";
        print $tabOffset . getRegisterName($passPos * 2) . " &= " . getRegisterMask($passPos * 2). ";\\\n";
        print $tabOffset . getRegisterName($passPos * 2) .
            " |= (unsigned int)(charsetForward[($maxCharsetLength * $passPos)] << " . getRegisterShiftBits($passPos * 2) .
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


// Now build an ugly case statement that expands out to the needed calls.

/*
 *                 case 6:
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 0);
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 1);
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 2);
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 3);
#endif
#if grt_vector_8 || grt_vector_16
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 4);
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 5);
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 6);
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 7);
#endif
#if grt_vector_16
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 8);
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 9);
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 10);
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 11);
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 12);
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 13);
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 14);
                    MFNSingleIncrementorsNTLMOpenCL6 (sharedCharsetPlainNTLM, sharedReverseCharsetPlainNTLM, sharedCharsetLengthsPlainNTLM, 15);
#endif
                break;

 */
?>