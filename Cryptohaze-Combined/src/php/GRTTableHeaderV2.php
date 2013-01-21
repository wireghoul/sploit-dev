<?php
// Functions to pack and unpack a table header type


/**
 * Unpacks a Version 2 table header and returns an associative array.
 * 
 * @param <binary string> $headerstring 
 * @return FALSE on failure, else associative array
 * 
 */

function unpackV2TableHeader($headerString) {
    $magic = unpack("c3Magic/Cversion", $headerString);

    // Check the magic
    if ( ($magic['Magic1'] != ord('G')) ||
         ($magic['Magic2'] != ord('R')) ||
         ($magic['Magic3'] != ord('T'))) {
        // Magic is not right - invalid table!
        return FALSE;
    }

    // Do a full decode.  Joy.

    $headerV2DecodeString = "";

    // 3 bytes of magic
    $headerV2DecodeString .= "C3Magic/";
    // 1 byte of Table Version
    $headerV2DecodeString .= "CTableVersion/";
    // 1 byte of Hash Version
    $headerV2DecodeString .= "CHashVersion/";
    // 16 bytes of hash name
    $headerV2DecodeString .= "C16HashName/";
    // 1 byte of BitsInPassword
    $headerV2DecodeString .= "CBitsInPassword/";
    // 1 byte of BitsInHash
    $headerV2DecodeString .= "CBitsInHash/";
    // 1 byte reserved
    $headerV2DecodeString .= "CReserved1/";

    // 4 bytes LE TableIndex
    $headerV2DecodeString .= "VTableIndex/";
    // 4 bytes LE ChainLength
    $headerV2DecodeString .= "VChainLength/";
    // 8 bytes NumberChains
    $headerV2DecodeString .= "VNumberChainsLow/";
    $headerV2DecodeString .= "VNumberChainsHi/";

    // 1 byte IsPerfect
    $headerV2DecodeString .= "CIsPerfect/";
    // 1 byte PasswordLength
    $headerV2DecodeString .= "CPasswordLength/";
    // 1 byte CharsetCount
    $headerV2DecodeString .= "CCharsetCount/";
    // 16 bytes of CharsetLength
    $headerV2DecodeString .= "C16CharsetLength/";
    // 16x256 of Charset
    for ($i = 0; $i < 16; $i++) {
        $headerV2DecodeString .= "C256Charset$i-/";
    }

    $tableHeaderArray = unpack($headerV2DecodeString, $headerString);

    // Clean some stuff up and create the strings as needed.
    $tableHeaderArray['Magic'] = "";
    for ($i = 1; $i <= 3; $i++) {
        $tableHeaderArray['Magic'] .= chr($tableHeaderArray["Magic$i"]);
        unset($tableHeaderArray["Magic$i"]);
    }
    $tableHeaderArray['HashName'] = "";
    for ($i = 1; $i <= 16; $i++) {
        $tableHeaderArray['HashName'] .= chr($tableHeaderArray["HashName$i"]);
        unset($tableHeaderArray["HashName$i"]);
    }

    $tableHeaderArray['NumberChains'] = ($tableHeaderArray['NumberChainsHi'] << 32) +
        $tableHeaderArray['NumberChainsLow'];
    unset($tableHeaderArray['NumberChainsHi']);
    unset($tableHeaderArray['NumberChainsLow']);

    // Charset joy
    for ($i = 0; $i <16; $i++) {
        $index = $i + 1;
        $tableHeaderArray['CharsetLength'][$i] = $tableHeaderArray["CharsetLength$index"];
        $tableHeaderArray['Charset'][$i] = "";
        for ($j = 1; $j <= 256; $j++) {
            $tableHeaderArray['Charset'][$i] .= chr($tableHeaderArray["Charset$i-$j"]);
            unset($tableHeaderArray["Charset$i-$j"]);
        }
        unset($tableHeaderArray["CharsetLength$index"]);
    }

    return $tableHeaderArray;
}

?>