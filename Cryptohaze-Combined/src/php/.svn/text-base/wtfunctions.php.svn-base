<?php
// Functions


function isValidGRT($filename) {
    // File exists - return the table header.
    $handle = fopen($filename, "rb");
    if ($handle) {
        // If the open was successful, return the first 8 bytes.
        $tableHeader = fread($handle, 3);

        if ($tableHeader === FALSE) {
            return 0;
        }

        $tableHeaderArray = unpack("Cmagic0/Cmagic1/Cmagic2", $tableHeader);

        // check magic
        if ($tableHeaderArray['magic0'] != ord('G')) {
            return 0;
        }
        if ($tableHeaderArray['magic1'] != ord('R')) {
            return 0;
        }
        if ($tableHeaderArray['magic2'] != ord('T')) {
            return 0;
        }
        // Magic passes.
        return 1;
    }
    return 0;
}


function getHashVersion($filename) {
    // File exists - return the table header.
    $handle = fopen($filename, "rb");
    if ($handle) {
        // If the open was successful, return the first 8 bytes.
        $tableHeader = fread($handle, 5);

        if ($tableHeader === FALSE) {
            return -1;
        }

        $tableHeaderArray = unpack("Cmagic0/Cmagic1/Cmagic2/Ctableversion/Chashversion", $tableHeader);

        // check magic
        if ($tableHeaderArray['magic0'] != ord('G')) {
            return -1;
        }
        if ($tableHeaderArray['magic1'] != ord('R')) {
            return -1;
        }
        if ($tableHeaderArray['magic2'] != ord('T')) {
            return -1;
        }

        return $tableHeaderArray['hashversion'];
    }
}

?>