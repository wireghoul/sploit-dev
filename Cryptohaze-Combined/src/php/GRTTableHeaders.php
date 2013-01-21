<?php
// Class to handle table header encoding.  This should work for all cases...

class GRTTableHeaderBuilder {

    // Store the variables related to the table header.  This stores for ALL
    // table header types, and only the relevant ones will be included.

    private $tableVersion = -1;
    private $hashVersionNumeric = -1;
    private $hashVersionString = "";

    // For V2 and V3 tables
    private $bitsInPassword = -1;
    private $bitsInHash = -1;

    private $tableIndex = -1;
    private $chainLength = -1;
    private $numberChains = -1;

    private $isPerfect = 0;

    private $passwordLength = -1;
    private $charsetCount = -1;
    private $charsetLengths = array(16);

    private $charset;

    // For V3 tables only
    private $randomSeedValue = -1;
    private $chainStartOffset = -1;

    // Initialize charset array
    function __construct() {
        //print "GRTTableHeaderBuilder::__construct()\n";
        for ($i = 0; $i < 16; $i++) {
            $this->charsetLengths[$i] = 0;
            $this->charset[$i] = array();
            for ($j = 0; $j < 256; $j++) {
                $this->charset[$i][$j] = 0;
            }
        }
    }

    // The setters.
    function setTableVersion($newTableVersion) {
        $this->tableVersion = $newTableVersion;
    }
    function setHashVersion($newHashVersion) {
        $this->hashVersionNumeric = $newHashVersion;
    }
    function setHashString($newHashString) {
        $this->hashVersionString = $newHashString;
    }
    function setBitsInPassword($newBitsInPassword) {
        $this->bitsInPassword = $newBitsInPassword;
    }
    function setBitsInHash($newBitsInHash) {
        $this->bitsInHash = $newBitsInHash;
    }
    function setTableIndex($newTableIndex) {
        $this->tableIndex = $newTableIndex;
    }
    function setChainLength($newChainLength) {
        $this->chainLength = $newChainLength;
    }
    function setNumberChains($newNumberChains) {
        $this->numberChains = $newNumberChains;
    }
    function setIsPerfect($newIsPerfect) {
        $this->isPerfect = $newIsPerfect;
    }
    function setPasswordLength($newPasswordLength) {
        $this->passwordLength = $newPasswordLength;
    }
    function setCharsetCount($newCharsetCount) {
        $this->charsetCount = $newCharsetCount;
    }
    // Only set the charset length for position 0 - we just support single charsets now
    function setSingleCharsetLength($newCharsetLength) {
        $this->charsetLengths[0] = $newCharsetLength;
    }
    function setSingleCharset($newCharsetString) {
        for ($i = 0; $i < strlen($newCharsetString); $i++) {
            $this->charset[0][$i] = $newCharsetString[$i];
        }
    }
    function setRandomSeedValue($newRandomSeedValue) {
        $this->randomSeedValue = $newRandomSeedValue;
    }
    function setChainStartOffset($newChainStartOffset) {
        $this->chainStartOffset = $newChainStartOffset;
    }

    // Returns an 8192 byte long string with the table header.
    function getTableHeaderString() {
        // Pack up a string for the table header.

        // Check values for all table formats.
        if ($this->tableVersion == -1) {
            print "ERROR: tableVersion must be set!\n";
            return NULL;
        }
        if ($this->hashVersionNumeric == -1) {
            print "ERROR: hashVersionNumeric must be set!\n";
            return NULL;
        }
        if ($this->hashVersionString == "") {
            print "ERROR: hashVersionString must be set!\n";
            return NULL;
        }
        if ($this->tableIndex == -1) {
            print "ERROR: tableIndex must be set!\n";
            return NULL;
        }
        if ($this->chainLength == -1) {
            print "ERROR: chainLength must be set!\n";
            return NULL;
        }
        if ($this->numberChains == -1) {
            print "ERROR: numberChains must be set!\n";
            return NULL;
        }
        if ($this->passwordLength == -1) {
            print "ERROR: passwordLength must be set!\n";
            return NULL;
        }
        if ($this->charsetCount == -1) {
            print "ERROR: charsetCount must be set!\n";
            return NULL;
        }

        if ($this->tableVersion >= 2) {
            if ($this->bitsInPassword == -1) {
                print "ERROR: bitsInPassword must be set!\n";
                return NULL;
            }
            if ($this->bitsInHash == -1) {
                print "ERROR: bitsInHash must be set!\n";
                return NULL;
            }
        }
        if ($this->tableVersion >= 3) {
            if ($this->randomSeedValue == -1) {
                print "ERROR: randomSeedValue must be set!\n";
                return NULL;
            }
            if ($this->chainStartOffset == -1) {
                print "ERROR: chainStartOffset must be set!\n";
                return NULL;
            }
        }

        // All sanity checks passed.  Go about building the table string!
        $tableHeaderString = "";

        // Put the table header magic in place.
        $tableHeaderString .= "GRT";

        // Table Version
        $tableHeaderString .= pack("C", $this->tableVersion);

        // Hash Version & Hash String
        $tableHeaderString .= pack("C", $this->hashVersionNumeric);
        for ($i = 0; $i < 16; $i++) {
            if ($i < strlen($this->hashVersionString)) {
                $tableHeaderString .= $this->hashVersionString[$i];
            } else {
                $tableHeaderString .= chr(0);
            }
        }
        // If table header version is 2 or 3, add bits in hash/password
        if ($this->tableVersion >= 2) {
            $tableHeaderString .= pack("C", $this->bitsInPassword);
            $tableHeaderString .= pack("C", $this->bitsInHash);
        } else {
            // Null padding bytes
            $tableHeaderString .= chr(0);
            $tableHeaderString .= chr(0);
        }
        // Add Reserved1
        $tableHeaderString .= chr(0);

        // Add TableIndex - 32-bit little endian unsigned
        $tableHeaderString .= pack("V", $this->tableIndex);

        // Add ChainLength: 32-bit little endian unsigned
        $tableHeaderString .= pack("V", $this->chainLength);

        // Add NumberChains: 64-bit little endian unsigned
        $tableHeaderString .= pack("V", ($this->numberChains & 0xffffffff));
        $tableHeaderString .= pack("V", ($this->numberChains >> 32));

        // Add IsPerfect: unsigned char
        $tableHeaderString .= pack("C", $this->isPerfect);

        // Add PasswordLength: unsigned char
        $tableHeaderString .= pack("C", $this->passwordLength);

        // Add CharsetCount: unsigned char
        $tableHeaderString .= pack("C", $this->charsetCount);

        // Add 16 bytes of CharsetLength.
        for ($i = 0; $i < 16; $i++) {
            if ($this->charsetLengths[$i] > 0) {
                $tableHeaderString .= pack("C", $this->charsetLengths[$i]);
            } else {
                $tableHeaderString .= chr(0);
            }
        }

        // Add the charset array in.
        for ($i = 0; $i < 16; $i++) {
            for ($j = 0; $j < 256; $j++) {
                // Ensure that the '0' character is passed in properly.
                // Otherwise it evaluates as false and messes things up.
                if ($this->charset[$i][$j] !== 0) { 
                    $tableHeaderString .= $this->charset[$i][$j];
                } else {
                    $tableHeaderString .= chr(0);
                }
            }
        }

        // If table version is 3, add the random seed & start offset
        if ($this->tableVersion == 3) {
            // Add RandomSeedValue: 32 bit unsigned int
            $tableHeaderString .= pack("V", $this->randomSeedValue);

            // Add ChainStartOffset: 64-bit little endian unsigned
            $tableHeaderString .= pack("V", ($this->chainStartOffset & 0xffffffff));
            $tableHeaderString .= pack("V", ($this->chainStartOffset >> 32));
        }

        // Now, pad to 8192 bytes.
        $bytesToPad = (8192 - strlen($tableHeaderString));
        for ($i = 0; $i < $bytesToPad; $i++) {
            $tableHeaderString .= chr(0);
        }


        //print "String length: " . strlen($tableHeaderString) . "\n";

        //echo bin2hex($tableHeaderString);
        //print "\n\n";
        return $tableHeaderString;
    }
}

/*
// TEST CODE

$GRTTableHeader = new GRTTableHeaderBuilder();


$GRTTableHeader->setTableVersion(2);
$GRTTableHeader->setHashVersion(1);
$GRTTableHeader->setHashString("MD5");
$GRTTableHeader->setTableIndex(100000);
$GRTTableHeader->setChainLength(1000);
$GRTTableHeader->setNumberChains(1000000);
$GRTTableHeader->setPasswordLength(8);

// Charset stuff
$charset = "abcdefghijklmnopqrstuvwxyz";
$GRTTableHeader->setCharsetCount(1);
$GRTTableHeader->setSingleCharsetLength(strlen($charset));
$GRTTableHeader->setSingleCharset($charset);
$GRTTableHeader->setBitsInPassword(48);
$GRTTableHeader->setBitsInHash(48);

//print_r($GRTTableHeader);

$tableHeader = $GRTTableHeader->getTableHeaderString();

if (strlen($tableHeader) != 8192) {
    print "ERROR: tableHeader not 8192 bytes!\n";
}

print "Got table header.\n";
*/

?>
