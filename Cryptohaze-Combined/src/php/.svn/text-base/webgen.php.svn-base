<?php
// Distributed Web Generation.

// =================== CONFIG ===================
$generateAlgorithmName = "MD5";
// How many chains to generate at a time.  10M is a good default.
$generateNumberChains = 10000000;
// Chain length
$generateChainLength = 200000;
// Password length
$generatePasswordLength = 8;
// Table index
$generateTableIndex = 0;
// Bits of hash to store for the V3 tables.
$generateBitsOfHash = 80;
// Charset stuff
$charset = ' !"#$%&'."'".'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~';

// =================== CONFIG ===================

include "wtconfig.php";
include "GRTTableHeaders.php";

// Algorithm name->ID mappings - must match GRTHashes.cpp
$algorithmIds = array(
    'NTLM' => 0,
    'MD5' => 1,
    'MD4' => 2,
    'SHA1' => 3,
    'SHA256' => 4,
    );

// Request for new table header to generate.
if (isset($_POST['readTableHeader']) && ($_POST['readTableHeader'] == 'generate')) {
    $GRTTableHeader = new GRTTableHeaderBuilder();

    // Always use table version 3.
    $GRTTableHeader->setTableVersion(3);
    $GRTTableHeader->setHashString($generateAlgorithmName);
    $GRTTableHeader->setHashVersion($algorithmIds[$generateAlgorithmName]);
    $GRTTableHeader->setTableIndex($generateTableIndex);
    $GRTTableHeader->setChainLength($generateChainLength);
    $GRTTableHeader->setNumberChains($generateNumberChains);
    $GRTTableHeader->setPasswordLength($generatePasswordLength);

    $GRTTableHeader->setCharsetCount(1);
    $GRTTableHeader->setSingleCharsetLength(strlen($charset));
    $GRTTableHeader->setSingleCharset($charset);
    $GRTTableHeader->setBitsInPassword(0);
    $GRTTableHeader->setBitsInHash($generateBitsOfHash);

    // Get a random seed value.
    $tableSeedValue = mt_rand(0, 0xffffffff);
    $GRTTableHeader->setRandomSeedValue($tableSeedValue);

    // Set a somewhat random chain start offset.
    $chainStartOffset = mt_rand(0, 100000);
    $GRTTableHeader->setChainStartOffset($chainStartOffset);

    $tableHeader = $GRTTableHeader->getTableHeaderString();

    if (strlen($tableHeader) != 8192) {
        exit;
    }
    ob_start();
    print $tableHeader;
    ob_flush();
    exit;
} else 
// Request to upload a file.
if (isset($_FILES['uploadFilename'])) {
    $uploadFilenameFullPath = $tableUploadPath . mt_rand(0x00, 0xffffffff) . basename($_FILES['uploadFilename']['name']);

    // Check to make sure it's actually an uploaded file
    if (is_uploaded_file($_FILES['uploadFilename']['tmp_name'])) {
        // Do stuff
        // Check 0.02% of chains
        $chainVerificationInterval = $generateNumberChains / 50;
        $verifyCommand = $tableVerifyBinaryPath . " " . $_FILES['uploadFilename']['tmp_name'] . " 0 $chainVerificationInterval > /dev/null";
        //print "Verify command: " . $verifyCommand . "\n";
        system($verifyCommand, $verifyReturnValue);
        //print "return value: $verifyReturnValue\n";
        if ($verifyReturnValue == 0) {
            print "OK";
            move_uploaded_file($_FILES['uploadFilename']['tmp_name'], $uploadFilenameFullPath);
        } else {
            print "FAILURE: Uploaded file failed verification.\n";
        }
    } else {
        // Something weird happened.  Quit.
        print "FAILURE: Invalid uploaded file!\n";
        exit;
    }
} else if (isset($_POST['getAlgorithmName']) && ($_POST['getAlgorithmName'] == 'get')) {
  // Just print the hash type.
  print $algorithmIds[$generateAlgorithmName];
}

?>