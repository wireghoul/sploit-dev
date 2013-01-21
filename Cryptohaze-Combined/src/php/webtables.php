<?php

include "wtconfig.php";
include "wtfunctions.php";


// If the file exists, return '1' else return '0'
// todo: Verify that the file is of the requested hash type
if (isset($_POST['isValidTable']) && !isset($_POST['hashVersion'])) {
    // If the only check is to see if this is a valid table, do so.
    $filePath = $tableRootPath . basename($_POST['isValidTable']);
    if (isValidGRT($filePath)) {
        print "1";
    } else {
        print "0";
    }
    exit;
}

if (isset($_POST['isValidTable']) && isset($_POST['hashVersion'])) {
    $filePath = $tableRootPath . basename($_POST['isValidTable']);
    $hashVersion = getHashVersion($filePath);

    // If it's not a valid result, return false.
    if ($hashVersion == -1) {
        print "0";
        exit;
    }
    if (isValidGRT($filePath) && ($hashVersion == $_POST['hashVersion'])) {
        print "1";
    } else {
        print "0";
    }
    exit;
}

// Read a table header & return it.
if (isset($_POST['readTableHeader'])) {
    $filePath = $tableRootPath . basename($_POST['readTableHeader']);
    if (file_exists($filePath)) {
        // File exists - return the table header.
        $handle = fopen($filePath, "rb");
        if ($handle) {
            // If the open was successful, return the first 8192 bytes.
            $tableHeader = fread($handle, 8192);

            if ($tableHeader) {
                // Buffer the output and send it all at once.
                ob_start();
                print $tableHeader;
                ob_flush();
            }
        }
    } else {
        // File does not exist - just return nothing.
    }
    exit;
}

if (isset($_POST['candidateHashes'])) {

    // Get a temporary name for the candidate hashes to be stored in.
    $candidateHashFilename = tempnam ("/tmp" , "GRT-CH-" );
    //$candidateHashFilename = "/tmp/ntlm-candidate-hashes"; // Testing!
    $regenerateChainsFilename = tempnam("/tmp", "GRT-Regen-");

    $searchTablePath = $tableRootPath . basename($_POST['candidateHashesFilename']);


    // Dump the received hashes into a file, newline separated.
    file_put_contents($candidateHashFilename, str_replace('|', "\n", $_POST['candidateHashes']));

    //print $tableSearchBinaryPath . " $searchTablePath $candidateHashFilename $regenerateChainsFilename";
    // Run the table search.
    print system($tableSearchBinaryPath . " $searchTablePath $candidateHashFilename $regenerateChainsFilename 1 > /tmp/output");

    // Get the output.
    $regenChains = file_get_contents($regenerateChainsFilename);

    print $regenChains;
    
    // Clean up.
    unlink($candidateHashFilename);
    unlink($regenerateChainsFilename);

    exit;
}

// Deal with the case of raw hash submission
if (isset($_POST['rawCandidateHashes'])) {
    $rawData = file_get_contents($_FILES['rawCandidateHashData']['tmp_name']);
    // Ensure we got data.
    if ($rawData === FALSE) {
        exit(1);
    }

    // Build the filenames
    $candidateHashFilename = tempnam ("/tmp" , "GRT-CH-" );
    $regenerateChainsFilename = tempnam("/tmp", "GRT-Regen-");
    $searchTablePath = $tableRootPath . basename($_POST['candidateHashesFilename']);

    $candidateHashesString = "";
    for ($i = 0; $i < $_FILES['rawCandidateHashData']['size']; $i++) {
        if ($i && (($i % 16) == 0)) {
            $candidateHashesString .= "\n";
        }
        $candidateHashesString .= sprintf("%02x", ord($rawData[$i]));
    }
    $candidateHashesString .= "\n";
    //print $candidateHashesString;

    file_put_contents($candidateHashFilename, $candidateHashesString);

    //print $tableSearchBinaryPath . " $searchTablePath $candidateHashFilename $regenerateChainsFilename";
    // Run the table search.
    print system($tableSearchBinaryPath . " $searchTablePath $candidateHashFilename $regenerateChainsFilename 1 > /tmp/output");

    // Get the output.
    $regenChains = file_get_contents($regenerateChainsFilename);

    print $regenChains;

    // Clean up.
    unlink($candidateHashFilename);
    unlink($regenerateChainsFilename);
}

// Deal with the case of raw hash submission
if (isset($_POST['rawCandidateHashesCondensed'])) {
    $rawData = file_get_contents($_FILES['rawCandidateHashData']['tmp_name']);
    // Ensure we got data.
    if ($rawData === FALSE) {
        exit(1);
    }

    // Build the filenames
    $candidateHashFilename = tempnam ("/tmp" , "GRT-CH-" );
    $regenerateChainsFilename = tempnam("/tmp", "GRT-Regen-");
    $searchTablePath = $tableRootPath . basename($_POST['candidateHashesFilename']);

    // Get how many bytes of hash we have.
    $significantBytesInHash = $_POST['rawHashLength'];

    $candidateHashesString = "";
    $bytesCopied = 0;

    while ($bytesCopied < $_FILES['rawCandidateHashData']['size']) {
        for ($i = 0; $i < $significantBytesInHash; $i++) {
            $candidateHashesString .= sprintf("%02x", ord($rawData[$bytesCopied]));
            $bytesCopied++;
        }
        for ($i = $significantBytesInHash; $i < 16; $i++) {
            $candidateHashesString .= "00";
        }
        $candidateHashesString .= "\n";
    }
    //print $candidateHashesString;

    file_put_contents($candidateHashFilename, $candidateHashesString);

    //print $tableSearchBinaryPath . " $searchTablePath $candidateHashFilename $regenerateChainsFilename";
    // Run the table search.
    print system($tableSearchBinaryPath . " $searchTablePath $candidateHashFilename $regenerateChainsFilename 1 > /tmp/output");

    // Get the output.
    $regenChains = file_get_contents($regenerateChainsFilename);

    print $regenChains;

    // Clean up.
    unlink($candidateHashFilename);
    unlink($regenerateChainsFilename);
}

// Get a list of tables based on hash ID
if (isset($_POST['getTableListByHashId'])) {

    if ($handle = opendir($tableRootPath)) {
        // Loop over the files, looking for the right hash name
        while (false !== ($file = readdir($handle))) {
            if ($file != ".." && $file != "." && (strpos($file, ".idx") === FALSE)) {
                $filePath = $tableRootPath . $file;
                if (isValidGRT($filePath) && (getHashVersion($filePath) == $_POST['getTableListByHashId'])) {
                    echo "$file\n";
                }
            }
        }
        closedir($handle);
    }
    exit;
}


if ($handle = opendir($tableRootPath)) {

    /* This is the correct way to loop over the directory. */
    while (false !== ($file = readdir($handle))) {
        if ($file != ".." && $file != "." && (strpos($file, ".idx") === FALSE))
		echo "$file\n";
    }

    closedir($handle);
}



//print_r($_GET);
//print_r($_POST);
//print_r($_FILES);
?>

