<?php

// Test MD5 correctness.  Generate hashes of a bunch of data, concatenate them
// together, hash the result, and check the final hash.  This should stress the
// hell out of the function and highlight any issues.

$longHashData = "";
$hashResult = "";
$longHashString = "";
$hashSum = "";

for ($i = 0; $i < 10000; $i++) {
    $longHashData .= '1';
    $hashResult .= md5($longHashData);
}

print "Result: " . md5($hashResult) . "\n";
?>