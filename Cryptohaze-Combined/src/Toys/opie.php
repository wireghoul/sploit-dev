<?php

/**
 * Sample set of hashes generated with:
 * opiekey -x -5 -n 10 50 foobar
 * Secret key: foobar
 * 
41: E6F9 24A6 FFFC D777
42: 1668 748B 8869 EE5D
43: 26A9 53D2 1D86 D82E
44: BA02 CC7B 1911 8187
45: 2159 46CF CA91 2573
46: 66BB 8459 C133 E9B1
47: 1E58 6C34 4412 B098
48: B215 9B11 B9F0 A9F4
49: 442F C14C 3AB3 4E9F
50: 355F 76DF 074B 1774
 */


// Set the initial key to element 48
$initialKey = pack("H*", "B2159B11B9F0A9F4");

// Perform an MD5 on the binary data, get result as packed binary
$hashResult = md5($initialKey, 1);

print "hashResult raw MD5 result: " . bin2hex($hashResult) . "\n";

// Do the folding

$newKey = "";
for ($i = 0; $i < 8; $i++) {
    $newKey .= $hashResult[$i] ^ $hashResult[$i + 8];
}

print "newKey: " . bin2hex($newKey) . "\n";

print "Key (49?): ";
for ($i = 0; $i < 8; $i++) {
    print strtoupper(bin2hex($newKey[$i]));
    if (($i % 2) == 1) {
        print " ";
    }
}
printf("\n\n");
