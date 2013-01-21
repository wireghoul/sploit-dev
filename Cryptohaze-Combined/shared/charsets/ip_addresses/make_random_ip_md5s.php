<?php

// make random IP addresses in a variety of formats.

$numAddresses = 1000;

$ip_file_contents = "";
$md5_file_contents = "";

for ($i = 0; $i < $numAddresses; $i++) {
    $o1 = rand(0, 255);
    $o2 = rand(0, 255);
    $o3 = rand(0, 255);
    $o4 = rand(0, 255);

    $ipaddress = "";
    
    // Build the IP address.
    // Some will use normal formatting (192.168.1.1), some will use
    // full formatting (192.168.001.001)
    
    if (rand(0, 10) < 3) {
        $ipaddress = sprintf("%03d.%03d.%03d.%03d", $o1, $o2, $o3, $o4);
    } else {
        $ipaddress = sprintf("%d.%d.%d.%d", $o1, $o2, $o3, $o4);
    }
    $ip_file_contents .= $ipaddress . "\n";
    $md5_file_contents .= md5($ipaddress) . "\n";
}

file_put_contents("Random_IP_Addresses.txt", $ip_file_contents);
file_put_contents("Random_IP_MD5s.txt", $md5_file_contents);