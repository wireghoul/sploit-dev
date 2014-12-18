#!/usr/bin/perl
# Decrypts Websphere Passwords stored in the {XOR} format
# Usage: WSxorcracker.pl "{XOR}abc="
#
use strict;
use warnings;
use MIME::Base64;


my $rawstr = $ARGV[0];
$rawstr=~ s/^{XOR}//i;
print "$rawstr\n";
my $xorstr = decode_base64($rawstr);
for my $echar (split //, $xorstr) {
  my $pchar = $echar ^ "_";
  print $pchar;
}
print "\n";
