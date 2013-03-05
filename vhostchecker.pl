#!/usr/bin/perl

# Perl script to test for vhosts on a given IP
# Usage - ./vhostchecker.pl --ips <file containing IPs> \
#	--hosts <file containing hosts> [-u 'useragent'] [--nocolour] [--append <domain to append>]

# geoff.jones@cyberis.co.uk - Geoff Jones 04/02/2013 - v0.1

# Copyright (C) 2013  Cyberis Limited

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

use warnings;
use strict;

use Getopt::Long;
use Term::ANSIColor;
require LWP::UserAgent;
require LWP::Protocol::https;

my @ips;
my @vhosts;
my $ipfile;
my $hostfile;
my $nocolour  = '';
my $useragent = 'Mozilla/5.0';
my $insecure  = '';
my $ua;
my $verbose;
my $append = '';

my $result = GetOptions(
    "ips=s"       => \$ipfile,
    "hosts=s"     => \$hostfile,
    "nocolour+"   => \$nocolour,
    "verbose+"    => \$verbose,
    "useragent=s" => \$useragent,
    "append=s"    => \$append
);

if ( !defined($ipfile) || !defined($hostfile) ) {
    print usage();
    exit;
}

open( F, $ipfile ) or die "Failed to open file containing IPs - $!\n";
while (<F>) {
    if (m/([0-9]{1,3}\.){3}[0-9]{1,3}(\:[0-9]{1,5})?/) {
        chomp;
        push( @ips, $_ );
    }
}
close(F);

print STDERR "[INFO] Read " . @ips . " IP's from file \"$ipfile\" \n";

# Load the vhosts into memory
open( F, $hostfile ) or die "Failed to open file containing hosts - $!\n";
while (<F>) {
    chomp;
    next if (m/^ *$/);
    push( @vhosts, $_ . $append );
}
close(F);

print STDERR "[INFO] Read " . @vhosts . " vhosts from file \"$hostfile\" \n";

foreach (@ips) {
    my $ip = $_;
    print "\nChecking IP: $ip ";

    $ua = LWP::UserAgent->new( agent => $useragent );
    $ua->default_header( 'Host' => $ip );

    my $response = $ua->get("http://$ip/");
    printResponse($response);

    foreach (@vhosts) {
        print "\tChecking VHOST against $ip: $_ ";

        $ua = LWP::UserAgent->new( agent => $useragent );
        $ua->default_header( 'Host' => $_ );

        my $response = $ua->get("http://$ip/");
        printResponse($response);
    }
}

sub printResponse {
    my $r = shift;

    if ( $r->code eq 200 && !$nocolour ) { print color 'green'; }
    if ( ( $r->code eq 301 || $r->code eq 302 ) && !$nocolour ) {
        print color 'blue';
    }
    if ( $r->code >= 400 && $r->code < 500 && !$nocolour ) {
        print color 'yellow';
    }
    if ( $r->code eq 500 && !$nocolour ) { print color 'red'; }

    if ( defined( $r->header('Location') ) ) {
        print "[C:"
          . $r->code . " L:"
          . length( $r->content ) . " R:"
          . $r->header('Location') . "]\n";
    }
    else {
        print "[C:" . $r->code . " L:" . length( $r->content ) . "]\n";
    }

    print color 'reset';
}

sub usage {
    print STDERR "\nPerl script to test for vhosts on a given IP\n\n";
    print STDERR
"\tUsage - $0 --ips <file containing IPs> --hosts <file containing hosts> \\\n";
    print STDERR
      "\t\t[-u 'useragent'] [--nocolour] [--append <domain to append>] <\n\n";
}
