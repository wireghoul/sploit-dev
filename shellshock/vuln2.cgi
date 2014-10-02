#!/usr/bin/perl
print "Content-type: text/plain\r\n\r\n";
# Requires something  like stream >/dev/null, |, ; & or similar to be vulnerable
# straight up comand is not vulnerable (see vuln2-safe.*)
system("env 2>/dev/null");
print "Hello World $a\n";
