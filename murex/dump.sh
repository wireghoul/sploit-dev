#!/bin/sh
./ls.pl $0 $1 $2 | ./filelist.sh | xargs -n1 ./nulldump.pl
