#!/bin/sh
mysql-proxy --log-file=/var/log/mysql/proxy.log --log-level=debug --proxy-lua-script=$HOME/sploit-dev/myproxy-test.lua
