#!/bin/sh
grep -o -E '<Name>[^<]+</Name>' | sed -e's/<Name>//' -e's/<.*//'
