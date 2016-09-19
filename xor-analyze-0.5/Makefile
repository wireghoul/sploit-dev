#
# $Id: Makefile,v 1.6 2001/09/25 17:19:30 marvin Exp $
#
TARGETS=xor-enc xor-dec xor-analyze

WIN32_CC=i586-mingw32msvc-gcc
WIN32_CXX=i586-mingw32msvc-g++
WIN32_CXXFLAGS=-DWIN32_CROSS

CC=gcc
CXX=g++

all: $(TARGETS)

xor-dec:
	ln -s xor-enc xor-dec
xor-enc: xor-enc.c
	$(CC) -o xor-enc xor-enc.c
xor-analyze: xor-analyze.cc
	$(CXX) -o xor-analyze xor-analyze.cc

win32:
	$(WIN32_CC) -c -o getopt_win32.o getopt.c
	$(WIN32_CXX) $(WIN32_CXXFLAGS) -o xor-analyze.exe getopt_win32.o xor-analyze.cc
	$(WIN32_CC) -o xor-enc.exe xor-enc.c
	$(WIN32_CC) -o xor-dec.exe xor-enc.c
clean:
	rm -f $(TARGETS) *.o *.exe
