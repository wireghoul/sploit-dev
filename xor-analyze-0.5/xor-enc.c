/*
 *  xor-enc: XOR Cipher
 *  Copyright (C) 2000 Marvin (marvin@rootbusters.net)
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public
 *  License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 * $Id: xor-enc.c,v 1.4 2001/09/25 17:19:31 marvin Exp $
 */
#include <stdio.h>

int main(int argc, char **argv)
{
	FILE *fi, *fo;
        char *kp;
	char c;

	if (argc < 4 || !*argv[1]) {
		fprintf(stderr, "%s <key> <infile> <outfile>\n", argv[0]);
		return 1;
	}
	kp = argv[1];
	if (!(fi = fopen(argv[2], "rb"))) {
		perror("fopen(input)");
		return 1;
	}
	if (!(fo = fopen(argv[3], "wb"))) {
		perror("fopen(ouput)");
		return 1;
	}
	for(;;) {
		c = getc(fi);
		if (feof(fi)) {
			break;
		}
		if (!*kp) {
			kp = argv[1];
		}
		c ^= *(kp++);
		fputc(c, fo);
	}
	fclose(fi);
	fclose(fo);
	return 0;
}
