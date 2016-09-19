/*
 *  xor-analyze: XOR Cipher cryptanalysis program
 *  Copyright (C) 2000-2003 Thomas Habets <thomas@habets.pp.se>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public
 *  License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 * $Id: xor-analyze.cc,v 1.12 2002/04/13 01:04:03 marvin Exp $
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef WIN32_CROSS
extern "C" {
#include "getopt.h"
}
#endif

const char *version = "0.5";

class xor_analyze {
private:
	/* pass 2 */
	unsigned int keylen;
	float freq[256];

	unsigned int verbose;
	unsigned int maxval,maxvalkey;
	unsigned int kstart, kend, len, maxlen;
	unsigned char *buffer;
	
	void coincidence(void);
	int count_byte(const unsigned char *b, int l, char ch);
	void redundancy(char *, int);
public:
	bool keyalphanum;
	bool strictfreq;
	bool status;
	bool allkeys;
	char *truekey;
	float totalpoints;
	unsigned char *key;
	unsigned char *plainkey;
	void print_results(FILE *);
	int run(int ml, int ks, int ke);
	void run2(int kl);
	void run2_allkeys(int);
	xor_analyze(const unsigned char *b0, int l, float *fq, int v);
};

void xor_analyze::run2_allkeys(int keylen)
{
	float minpoints = 1024*1024;
	unsigned int minpointskey;
	int c;
	if (!(truekey = (char*)malloc(keylen+1))) {
		throw "Out of memory";
	}
	if (verbose) {
		printf("Calculating all keys between kstart and kend...\n");
		printf("Key length    Points  Key\n");
	}
	for (c = kend; c >= kstart; c--) {
		float scaled_points;
		
		if (status && !verbose) {
			printf("\rCalculating all keys between kstart and "
			       "kend... %d / %d", (kend-kstart)-c+2,
			       kend - kstart+1);
			fflush(stdout);
		}
		run2(c);
		
		scaled_points = (float)(totalpoints/c);
		
		if (minpoints > scaled_points) {
			minpoints = scaled_points;
			minpointskey = c;
			memcpy(truekey, plainkey, c);
			truekey[c] = 0;
		}
		if (verbose) {
			printf("%10d  %8.2f \"%s\"\n", 
				c, scaled_points,
				plainkey);
		}
	}
	if (status && !verbose) {
		printf("\n");
	}
}

void xor_analyze::redundancy(char *s, int t)
{
	int c,d;
	char iseq;

	for (c = t; c-1; c--) {
		if ((status || verbose) && !allkeys) {
			printf("\rChecking redundancy...");
			fflush(stdout);
		}
		if (strlen(s) % c)
			continue;
		iseq = 1;
		for (d = 0; d < c; d++) {
			if (memcmp(s, &s[d * strlen(s)/c], strlen(s)/c)) {
				iseq = 0;
				break;
			}
		}
		if (iseq) {
			keylen = strlen(s)/c;
			if ((status || verbose) && !allkeys) {
				printf(" key shortened by %d to length %d\n",
				       c, keylen);
			}
			plainkey[strlen(s)/c] = 0;
			return;
		} else {
			if ((status || verbose) && !allkeys) {
				printf(" %6.2f %%",
				       100*(float)(t-c+2)/(float)t);
				fflush(stdout);
			}
		}
	}
	if ((status || verbose) && !allkeys) {
		printf("\n");
	}
}


void xor_analyze::run2(int kl = 0)
{
	float points[256]; /* points for each key */
	float bc[256]; /* bytecount */
	unsigned int kc;
	if (kl) {
		keylen = kl;
	} else {
		keylen = maxvalkey;
	}
	if (!(plainkey = (unsigned char*)malloc(keylen+1))) {
		throw "out of memory";
	}
	totalpoints = 0;
	for (kc = 0; kc < keylen; kc++) {
		if ((status && !verbose) && !allkeys) {
			printf("\rFinding key based on byte frequency... "
			       "%d / %d", kc+1, keylen);
			fflush(stdout);
		}
		memset(points, 0, sizeof(points));
		for (int k = 0; k < 256; k++) {
			memset(bc, 0, sizeof(float)*256);
			/*
			 * statistically check each 8-bit key
			 */
			float l = 0;
			for (unsigned int pos = kc; pos < len; pos += keylen) {
				bc[buffer[pos]^k] += 1/(float)(len / keylen);
				l += bc[buffer[pos]^k];
				
			}
			/*
			 * give points for bad hits
			 */
			for (int c = 0; c < 256; c++) {
				points[k] += fabs(bc[c] - freq[c]);
				if (strictfreq) {
				/* assume if not in freq then not in plain */
					if (!freq[c] && bc[c]) {
						points[k] += 100;
					}
				}
				if (keyalphanum) {
				/* remove non-alphanumeric passwords */
					if(!strchr("abcdefghijklmnopqrstuvwqyz"
						   "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
						   "0123456789 ", k)) {
						points[k] += 100;
					}
				}
			}
		}
		float minval = 1024*1024;
		unsigned int min;
		for (int c = 0; c < 256; c++) {
			if (minval > points[c]) {
				minval = points[c];
				min = c;
			}
		}
		if (verbose > 1) {
			printf("Partial key (part %d / %d) = %c (%f points)\n",
			       kc+1, keylen, min, minval);
		}
		totalpoints += minval;
		plainkey[kc] = min;
	}
	plainkey[kc] = 0;
	if ((status && !verbose) && !allkeys) {
		printf("\n");
	}
	redundancy((char*)plainkey, keylen);
}

xor_analyze::xor_analyze(const unsigned char *b0, int l, float *fq, int v = 0)
{
	len = l;
	verbose = v;
	memcpy(freq, fq, sizeof(float)*256);
	strictfreq = 0;
	keyalphanum = 0;
	status = 1;
	maxval = 0;

	if (!((buffer = (unsigned char*)malloc(len))
	      && (key = (unsigned char*)malloc(len)))) {
		throw "out of memory";
	}
	memcpy(buffer, b0, len);
}

int xor_analyze::run(int ml, int ks, int ke)
{
	maxlen = ml;
	kstart = ks;
	kend = ke;
	coincidence();
	return maxvalkey;
}

void xor_analyze::coincidence(void)
{
	unsigned int i,ofs;
	unsigned int end;
	unsigned char *cbuffer; /* coincidence buffer */

	if (!(cbuffer = (unsigned char *)malloc(len))) {
		throw "out of memory";
	}

	end = (kend < len) ? kend : len;
	for (ofs = kstart; ofs <= end; ofs++) {
		if (status || verbose) {
			printf("\rCounting coincidences... %d / %d",
			       ofs-kstart+1, end-kstart+1);
			fflush(stdout);
		}
		for(i = 0; i < len; i++) {
			cbuffer[i] = buffer[i] ^ buffer[(i + ofs) % len];
		}
		key[ofs-kstart] = count_byte(cbuffer, len, 0);
		if (key[ofs-kstart] > maxval) {
			maxval = key[ofs-kstart];
			maxvalkey = ofs;
		}
	}
	free(cbuffer);
	if (status || verbose) {
		printf("\nKey length is probably %d (or a factor of it)\n",
		       maxvalkey);
	}
}

int xor_analyze::count_byte(const unsigned char *b, int l, char ch)
{
	int i;
	int r = 0;
	for (i = 0; i < l; i++) {
		if (b[i] == ch)
			r++;
	}
	return r;
}

void xor_analyze::print_results(FILE *stream)
{
	unsigned int i;
	fprintf(stream,	"Key length  Coincidents       Bytes    Coincidents"
		" (in percent)\n");
	for (i = 0; i <= ((kend - kstart < len) ? kend-kstart : len-1); i++) {
		fprintf(stream, "%10d %12d  / %8d =      %6.2f %%",
		       i + kstart, key[i], len,
		       100*(float)key[i] / (float)len);
		if (key[i] == maxval) {
			fprintf(stream, " (winner)");
		}
		fprintf(stream, "\n");
	}
}

void usage(int err)
{
	printf("usage: xor-analyze [ options ] <encrypted file> "
	       "[ <frequency table> ]\n\n"
	       "Frequency table is not needed if in length-only mode (-l)\n"
	       "Options:\n"
	       "   -m <number>  Minimum key length (default: 1)\n"
	       "   -M <number>  Maximum key length (default: 20)\n"
	       "   -v           Increment verbosity level (default: 0)\n"
	       "   -q           Quiet mode\n"
	       "   -h           Show this help text\n"
	       "   -k <number>  Set key length\n"
	       "   -s <number>  Maximum size to load (default: 1000)\n"
	       "   -a           Run statistict on all keylengths less than "
	       "the -M value\n"
	       "   -l           Only run first pass which finds password "
	       "length\n"
	       "   -f           Don't be strict about zero frequency\n"
	       "   -n           Enable all chars in key, not just alphanumeric"
	       "\n");
	exit(err);
}

int main(int argc, char **argv)
{
	xor_analyze *analyze;
	FILE *f, *fq;
	unsigned int maxlen = 1000;
	unsigned int kstart = 1, kend = 20;
	int verbose = 0;
	unsigned int len;
	int c;
	unsigned char *buffer;
	float freq[256];
	unsigned int keylen = 0;
	char allkeys = 0;
	char lengthonly = 0;
	char keyalphanum = 1,
		strictfreq = 1,
		status = 1;
	
        while ((c = getopt(argc, argv, "vhm:M:k:s:alqfn")) != EOF) {
                switch (c) {
		case 'q':
			status = 0;
			break;
                case 'v':
                        verbose++;
                        break;
                case 'h':
                        usage(0);
		case 'm':
			kstart = atoi(optarg);
			break;
		case 'M':
			kend = atoi(optarg);
			break;
		case 'k':
			keylen = atoi(optarg);
			break;
		case 's':
			maxlen = atoi(optarg);
			break;
		case 'a':
			allkeys = 1;
			break;
		case 'l':
			lengthonly = 1;
			break;
		case 'f':
			strictfreq = 0;
			break;
		case 'n':
			keyalphanum = 0;
			break;
                default:
                        usage(1);
                }
        }
	if (status) {
		printf("xor-analyze version %s by Thomas Habets <thomas@habets.pp.se>\n",
		       version);
	}

        if (optind + 1 + (1-lengthonly) != argc) {
		fprintf(stderr, "Wrong number of args\n");
                usage(1);
                exit(1);
        }

	if (!(f = fopen(argv[optind], "rb"))) {
		perror("fopen(input file)");
		return 1;
	}

	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);
	
	if (len > maxlen) {
		len = maxlen;
	}

	if (!(buffer = (unsigned char *)malloc(len))) {
		perror("malloc()");
		return 1;
	}

	fread(buffer, 1, len,f);
	fclose(f);

	if (!lengthonly) {
		if (!(fq = fopen(argv[optind+1], "r"))) {
			perror("fopen(frequency file)");
			return 1;
		}
		for (int c = 0; c < 256; c++) {
			fscanf(fq, "%f\n", &freq[c]);
		}
		
		fclose(fq);
	}


	
	analyze = new xor_analyze(buffer, len, freq, verbose);
	analyze->keyalphanum = keyalphanum;
	analyze->strictfreq = strictfreq;
	analyze->status = status;
	analyze->allkeys = allkeys;

	if (!keylen) {
		keylen = analyze->run(maxlen, kstart, kend);
	}
	if (verbose) {
		analyze->print_results(stdout);
	}
	if (lengthonly) {
		return 0;
	}

	if (allkeys) {
                analyze->run2_allkeys(keylen);
		printf("Probable key: \"%s\"\n", analyze->truekey);
	} else {
		analyze->run2(keylen);
		printf("Probable key: \"%s\"\n", analyze->plainkey);
	}

	return 0;
}
