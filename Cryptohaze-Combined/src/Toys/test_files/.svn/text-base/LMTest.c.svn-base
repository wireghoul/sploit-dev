// Testing LM hash operations  with the FRT code

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define  ITERATIONS    16


  int shifts2[16] = { 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0 };

  unsigned int des_skb[8][64] = {
{
/* for C bits (numbered as per FIPS 46) 1 2 3 4 5 6 */
0x00000000L,0x00000010L,0x20000000L,0x20000010L,
0x00010000L,0x00010010L,0x20010000L,0x20010010L,
0x00000800L,0x00000810L,0x20000800L,0x20000810L,
0x00010800L,0x00010810L,0x20010800L,0x20010810L,
0x00000020L,0x00000030L,0x20000020L,0x20000030L,
0x00010020L,0x00010030L,0x20010020L,0x20010030L,
0x00000820L,0x00000830L,0x20000820L,0x20000830L,
0x00010820L,0x00010830L,0x20010820L,0x20010830L,
0x00080000L,0x00080010L,0x20080000L,0x20080010L,
0x00090000L,0x00090010L,0x20090000L,0x20090010L,
0x00080800L,0x00080810L,0x20080800L,0x20080810L,
0x00090800L,0x00090810L,0x20090800L,0x20090810L,
0x00080020L,0x00080030L,0x20080020L,0x20080030L,
0x00090020L,0x00090030L,0x20090020L,0x20090030L,
0x00080820L,0x00080830L,0x20080820L,0x20080830L,
0x00090820L,0x00090830L,0x20090820L,0x20090830L,
},{
/* for C bits (numbered as per FIPS 46) 7 8 10 11 12 13 */
0x00000000L,0x02000000L,0x00002000L,0x02002000L,
0x00200000L,0x02200000L,0x00202000L,0x02202000L,
0x00000004L,0x02000004L,0x00002004L,0x02002004L,
0x00200004L,0x02200004L,0x00202004L,0x02202004L,
0x00000400L,0x02000400L,0x00002400L,0x02002400L,
0x00200400L,0x02200400L,0x00202400L,0x02202400L,
0x00000404L,0x02000404L,0x00002404L,0x02002404L,
0x00200404L,0x02200404L,0x00202404L,0x02202404L,
0x10000000L,0x12000000L,0x10002000L,0x12002000L,
0x10200000L,0x12200000L,0x10202000L,0x12202000L,
0x10000004L,0x12000004L,0x10002004L,0x12002004L,
0x10200004L,0x12200004L,0x10202004L,0x12202004L,
0x10000400L,0x12000400L,0x10002400L,0x12002400L,
0x10200400L,0x12200400L,0x10202400L,0x12202400L,
0x10000404L,0x12000404L,0x10002404L,0x12002404L,
0x10200404L,0x12200404L,0x10202404L,0x12202404L,
},{
/* for C bits (numbered as per FIPS 46) 14 15 16 17 19 20 */
0x00000000L,0x00000001L,0x00040000L,0x00040001L,
0x01000000L,0x01000001L,0x01040000L,0x01040001L,
0x00000002L,0x00000003L,0x00040002L,0x00040003L,
0x01000002L,0x01000003L,0x01040002L,0x01040003L,
0x00000200L,0x00000201L,0x00040200L,0x00040201L,
0x01000200L,0x01000201L,0x01040200L,0x01040201L,
0x00000202L,0x00000203L,0x00040202L,0x00040203L,
0x01000202L,0x01000203L,0x01040202L,0x01040203L,
0x08000000L,0x08000001L,0x08040000L,0x08040001L,
0x09000000L,0x09000001L,0x09040000L,0x09040001L,
0x08000002L,0x08000003L,0x08040002L,0x08040003L,
0x09000002L,0x09000003L,0x09040002L,0x09040003L,
0x08000200L,0x08000201L,0x08040200L,0x08040201L,
0x09000200L,0x09000201L,0x09040200L,0x09040201L,
0x08000202L,0x08000203L,0x08040202L,0x08040203L,
0x09000202L,0x09000203L,0x09040202L,0x09040203L,
},{
/* for C bits (numbered as per FIPS 46) 21 23 24 26 27 28 */
0x00000000L,0x00100000L,0x00000100L,0x00100100L,
0x00000008L,0x00100008L,0x00000108L,0x00100108L,
0x00001000L,0x00101000L,0x00001100L,0x00101100L,
0x00001008L,0x00101008L,0x00001108L,0x00101108L,
0x04000000L,0x04100000L,0x04000100L,0x04100100L,
0x04000008L,0x04100008L,0x04000108L,0x04100108L,
0x04001000L,0x04101000L,0x04001100L,0x04101100L,
0x04001008L,0x04101008L,0x04001108L,0x04101108L,
0x00020000L,0x00120000L,0x00020100L,0x00120100L,
0x00020008L,0x00120008L,0x00020108L,0x00120108L,
0x00021000L,0x00121000L,0x00021100L,0x00121100L,
0x00021008L,0x00121008L,0x00021108L,0x00121108L,
0x04020000L,0x04120000L,0x04020100L,0x04120100L,
0x04020008L,0x04120008L,0x04020108L,0x04120108L,
0x04021000L,0x04121000L,0x04021100L,0x04121100L,
0x04021008L,0x04121008L,0x04021108L,0x04121108L,
},{
/* for D bits (numbered as per FIPS 46) 1 2 3 4 5 6 */
0x00000000L,0x10000000L,0x00010000L,0x10010000L,
0x00000004L,0x10000004L,0x00010004L,0x10010004L,
0x20000000L,0x30000000L,0x20010000L,0x30010000L,
0x20000004L,0x30000004L,0x20010004L,0x30010004L,
0x00100000L,0x10100000L,0x00110000L,0x10110000L,
0x00100004L,0x10100004L,0x00110004L,0x10110004L,
0x20100000L,0x30100000L,0x20110000L,0x30110000L,
0x20100004L,0x30100004L,0x20110004L,0x30110004L,
0x00001000L,0x10001000L,0x00011000L,0x10011000L,
0x00001004L,0x10001004L,0x00011004L,0x10011004L,
0x20001000L,0x30001000L,0x20011000L,0x30011000L,
0x20001004L,0x30001004L,0x20011004L,0x30011004L,
0x00101000L,0x10101000L,0x00111000L,0x10111000L,
0x00101004L,0x10101004L,0x00111004L,0x10111004L,
0x20101000L,0x30101000L,0x20111000L,0x30111000L,
0x20101004L,0x30101004L,0x20111004L,0x30111004L,
},{
/* for D bits (numbered as per FIPS 46) 8 9 11 12 13 14 */
0x00000000L,0x08000000L,0x00000008L,0x08000008L,
0x00000400L,0x08000400L,0x00000408L,0x08000408L,
0x00020000L,0x08020000L,0x00020008L,0x08020008L,
0x00020400L,0x08020400L,0x00020408L,0x08020408L,
0x00000001L,0x08000001L,0x00000009L,0x08000009L,
0x00000401L,0x08000401L,0x00000409L,0x08000409L,
0x00020001L,0x08020001L,0x00020009L,0x08020009L,
0x00020401L,0x08020401L,0x00020409L,0x08020409L,
0x02000000L,0x0A000000L,0x02000008L,0x0A000008L,
0x02000400L,0x0A000400L,0x02000408L,0x0A000408L,
0x02020000L,0x0A020000L,0x02020008L,0x0A020008L,
0x02020400L,0x0A020400L,0x02020408L,0x0A020408L,
0x02000001L,0x0A000001L,0x02000009L,0x0A000009L,
0x02000401L,0x0A000401L,0x02000409L,0x0A000409L,
0x02020001L,0x0A020001L,0x02020009L,0x0A020009L,
0x02020401L,0x0A020401L,0x02020409L,0x0A020409L,
},{
/* for D bits (numbered as per FIPS 46) 16 17 18 19 20 21 */
0x00000000L,0x00000100L,0x00080000L,0x00080100L,
0x01000000L,0x01000100L,0x01080000L,0x01080100L,
0x00000010L,0x00000110L,0x00080010L,0x00080110L,
0x01000010L,0x01000110L,0x01080010L,0x01080110L,
0x00200000L,0x00200100L,0x00280000L,0x00280100L,
0x01200000L,0x01200100L,0x01280000L,0x01280100L,
0x00200010L,0x00200110L,0x00280010L,0x00280110L,
0x01200010L,0x01200110L,0x01280010L,0x01280110L,
0x00000200L,0x00000300L,0x00080200L,0x00080300L,
0x01000200L,0x01000300L,0x01080200L,0x01080300L,
0x00000210L,0x00000310L,0x00080210L,0x00080310L,
0x01000210L,0x01000310L,0x01080210L,0x01080310L,
0x00200200L,0x00200300L,0x00280200L,0x00280300L,
0x01200200L,0x01200300L,0x01280200L,0x01280300L,
0x00200210L,0x00200310L,0x00280210L,0x00280310L,
0x01200210L,0x01200310L,0x01280210L,0x01280310L,
},{
/* for D bits (numbered as per FIPS 46) 22 23 24 25 27 28 */
0x00000000L,0x04000000L,0x00040000L,0x04040000L,
0x00000002L,0x04000002L,0x00040002L,0x04040002L,
0x00002000L,0x04002000L,0x00042000L,0x04042000L,
0x00002002L,0x04002002L,0x00042002L,0x04042002L,
0x00000020L,0x04000020L,0x00040020L,0x04040020L,
0x00000022L,0x04000022L,0x00040022L,0x04040022L,
0x00002020L,0x04002020L,0x00042020L,0x04042020L,
0x00002022L,0x04002022L,0x00042022L,0x04042022L,
0x00000800L,0x04000800L,0x00040800L,0x04040800L,
0x00000802L,0x04000802L,0x00040802L,0x04040802L,
0x00002800L,0x04002800L,0x00042800L,0x04042800L,
0x00002802L,0x04002802L,0x00042802L,0x04042802L,
0x00000820L,0x04000820L,0x00040820L,0x04040820L,
0x00000822L,0x04000822L,0x00040822L,0x04040822L,
0x00002820L,0x04002820L,0x00042820L,0x04042820L,
0x00002822L,0x04002822L,0x00042822L,0x04042822L,
}};

  unsigned int des_SPtrans[8][64] = {
{
/* nibble 0 */
0x02080800L, 0x00080000L, 0x02000002L, 0x02080802L,
0x02000000L, 0x00080802L, 0x00080002L, 0x02000002L,
0x00080802L, 0x02080800L, 0x02080000L, 0x00000802L,
0x02000802L, 0x02000000L, 0x00000000L, 0x00080002L,
0x00080000L, 0x00000002L, 0x02000800L, 0x00080800L,
0x02080802L, 0x02080000L, 0x00000802L, 0x02000800L,
0x00000002L, 0x00000800L, 0x00080800L, 0x02080002L,
0x00000800L, 0x02000802L, 0x02080002L, 0x00000000L,
0x00000000L, 0x02080802L, 0x02000800L, 0x00080002L,
0x02080800L, 0x00080000L, 0x00000802L, 0x02000800L,
0x02080002L, 0x00000800L, 0x00080800L, 0x02000002L,
0x00080802L, 0x00000002L, 0x02000002L, 0x02080000L,
0x02080802L, 0x00080800L, 0x02080000L, 0x02000802L,
0x02000000L, 0x00000802L, 0x00080002L, 0x00000000L,
0x00080000L, 0x02000000L, 0x02000802L, 0x02080800L,
0x00000002L, 0x02080002L, 0x00000800L, 0x00080802L,
},{
/* nibble 1 */
0x40108010L, 0x00000000L, 0x00108000L, 0x40100000L,
0x40000010L, 0x00008010L, 0x40008000L, 0x00108000L,
0x00008000L, 0x40100010L, 0x00000010L, 0x40008000L,
0x00100010L, 0x40108000L, 0x40100000L, 0x00000010L,
0x00100000L, 0x40008010L, 0x40100010L, 0x00008000L,
0x00108010L, 0x40000000L, 0x00000000L, 0x00100010L,
0x40008010L, 0x00108010L, 0x40108000L, 0x40000010L,
0x40000000L, 0x00100000L, 0x00008010L, 0x40108010L,
0x00100010L, 0x40108000L, 0x40008000L, 0x00108010L,
0x40108010L, 0x00100010L, 0x40000010L, 0x00000000L,
0x40000000L, 0x00008010L, 0x00100000L, 0x40100010L,
0x00008000L, 0x40000000L, 0x00108010L, 0x40008010L,
0x40108000L, 0x00008000L, 0x00000000L, 0x40000010L,
0x00000010L, 0x40108010L, 0x00108000L, 0x40100000L,
0x40100010L, 0x00100000L, 0x00008010L, 0x40008000L,
0x40008010L, 0x00000010L, 0x40100000L, 0x00108000L,
},{
/* nibble 2 */
0x04000001L, 0x04040100L, 0x00000100L, 0x04000101L,
0x00040001L, 0x04000000L, 0x04000101L, 0x00040100L,
0x04000100L, 0x00040000L, 0x04040000L, 0x00000001L,
0x04040101L, 0x00000101L, 0x00000001L, 0x04040001L,
0x00000000L, 0x00040001L, 0x04040100L, 0x00000100L,
0x00000101L, 0x04040101L, 0x00040000L, 0x04000001L,
0x04040001L, 0x04000100L, 0x00040101L, 0x04040000L,
0x00040100L, 0x00000000L, 0x04000000L, 0x00040101L,
0x04040100L, 0x00000100L, 0x00000001L, 0x00040000L,
0x00000101L, 0x00040001L, 0x04040000L, 0x04000101L,
0x00000000L, 0x04040100L, 0x00040100L, 0x04040001L,
0x00040001L, 0x04000000L, 0x04040101L, 0x00000001L,
0x00040101L, 0x04000001L, 0x04000000L, 0x04040101L,
0x00040000L, 0x04000100L, 0x04000101L, 0x00040100L,
0x04000100L, 0x00000000L, 0x04040001L, 0x00000101L,
0x04000001L, 0x00040101L, 0x00000100L, 0x04040000L,
},{
/* nibble 3 */
0x00401008L, 0x10001000L, 0x00000008L, 0x10401008L,
0x00000000L, 0x10400000L, 0x10001008L, 0x00400008L,
0x10401000L, 0x10000008L, 0x10000000L, 0x00001008L,
0x10000008L, 0x00401008L, 0x00400000L, 0x10000000L,
0x10400008L, 0x00401000L, 0x00001000L, 0x00000008L,
0x00401000L, 0x10001008L, 0x10400000L, 0x00001000L,
0x00001008L, 0x00000000L, 0x00400008L, 0x10401000L,
0x10001000L, 0x10400008L, 0x10401008L, 0x00400000L,
0x10400008L, 0x00001008L, 0x00400000L, 0x10000008L,
0x00401000L, 0x10001000L, 0x00000008L, 0x10400000L,
0x10001008L, 0x00000000L, 0x00001000L, 0x00400008L,
0x00000000L, 0x10400008L, 0x10401000L, 0x00001000L,
0x10000000L, 0x10401008L, 0x00401008L, 0x00400000L,
0x10401008L, 0x00000008L, 0x10001000L, 0x00401008L,
0x00400008L, 0x00401000L, 0x10400000L, 0x10001008L,
0x00001008L, 0x10000000L, 0x10000008L, 0x10401000L,
},{
/* nibble 4 */
0x08000000L, 0x00010000L, 0x00000400L, 0x08010420L,
0x08010020L, 0x08000400L, 0x00010420L, 0x08010000L,
0x00010000L, 0x00000020L, 0x08000020L, 0x00010400L,
0x08000420L, 0x08010020L, 0x08010400L, 0x00000000L,
0x00010400L, 0x08000000L, 0x00010020L, 0x00000420L,
0x08000400L, 0x00010420L, 0x00000000L, 0x08000020L,
0x00000020L, 0x08000420L, 0x08010420L, 0x00010020L,
0x08010000L, 0x00000400L, 0x00000420L, 0x08010400L,
0x08010400L, 0x08000420L, 0x00010020L, 0x08010000L,
0x00010000L, 0x00000020L, 0x08000020L, 0x08000400L,
0x08000000L, 0x00010400L, 0x08010420L, 0x00000000L,
0x00010420L, 0x08000000L, 0x00000400L, 0x00010020L,
0x08000420L, 0x00000400L, 0x00000000L, 0x08010420L,
0x08010020L, 0x08010400L, 0x00000420L, 0x00010000L,
0x00010400L, 0x08010020L, 0x08000400L, 0x00000420L,
0x00000020L, 0x00010420L, 0x08010000L, 0x08000020L,
},{
/* nibble 5 */
0x80000040L, 0x00200040L, 0x00000000L, 0x80202000L,
0x00200040L, 0x00002000L, 0x80002040L, 0x00200000L,
0x00002040L, 0x80202040L, 0x00202000L, 0x80000000L,
0x80002000L, 0x80000040L, 0x80200000L, 0x00202040L,
0x00200000L, 0x80002040L, 0x80200040L, 0x00000000L,
0x00002000L, 0x00000040L, 0x80202000L, 0x80200040L,
0x80202040L, 0x80200000L, 0x80000000L, 0x00002040L,
0x00000040L, 0x00202000L, 0x00202040L, 0x80002000L,
0x00002040L, 0x80000000L, 0x80002000L, 0x00202040L,
0x80202000L, 0x00200040L, 0x00000000L, 0x80002000L,
0x80000000L, 0x00002000L, 0x80200040L, 0x00200000L,
0x00200040L, 0x80202040L, 0x00202000L, 0x00000040L,
0x80202040L, 0x00202000L, 0x00200000L, 0x80002040L,
0x80000040L, 0x80200000L, 0x00202040L, 0x00000000L,
0x00002000L, 0x80000040L, 0x80002040L, 0x80202000L,
0x80200000L, 0x00002040L, 0x00000040L, 0x80200040L,
},{
/* nibble 6 */
0x00004000L, 0x00000200L, 0x01000200L, 0x01000004L,
0x01004204L, 0x00004004L, 0x00004200L, 0x00000000L,
0x01000000L, 0x01000204L, 0x00000204L, 0x01004000L,
0x00000004L, 0x01004200L, 0x01004000L, 0x00000204L,
0x01000204L, 0x00004000L, 0x00004004L, 0x01004204L,
0x00000000L, 0x01000200L, 0x01000004L, 0x00004200L,
0x01004004L, 0x00004204L, 0x01004200L, 0x00000004L,
0x00004204L, 0x01004004L, 0x00000200L, 0x01000000L,
0x00004204L, 0x01004000L, 0x01004004L, 0x00000204L,
0x00004000L, 0x00000200L, 0x01000000L, 0x01004004L,
0x01000204L, 0x00004204L, 0x00004200L, 0x00000000L,
0x00000200L, 0x01000004L, 0x00000004L, 0x01000200L,
0x00000000L, 0x01000204L, 0x01000200L, 0x00004200L,
0x00000204L, 0x00004000L, 0x01004204L, 0x01000000L,
0x01004200L, 0x00000004L, 0x00004004L, 0x01004204L,
0x01000004L, 0x01004200L, 0x01004000L, 0x00004004L,
},{
/* nibble 7 */
0x20800080L, 0x20820000L, 0x00020080L, 0x00000000L,
0x20020000L, 0x00800080L, 0x20800000L, 0x20820080L,
0x00000080L, 0x20000000L, 0x00820000L, 0x00020080L,
0x00820080L, 0x20020080L, 0x20000080L, 0x20800000L,
0x00020000L, 0x00820080L, 0x00800080L, 0x20020000L,
0x20820080L, 0x20000080L, 0x00000000L, 0x00820000L,
0x20000000L, 0x00800000L, 0x20020080L, 0x20800080L,
0x00800000L, 0x00020000L, 0x20820000L, 0x00000080L,
0x00800000L, 0x00020000L, 0x20000080L, 0x20820080L,
0x00020080L, 0x20000000L, 0x00000000L, 0x00820000L,
0x20800080L, 0x20020080L, 0x20020000L, 0x00800080L,
0x20820000L, 0x00000080L, 0x00800080L, 0x20020000L,
0x20820080L, 0x00800000L, 0x20800000L, 0x20000080L,
0x00820000L, 0x00020080L, 0x20020080L, 0x20800000L,
0x00000080L, 0x20820000L, 0x00820080L, 0x00000000L,
0x20000000L, 0x20800080L, 0x00020000L, 0x00820080L,
}};


 void PERM_OP(int ia, int ib, int it, unsigned int n, unsigned int m, unsigned int* data) {
	data[it] =((data[ia] >> n ) ^ data[ib]) & m;
	data[ib] ^= data[it];
	data[ia] ^= data[it] << n;
}

 void HPERM_OP(int ia, int it, int n, unsigned int m, unsigned int* data) {
	data[it] = ((data[ia] << (16-n)) ^ data[ia]) & m;
	data[ia] = data[ia] ^ data[it] ^ (data[it]>>(16-n));
}

 void IP(int il, int ir, int it, unsigned int* data) {
	PERM_OP(ir, il, it, 4, 0x0f0f0f0f, data);
	PERM_OP(il, ir, it, 16, 0x0000ffff, data);
	PERM_OP(ir, il, it, 2, 0x33333333, data);
	PERM_OP(il, ir, it, 8, 0x00ff00ff, data);
	PERM_OP(ir, il, it, 1, 0x55555555, data);
}

void FP(int il, int ir, int it, unsigned int* data) {
	PERM_OP(il, ir, it, 1, 0x55555555, data);
	PERM_OP(ir, il, it, 8, 0x00ff00ff, data);
	PERM_OP(il, ir, it, 2, 0x33333333, data);
	PERM_OP(ir, il, it, 16, 0x0000ffff, data);
	PERM_OP(il, ir, it, 4, 0x0f0f0f0f, data);
}

unsigned int D_ENCRYPT(unsigned int ll, unsigned int uu, unsigned int tt) {
	tt = (tt>>4)|(tt<<28);
	return ll ^ des_SPtrans[0][(uu>>2)&0x3f] ^
			des_SPtrans[2][(uu>>10)&0x3f] ^
			des_SPtrans[4][(uu>>18)&0x3f] ^
			des_SPtrans[6][(uu>>26)&0x3f] ^
			des_SPtrans[1][(tt>>2)&0x3f] ^
			des_SPtrans[3][(tt>>10)&0x3f] ^
			des_SPtrans[5][(tt>>18)&0x3f] ^
			des_SPtrans[7][(tt>>26)&0x3f];
}

void printData(const char *Name, unsigned char *data, int length) {
    int i;
    printf("%s: ", Name);
    for (i = 0 ; i < length; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

#define New_PERM_OP(data_a, data_b, data_temp, n, m) { \
        data_temp =((data_a >> n ) ^ data_b) & m; \
	data_b ^= data_temp; \
	data_a ^= data_temp << n; \
}


#define New_HPERM_OP(data_a, data_temp, n, m) { \
	data_temp = ((data_a << (16-n)) ^ data_a) & m; \
	data_a = data_a ^ data_temp ^ (data_temp>>(16-n)); \
}


#define New_IP(data_left, data_right, data_temp) { \
	New_PERM_OP(data_right, data_left, data_temp, 4, 0x0f0f0f0f); \
	New_PERM_OP(data_left, data_right, data_temp, 16, 0x0000ffff); \
	New_PERM_OP(data_right, data_left, data_temp, 2, 0x33333333); \
	New_PERM_OP(data_left, data_right, data_temp, 8, 0x00ff00ff); \
	New_PERM_OP(data_right, data_left, data_temp, 1, 0x55555555); \
}

#define New_FP(data_left, data_right, data_temp) { \
	New_PERM_OP(data_left, data_right, data_temp, 1, 0x55555555); \
	New_PERM_OP(data_right, data_left, data_temp, 8, 0x00ff00ff); \
	New_PERM_OP(data_left, data_right, data_temp, 2, 0x33333333); \
	New_PERM_OP(data_right, data_left, data_temp, 16, 0x0000ffff); \
	New_PERM_OP(data_left, data_right, data_temp, 4, 0x0f0f0f0f); \
}


#define SHIDX(value) value


void cudaLM(uint32_t &b0, uint32_t &b1, uint32_t &hash0, uint32_t &hash1) {
    uint32_t data0, data1, datatemp;
    uint32_t key0, key1, keytemp, roundkey0, roundkey1;
    uint32_t ii, rs, rt;

    // Init data to post-IP mixing
    data0 = 0x2400b807;
    data1 = 0xaa190747;

    key0 = ((((b0 >> 16) & 0xff) << 5) | (((b0 >> 24) & 0xff) >> 3))&0xff;
    key0 = (key0<<8) | (((((b0 >> 8) & 0xff) << 6) | (((b0 >> 16) & 0xff) >> 2))&0xff);
    key0 = (key0<<8) | ((((b0 & 0xff) << 7) | (((b0 >> 8) & 0xff) >> 1))&0xff);
    key0 = (key0<<8) | (b0 & 0xff);

    key1 = (((b1 >> 16) & 0xff) << 1)&0xff;
    key1 = (key1<<8) | (((((b1 >> 8) & 0xff) << 2) | (((b1 >> 16) & 0xff) >> 6))&0xff);
    key1 = (key1<<8) | ((((b1 & 0xff) << 3) | (((b1 >> 8) & 0xff) >> 5))&0xff);
    key1 = (key1<<8) | (((((b0 >> 24) & 0xff) << 4) | ((b1 & 0xff) >> 4))&0xff);

    //printf("b0 key0:     %08x %08x\n", key0, key1);

    New_PERM_OP(key1, key0, keytemp, 4, 0x0f0f0f0f);
    New_HPERM_OP(key0, keytemp, -2, 0xcccc0000);
    New_HPERM_OP(key1, keytemp, -2, 0xcccc0000);
    New_PERM_OP(key1, key0, keytemp, 1, 0x55555555);
    New_PERM_OP(key0, key1, keytemp, 8, 0x00ff00ff);
    New_PERM_OP(key1, key0, keytemp, 1, 0x55555555);

    //printf("IP key: %08x %08x\n", key0, key1);


    key1 = ((key1&0x000000ff)<<16) | (key1&0x0000ff00) | ((key1&0x00ff0000)>>16) | ((key0&0xf0000000)>>4);
    key0 &= 0x0fffffff;

    //printf("Trimmed key: %08x %08x\n", key0, key1);

    for(ii = 0; ii < 16; ii+=1) {
        if(shifts2[ii]) {
                key0 = ((key0>>2)|(key0<<26));
                key1 =((key1>>2)|(key1<<26));
        } else {
                key0 = ((key0>>1)|(key0<<27));
                key1 = ((key1>>1)|(key1<<27));
        }
        key0 &= 0x0fffffff;
        key1 &= 0x0fffffff;

        rs = des_skb[0][key0&0x3f] |
                des_skb[1][((key0>>6)&0x03)|((key0>>7)&0x3c)] |
                des_skb[2][((key0>>13)&0x0f)|((key0>>14)&0x30)] |
                des_skb[3][((key0>>20)&0x01)|((key0>>21)&0x06) |
                ((key0>>22)&0x38)];
        rt = des_skb[4][key1&0x3f] |
                des_skb[5][((key1>>7)&0x03)|((key1>>8)&0x3c)] |
                des_skb[6][(key1>>15)&0x3f] |
                des_skb[7][((key1>>21)&0x0f)|((key1>>22)&0x30)];

        /* table contained 0213 4657 */
        roundkey0 = (rt<<16)|(rs&0x0000ffff);
        roundkey0 = (roundkey0>>30)|(roundkey0<<2);
        //printf("roundkey0: %08x\n", roundkey0);
        roundkey1 = (rs>>16)|(rt&0xffff0000);
        roundkey1 = (roundkey1>>26)|(roundkey1<<6);
        //printf("roundkey1: %08x\n", roundkey1);



        data1 = D_ENCRYPT(data1, data0^roundkey0, data0^roundkey1);
        //printf("After Round %d: \n", ii);
        //printf("Data0: %08x\n", data0);
        //printf("Data1: %08x\n", data1);
        datatemp = data0;
        data0 = data1;
        data1 = datatemp;
    }

    data0 = ((data0>>3)|(data0<<29));
    data1 = ((data1>>3)|(data1<<29));
    New_FP(data0, data1, datatemp);

    datatemp = data0;
    data0 = data1;
    data1 = datatemp;

    hash0 = data0;
    hash1 = data1;
    printf("Final: %08x %08x\n", data0, data1);
}


int main() {
    printf("LM algo testing\n");

    unsigned char password[7];
    unsigned char hash[8];
    unsigned char data[8];

    uint32_t hData[128], hData2[128], hData3[128];

    uint32_t data0, data1, datatemp;
    uint32_t key0, key1, keytemp, roundkey0, roundkey1;

    uint32_t uiVal, uiDiv;
    uint32_t ii, jj, rs, rt, idx;


    uint32_t b0, b1;

    memset(hData, 0, 128 * sizeof(uint32_t));
    memset(hData2, 0, 128 * sizeof(uint32_t));
    memset(hData3, 0, 128 * sizeof(uint32_t));

    hData[8] = 'A';
    hData[9] = 'B';
    hData[10] = 'C';
    hData[11] = 'D';
    hData[12] = 'E';
    hData[13] = 'F';
    hData[14] = 'G';



	// set key
	ii = 255;
	uiVal = ((hData[SHIDX(10)] << 5) | (hData[SHIDX(11)] >> 3))&ii;
        uiVal = (uiVal<<8) | (((hData[SHIDX(9)] << 6) | (hData[SHIDX(10)] >> 2))&ii);
	uiVal = (uiVal<<8) | (((hData[SHIDX(8)] << 7) | (hData[SHIDX(9)] >> 1))&ii);
	uiVal = (uiVal<<8) | hData[SHIDX(8)];

	uiDiv = (hData[SHIDX(14)] << 1)&ii;
	uiDiv = (uiDiv<<8) | (((hData[SHIDX(13)] << 2) | (hData[SHIDX(14)] >> 6))&ii);
	uiDiv = (uiDiv<<8) | (((hData[SHIDX(12)] << 3) | (hData[SHIDX(13)] >> 5))&ii);
	uiDiv = (uiDiv<<8) | (((hData[SHIDX(11)] << 4) | (hData[SHIDX(12)] >> 4))&ii);


	hData[SHIDX(0)] = uiVal;
	hData[SHIDX(1)] = uiDiv;

        printf("Initial key: %08x %08x\n", hData[0], hData[1]);

        PERM_OP(SHIDX(1), SHIDX(0), SHIDX(2), 4, 0x0f0f0f0f, hData);
	HPERM_OP(SHIDX(0), SHIDX(2), -2, 0xcccc0000, hData);
	HPERM_OP(SHIDX(1), SHIDX(2), -2, 0xcccc0000, hData);
	PERM_OP(SHIDX(1), SHIDX(0), SHIDX(2), 1, 0x55555555, hData);
	PERM_OP(SHIDX(0), SHIDX(1), SHIDX(2), 8, 0x00ff00ff, hData);
	PERM_OP(SHIDX(1), SHIDX(0), SHIDX(2), 1, 0x55555555, hData);
	uiVal = hData[SHIDX(0)];
	uiDiv = hData[SHIDX(1)];

        printf("IP key: %08x %08x\n", hData[0], hData[1]);

        uiDiv =	((uiDiv&0x000000ff)<<16) | (uiDiv&0x0000ff00) | ((uiDiv&0x00ff0000)>>16) | ((uiVal&0xf0000000)>>4);
	uiVal &= 0x0fffffff;

        printf("Trimmed key: %08x %08x\n", uiVal, uiDiv);

	for(ii = 0; ii < ITERATIONS; ii++) {
		if(shifts2[ii]) {
			uiVal = ((uiVal>>2)|(uiVal<<26));
			uiDiv =((uiDiv>>2)|(uiDiv<<26));
		} else {
			uiVal = ((uiVal>>1)|(uiVal<<27));
			uiDiv = ((uiDiv>>1)|(uiDiv<<27));
		}
		uiVal &= 0x0fffffff;
		uiDiv &= 0x0fffffff;

		rs = des_skb[0][uiVal&0x3f] |
			des_skb[1][((uiVal>>6)&0x03)|((uiVal>>7)&0x3c)] |
			des_skb[2][((uiVal>>13)&0x0f)|((uiVal>>14)&0x30)] |
			des_skb[3][((uiVal>>20)&0x01)|((uiVal>>21)&0x06) |
			((uiVal>>22)&0x38)];
		rt = des_skb[4][uiDiv&0x3f] |
			des_skb[5][((uiDiv>>7)&0x03)|((uiDiv>>8)&0x3c)] |
			des_skb[6][(uiDiv>>15)&0x3f] |
			des_skb[7][((uiDiv>>21)&0x0f)|((uiDiv>>22)&0x30)];

		/* table contained 0213 4657 */
		idx = (rt<<16)|(rs&0x0000ffff);
		hData[SHIDX(ii)] = (idx>>30)|(idx<<2);
                printf("Key %d [0]: %08x\n", ii, hData[SHIDX(ii)]);
		idx = (rs>>16)|(rt&0xffff0000);
		hData2[SHIDX(ii)] = (idx>>26)|(idx<<6);
                printf("Key %d [1]: %08x\n", ii, hData2[SHIDX(ii)]);
	}

	// encrypt the "magic" data
	hData3[SHIDX(0)] = 0x2153474B;
	hData3[SHIDX(1)] = 0x25242340;

	IP(SHIDX(0), SHIDX(1), SHIDX(2), hData3);
	uiVal = hData3[SHIDX(0)];
	uiVal = ((uiVal>>29)|(uiVal<<3));
	uiDiv = hData3[SHIDX(1)];
	uiDiv = ((uiDiv>>29)|(uiDiv<<3));

        printf("After IP: \n");
        printf("Data0: %08x\n", uiVal);
        printf("Data1: %08x\n", uiDiv);

	for(ii = 0; ii < 16; ii+=2) {
            uiDiv = D_ENCRYPT(uiDiv, uiVal^hData[SHIDX(ii)], uiVal^hData2[SHIDX(ii)]);
            uiVal = D_ENCRYPT(uiVal, uiDiv^hData[SHIDX(ii+1)], uiDiv^hData2[SHIDX(ii+1)]);
	    printf("After Round %d: \n", ii);
            printf("Data0: %08x\n", uiDiv);
            printf("Data1: %08x\n", uiVal);
        }

	hData3[SHIDX(0)] = ((uiVal>>3)|(uiVal<<29));
	hData3[SHIDX(1)] = ((uiDiv>>3)|(uiDiv<<29));
	FP(SHIDX(0), SHIDX(1), SHIDX(2), hData3);

	hData[SHIDX(0)] = hData3[SHIDX(1)];
	hData[SHIDX(1)] = hData3[SHIDX(0)];


        printf("Final: %08x %08x\n", hData[0], hData[1]);




    printf("========== My attempts =========\n");


    memset(password, 0, 7);
    memset(hash, 0, 8);

    strcpy((char *)data, "KGS!@#$%");

    printData("Data", data, 8);

    password[0] = 'A';
    password[1] = 'B';
    password[2] = 'C';
    password[3] = 'D';
    password[4] = 'E';
    password[5] = 'F';
    password[6] = 'G';

    b0 = *(uint32_t *)&password[0];
    b1 = *(uint32_t *)&password[4] & 0x00ffffff;
    printf("b0: %08x  b1: %08x\n", b0, b1);

    key0 = ((password[(2)] << 5) | (password[(3)] >> 3))&0xff;
    key0 = (key0<<8) | (((password[(1)] << 6) | (password[(2)] >> 2))&0xff);
    key0 = (key0<<8) | (((password[(0)] << 7) | (password[(1)] >> 1))&0xff);
    key0 = (key0<<8) | password[(0)];

    key1 = (password[(6)] << 1)&0xff;
    key1 = (key1<<8) | (((password[(5)] << 2) | (password[(6)] >> 6))&0xff);
    key1 = (key1<<8) | (((password[(4)] << 3) | (password[(5)] >> 5))&0xff);
    key1 = (key1<<8) | (((password[(3)] << 4) | (password[(4)] >> 4))&0xff);
    printf("Initial key: %08x %08x\n", key0, key1);

    key0 = ((((b0 >> 16) & 0xff) << 5) | (((b0 >> 24) & 0xff) >> 3))&0xff;
    key0 = (key0<<8) | (((((b0 >> 8) & 0xff) << 6) | (((b0 >> 16) & 0xff) >> 2))&0xff);
    key0 = (key0<<8) | ((((b0 & 0xff) << 7) | (((b0 >> 8) & 0xff) >> 1))&0xff);
    key0 = (key0<<8) | (b0 & 0xff);

    key1 = (((b1 >> 16) & 0xff) << 1)&0xff;
    key1 = (key1<<8) | (((((b1 >> 8) & 0xff) << 2) | (((b1 >> 16) & 0xff) >> 6))&0xff);
    key1 = (key1<<8) | ((((b1 & 0xff) << 3) | (((b1 >> 8) & 0xff) >> 5))&0xff);
    key1 = (key1<<8) | (((((b0 >> 24) & 0xff) << 4) | ((b1 & 0xff) >> 4))&0xff);

    printf("b0 key0:     %08x %08x\n", key0, key1);

    New_PERM_OP(key1, key0, keytemp, 4, 0x0f0f0f0f);
    New_HPERM_OP(key0, keytemp, -2, 0xcccc0000);
    New_HPERM_OP(key1, keytemp, -2, 0xcccc0000);
    New_PERM_OP(key1, key0, keytemp, 1, 0x55555555);
    New_PERM_OP(key0, key1, keytemp, 8, 0x00ff00ff);
    New_PERM_OP(key1, key0, keytemp, 1, 0x55555555);

    printf("IP key: %08x %08x\n", key0, key1);


    key1 = ((key1&0x000000ff)<<16) | (key1&0x0000ff00) | ((key1&0x00ff0000)>>16) | ((key0&0xf0000000)>>4);
    key0 &= 0x0fffffff;

    printf("Trimmed key: %08x %08x\n", key0, key1);




    data0 = *(uint32_t *)&data[0];
    data1 = *(uint32_t *)&data[4];

    printf("Secret Data0: %08x\n", data0);
    printf("Secret Data1: %08x\n", data1);

    New_IP(data0, data1, datatemp);
    data0 = ((data0>>29)|(data0<<3));
    data1 = ((data1>>29)|(data1<<3));

    printf("After IP: \n");
    printf("Data0: %08x\n", data0);
    printf("Data1: %08x\n", data1);
    

    for(ii = 0; ii < 16; ii+=1) {
        if(shifts2[ii]) {
                key0 = ((key0>>2)|(key0<<26));
                key1 =((key1>>2)|(key1<<26));
        } else {
                key0 = ((key0>>1)|(key0<<27));
                key1 = ((key1>>1)|(key1<<27));
        }
        key0 &= 0x0fffffff;
        key1 &= 0x0fffffff;

        rs = des_skb[0][key0&0x3f] |
                des_skb[1][((key0>>6)&0x03)|((key0>>7)&0x3c)] |
                des_skb[2][((key0>>13)&0x0f)|((key0>>14)&0x30)] |
                des_skb[3][((key0>>20)&0x01)|((key0>>21)&0x06) |
                ((key0>>22)&0x38)];
        rt = des_skb[4][key1&0x3f] |
                des_skb[5][((key1>>7)&0x03)|((key1>>8)&0x3c)] |
                des_skb[6][(key1>>15)&0x3f] |
                des_skb[7][((key1>>21)&0x0f)|((key1>>22)&0x30)];

        /* table contained 0213 4657 */
        roundkey0 = (rt<<16)|(rs&0x0000ffff);
        roundkey0 = (roundkey0>>30)|(roundkey0<<2);
        printf("roundkey0: %08x\n", roundkey0);
        roundkey1 = (rs>>16)|(rt&0xffff0000);
        roundkey1 = (roundkey1>>26)|(roundkey1<<6);
        printf("roundkey1: %08x\n", roundkey1);



        data1 = D_ENCRYPT(data1, data0^roundkey0, data0^roundkey1);
        printf("After Round %d: \n", ii);
        printf("Data0: %08x\n", data0);
        printf("Data1: %08x\n", data1);
        datatemp = data0;
        data0 = data1;
        data1 = datatemp;
    }

    data0 = ((data0>>3)|(data0<<29));
    data1 = ((data1>>3)|(data1<<29));
    New_FP(data0, data1, datatemp);

    printf("Final: %08x %08x\n", data1, data0);
    printf("Final: %08x %08x\n", hData[0], hData[1]);
    
    uint32_t hash0, hash1;
    cudaLM(b0, b1, hash0, hash1);
    printf("Final: %08x %08x\n", hash0, hash1);

}