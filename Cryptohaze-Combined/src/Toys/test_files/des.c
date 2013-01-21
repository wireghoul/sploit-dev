/*****************************************************************************
 * des.c                                                                     *
 *                 Software Model of ASIC DES Implementation                 *
 *                                                                           *
 *   Written 1995-8 by Cryptography Research (http://www.cryptography.com)   *
 *   Original version by Paul Kocher. Placed in the public domain in 1998.   *
 *  THIS IS UNSUPPORTED FREE SOFTWARE. USE AND DISTRIBUTE AT YOUR OWN RISK.  *
 *                                                                           *
 *  IMPORTANT: U.S. LAW MAY REGULATE THE USE AND/OR EXPORT OF THIS PROGRAM.  *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   IMPLEMENTATION NOTES:                                                   *
 *                                                                           *
 *   This DES implementation adheres to the FIPS PUB 46 spec and produces    *
 *   standard output.  The internal operation of the algorithm is slightly   *
 *   different from FIPS 46.  For example, bit orderings are reversed        *
 *   (the right-hand bit is now labelled as bit 0), the S tables have        *
 *   rearranged to simplify implementation, and several permutations have    *
 *   been inverted.  For simplicity and to assist with testing of hardware   *
 *   implementations, code size and performance optimizations are omitted.   *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   REVISION HISTORY:                                                       *
 *                                                                           *
 *   Version 1.0:  Initial release  -- PCK.                                  *
 *   Version 1.1:  Altered DecryptDES exchanges to match EncryptDES. -- PCK  *
 *   Version 1.2:  Minor edits and beautifications.  -- PCK                  *
 *   Version 1.3:  Changes and edits for EFF DES Cracker project.            *
 *                                                                           *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "des.h"

static void ComputeRoundKey(bool roundKey[56], bool key[56]);
static void RotateRoundKeyLeft(bool roundKey[56]);
static void RotateRoundKeyRight(bool roundKey[56]);
static void ComputeIP(bool L[32], bool R[32], bool inBlk[64]);
static void ComputeFP(bool outBlk[64], bool L[32], bool R[32]);
static void ComputeF(bool fout[32], bool R[32], bool roundKey[56]);
static void ComputeP(bool output[32], bool input[32]);
static void ComputeS_Lookup(int k, bool output[4], bool input[6]);
static void ComputePC2(bool subkey[48], bool roundKey[56]);
static void ComputeExpansionE(bool expandedBlock[48], bool R[32]);
static void DumpBin(char *str, bool *b, int bits);
static void Exchange_L_and_R(bool L[32], bool R[32]);

static int EnableDumpBin = 0;



/**********************************************************************/
/*                                                                    */
/*                            DES TABLES                              */
/*                                                                    */
/**********************************************************************/


/*
 *  IP: Output bit table_DES_IP[i] equals input bit i.
 */
static int table_DES_IP[64] = {
    39,  7, 47, 15, 55, 23, 63, 31,
    38,  6, 46, 14, 54, 22, 62, 30,
    37,  5, 45, 13, 53, 21, 61, 29,
    36,  4, 44, 12, 52, 20, 60, 28,
    35,  3, 43, 11, 51, 19, 59, 27,
    34,  2, 42, 10, 50, 18, 58, 26,
    33,  1, 41,  9, 49, 17, 57, 25,
    32,  0, 40,  8, 48, 16, 56, 24
};


/*
 *  FP: Output bit table_DES_FP[i] equals input bit i.
 */
static int table_DES_FP[64] = {
    57, 49, 41, 33, 25, 17,  9,  1,
    59, 51, 43, 35, 27, 19, 11,  3,
    61, 53, 45, 37, 29, 21, 13,  5,
    63, 55, 47, 39, 31, 23, 15,  7,
    56, 48, 40, 32, 24, 16,  8,  0,
    58, 50, 42, 34, 26, 18, 10,  2,
    60, 52, 44, 36, 28, 20, 12,  4,
    62, 54, 46, 38, 30, 22, 14,  6
};


/*
 *  PC1: Permutation choice 1, used to pre-process the key
 */
static int table_DES_PC1[56] = {
    27, 19, 11, 31, 39, 47, 55,
    26, 18, 10, 30, 38, 46, 54,
    25, 17,  9, 29, 37, 45, 53,
    24, 16,  8, 28, 36, 44, 52,
    23, 15,  7,  3, 35, 43, 51,
    22, 14,  6,  2, 34, 42, 50,
    21, 13,  5,  1, 33, 41, 49,
    20, 12,  4,  0, 32, 40, 48
};


/*
 *  PC2: Map 56-bit round key to a 48-bit subkey
 */
static int table_DES_PC2[48] = {
    24, 27, 20,  6, 14, 10,  3, 22,
     0, 17,  7, 12,  8, 23, 11,  5,
    16, 26,  1,  9, 19, 25,  4, 15,
    54, 43, 36, 29, 49, 40, 48, 30,
    52, 44, 37, 33, 46, 35, 50, 41,
    28, 53, 51, 55, 32, 45, 39, 42
};


/*
 *  E: Expand 32-bit R to 48 bits.
 */
static int table_DES_E[48] = {
    31,  0,  1,  2,  3,  4,  3,  4,
     5,  6,  7,  8,  7,  8,  9, 10,
    11, 12, 11, 12, 13, 14, 15, 16,
    15, 16, 17, 18, 19, 20, 19, 20,
    21, 22, 23, 24, 23, 24, 25, 26,
    27, 28, 27, 28, 29, 30, 31,  0
};


/*
 *  P: Permutation of S table outputs
 */
static int table_DES_P[32] = {
    11, 17,  5, 27, 25, 10, 20,  0,
    13, 21,  3, 28, 29,  7, 18, 24,
    31, 22, 12,  6, 26,  2, 16,  8,
    14, 30,  4, 19,  1,  9, 15, 23
};


/*
 *  S Tables: Introduce nonlinearity and avalanche
 */
static int table_DES_S[8][64] = {
    /* table S[0] */
        {   13,  1,  2, 15,  8, 13,  4,  8,  6, 10, 15,  3, 11,  7,  1,  4,
            10, 12,  9,  5,  3,  6, 14, 11,  5,  0,  0, 14, 12,  9,  7,  2,
             7,  2, 11,  1,  4, 14,  1,  7,  9,  4, 12, 10, 14,  8,  2, 13,
             0, 15,  6, 12, 10,  9, 13,  0, 15,  3,  3,  5,  5,  6,  8, 11  },
    /* table S[1] */
        {    4, 13, 11,  0,  2, 11, 14,  7, 15,  4,  0,  9,  8,  1, 13, 10,
             3, 14, 12,  3,  9,  5,  7, 12,  5,  2, 10, 15,  6,  8,  1,  6,
             1,  6,  4, 11, 11, 13, 13,  8, 12,  1,  3,  4,  7, 10, 14,  7,
            10,  9, 15,  5,  6,  0,  8, 15,  0, 14,  5,  2,  9,  3,  2, 12  },
    /* table S[2] */
        {   12, 10,  1, 15, 10,  4, 15,  2,  9,  7,  2, 12,  6,  9,  8,  5,
             0,  6, 13,  1,  3, 13,  4, 14, 14,  0,  7, 11,  5,  3, 11,  8,
             9,  4, 14,  3, 15,  2,  5, 12,  2,  9,  8,  5, 12, 15,  3, 10,
             7, 11,  0, 14,  4,  1, 10,  7,  1,  6, 13,  0, 11,  8,  6, 13  },
    /* table S[3] */
        {    2, 14, 12, 11,  4,  2,  1, 12,  7,  4, 10,  7, 11, 13,  6,  1,
             8,  5,  5,  0,  3, 15, 15, 10, 13,  3,  0,  9, 14,  8,  9,  6,
             4, 11,  2,  8,  1, 12, 11,  7, 10,  1, 13, 14,  7,  2,  8, 13,
            15,  6,  9, 15, 12,  0,  5,  9,  6, 10,  3,  4,  0,  5, 14,  3  },
    /* table S[4] */
        {    7, 13, 13,  8, 14, 11,  3,  5,  0,  6,  6, 15,  9,  0, 10,  3,
             1,  4,  2,  7,  8,  2,  5, 12, 11,  1, 12, 10,  4, 14, 15,  9,
            10,  3,  6, 15,  9,  0,  0,  6, 12, 10, 11,  1,  7, 13, 13,  8,
            15,  9,  1,  4,  3,  5, 14, 11,  5, 12,  2,  7,  8,  2,  4, 14  },
    /* table S[5] */
        {   10, 13,  0,  7,  9,  0, 14,  9,  6,  3,  3,  4, 15,  6,  5, 10,
             1,  2, 13,  8, 12,  5,  7, 14, 11, 12,  4, 11,  2, 15,  8,  1,
            13,  1,  6, 10,  4, 13,  9,  0,  8,  6, 15,  9,  3,  8,  0,  7,
            11,  4,  1, 15,  2, 14, 12,  3,  5, 11, 10,  5, 14,  2,  7, 12  },
    /* table S[6] */
        {   15,  3,  1, 13,  8,  4, 14,  7,  6, 15, 11,  2,  3,  8,  4, 14,
             9, 12,  7,  0,  2,  1, 13, 10, 12,  6,  0,  9,  5, 11, 10,  5,
             0, 13, 14,  8,  7, 10, 11,  1, 10,  3,  4, 15, 13,  4,  1,  2,
             5, 11,  8,  6, 12,  7,  6, 12,  9,  0,  3,  5,  2, 14, 15,  9  },
    /* table S[7] */
        {   14,  0,  4, 15, 13,  7,  1,  4,  2, 14, 15,  2, 11, 13,  8,  1,
             3, 10, 10,  6,  6, 12, 12, 11,  5,  9,  9,  5,  0,  3,  7,  8,
             4, 15,  1, 12, 14,  8,  8,  2, 13,  4,  6,  9,  2,  1, 11,  7,
            15,  5, 12, 11,  9,  3,  7, 14,  3, 10, 10,  0,  5,  6,  0, 13  }
};




/**********************************************************************/
/*                                                                    */
/*                             DES CODE                               */
/*                                                                    */
/**********************************************************************/


/*
 *  EncryptDES: Encrypt a block using DES. Set verbose for debugging info.
 *  (This loop does both loops on the "DES Encryption" page of the flowchart.)
 */
void EncryptDES(bool key[56], bool outBlk[64], bool inBlk[64], int verbose) {
  int i,round;
  bool R[32], L[32], fout[32];
  bool roundKey[56];

  EnableDumpBin = verbose;                      /* set debugging on/off flag */
  DumpBin("input(left)", inBlk+32, 32);
  DumpBin("input(right)", inBlk, 32);
  DumpBin("raw key(left )", key+28, 28);
  DumpBin("raw key(right)", key, 28);

  /* Compute the first roundkey by performing PC1 */
  ComputeRoundKey(roundKey, key);

  DumpBin("roundKey(L)", roundKey+28, 28);
  DumpBin("roundKey(R)", roundKey, 28);

  /* Compute the initial permutation and divide the result into L and R */
  ComputeIP(L,R,inBlk);

  DumpBin("after IP(L)", L, 32);
  DumpBin("after IP(R)", R, 32);

  for (round = 0; round < 16; round++) {
    if (verbose)
      printf("-------------- BEGIN ENCRYPT ROUND %d -------------\n", round);
    DumpBin("round start(L)", L, 32);
    DumpBin("round start(R)", R, 32);

    /* Rotate roundKey halves left once or twice (depending on round) */
    RotateRoundKeyLeft(roundKey);
    if (round != 0 && round != 1 && round != 8 && round != 15)
      RotateRoundKeyLeft(roundKey);
    DumpBin("roundKey(L)", roundKey+28, 28);
    DumpBin("roundKey(R)", roundKey, 28);

    /* Compute f(R, roundKey) and exclusive-OR onto the value in L */
    ComputeF(fout, R, roundKey);
    DumpBin("f(R,key)", fout, 32);
    for (i = 0; i < 32; i++)
      L[i] ^= fout[i];
    DumpBin("L^f(R,key)", L, 32);

    Exchange_L_and_R(L,R);

    DumpBin("round end(L)", L, 32);
    DumpBin("round end(R)", R, 32);
    if (verbose)
      printf("--------------- END ROUND %d --------------\n", round);
  }

  Exchange_L_and_R(L,R);

  /* Combine L and R then compute the final permutation */
  ComputeFP(outBlk,L,R);
  DumpBin("FP out( left)", outBlk+32, 32);
  DumpBin("FP out(right)", outBlk, 32);
}



/*
 *  DecryptDES: Decrypt a block using DES. Set verbose for debugging info.
 *  (This loop does both loops on the "DES Decryption" page of the flowchart.)
 */
void DecryptDES(bool key[56], bool outBlk[64], bool inBlk[64], int verbose) {
  int i,round;
  bool R[32], L[32], fout[32];
  bool roundKey[56];

  EnableDumpBin = verbose;                      /* set debugging on/off flag */
  DumpBin("input(left)", inBlk+32, 32);
  DumpBin("input(right)", inBlk, 32);
  DumpBin("raw key(left )", key+28, 28);
  DumpBin("raw key(right)", key, 28);

  /* Compute the first roundkey by performing PC1 */
  ComputeRoundKey(roundKey, key);

  DumpBin("roundKey(L)", roundKey+28, 28);
  DumpBin("roundKey(R)", roundKey, 28);

  /* Compute the initial permutation and divide the result into L and R */
  ComputeIP(L,R,inBlk);

  DumpBin("after IP(L)", L, 32);
  DumpBin("after IP(R)", R, 32);

  for (round = 0; round < 16; round++) {
    if (verbose)
      printf("-------------- BEGIN DECRYPT ROUND %d -------------\n", round);
    DumpBin("round start(L)", L, 32);
    DumpBin("round start(R)", R, 32);

    /* Compute f(R, roundKey) and exclusive-OR onto the value in L */
    ComputeF(fout, R, roundKey);
    DumpBin("f(R,key)", fout, 32);
    for (i = 0; i < 32; i++)
      L[i] ^= fout[i];
    DumpBin("L^f(R,key)", L, 32);

    Exchange_L_and_R(L,R);

    /* Rotate roundKey halves right once or twice (depending on round) */
    DumpBin("roundKey(L)", roundKey+28, 28);       /* show keys before shift */
    DumpBin("roundKey(R)", roundKey, 28);
    RotateRoundKeyRight(roundKey);
    if (round != 0 && round != 7 && round != 14 && round != 15)
      RotateRoundKeyRight(roundKey);

    DumpBin("round end(L)", L, 32);
    DumpBin("round end(R)", R, 32);
    if (verbose)
      printf("--------------- END ROUND %d --------------\n", round);
  }

  Exchange_L_and_R(L,R);

  /* Combine L and R then compute the final permutation */
  ComputeFP(outBlk,L,R);
  DumpBin("FP out( left)", outBlk+32, 32);
  DumpBin("FP out(right)", outBlk, 32);
}



/*
 *  ComputeRoundKey: Compute PC1 on the key and store the result in roundKey
 */
static void ComputeRoundKey(bool roundKey[56], bool key[56]) {
  int i;

  for (i = 0; i < 56; i++)
    roundKey[table_DES_PC1[i]] = key[i];
}



/*
 *  RotateRoundKeyLeft: Rotate each of the halves of roundKey left one bit
 */
static void RotateRoundKeyLeft(bool roundKey[56]) {
  bool temp1, temp2;
  int i;

  temp1 = roundKey[27];
  temp2 = roundKey[55];
  for (i = 27; i >= 1; i--) {
    roundKey[i] = roundKey[i-1];
    roundKey[i+28] = roundKey[i+28-1];
  }
  roundKey[ 0] = temp1;
  roundKey[28] = temp2;
}



/*
 *  RotateRoundKeyRight: Rotate each of the halves of roundKey right one bit
 */
static void RotateRoundKeyRight(bool roundKey[56]) {
  bool temp1, temp2;
  int i;

  temp1 = roundKey[0];
  temp2 = roundKey[28];
  for (i = 0; i < 27; i++) {
    roundKey[i] = roundKey[i+1];
    roundKey[i+28] = roundKey[i+28+1];
  }
  roundKey[27] = temp1;
  roundKey[55] = temp2;
}



/*
 *  ComputeIP: Compute the initial permutation and split into L and R halves.
 */
static void ComputeIP(bool L[32], bool R[32], bool inBlk[64]) {
  bool output[64];
  int i;

  /* Permute
   */
  for (i = 63; i >= 0; i--)
    output[table_DES_IP[i]] = inBlk[i];

  /* Split into R and L.  Bits 63..32 go in L, bits 31..0 go in R.
   */
  for (i = 63; i >= 0; i--) {
    if (i >= 32)
      L[i-32] = output[i];
    else
      R[i] = output[i];
  }
}



/*
 *  ComputeFP: Combine the L and R halves and do the final permutation.
 */
static void ComputeFP(bool outBlk[64], bool L[32], bool R[32]) {
  bool input[64];
  int i;

  /* Combine L and R into input[64]
   */
  for (i = 63; i >= 0; i--)
    input[i] = (i >= 32) ? L[i - 32] : R[i];

  /* Permute
   */
  for (i = 63; i >= 0; i--)
    outBlk[table_DES_FP[i]] = input[i];
}



/*
 *  ComputeF: Compute the DES f function and store the result in fout.
 */
static void ComputeF(bool fout[32], bool R[32], bool roundKey[56]) {
  bool expandedBlock[48], subkey[48], sout[32];
  int i,k;

  /* Expand R into 48 bits using the E expansion */
  ComputeExpansionE(expandedBlock, R);
  DumpBin("expanded E", expandedBlock, 48);

  /* Convert the roundKey into the subkey using PC2 */
  ComputePC2(subkey, roundKey);
  DumpBin("subkey", subkey, 48);

  /* XOR the subkey onto the expanded block */
  for (i = 0; i < 48; i++)
    expandedBlock[i] ^= subkey[i];

  /* Divide expandedBlock into 6-bit chunks and do S table lookups */
  for (k = 0; k < 8; k++)
    ComputeS_Lookup(k, sout+4*k, expandedBlock+6*k);

  /* To complete the f() calculation, do permutation P on the S table output */
  ComputeP(fout, sout);
}



/*
 *  ComputeP: Compute the P permutation on the S table outputs.
 */
static void ComputeP(bool output[32], bool input[32]) {
  int i;

  for (i = 0; i < 32; i++)
    output[table_DES_P[i]] = input[i];
}



/*
 *  Look up a 6-bit input in S table k and store the result as a 4-bit output.
 */
static void ComputeS_Lookup(int k, bool output[4], bool input[6]) {
  int inputValue, outputValue;

  /* Convert the input bits into an integer */
  inputValue = input[0] + 2*input[1] + 4*input[2] + 8*input[3] +
          16*input[4] + 32*input[5];

  /* Do the S table lookup */
  outputValue = table_DES_S[k][inputValue];

  /* Convert the result into binary form */
  output[0] = (outputValue & 1) ? 1 : 0;
  output[1] = (outputValue & 2) ? 1 : 0;
  output[2] = (outputValue & 4) ? 1 : 0;
  output[3] = (outputValue & 8) ? 1 : 0;
}



/*
 *  ComputePC2: Map a 56-bit round key onto a 48-bit subkey
 */
static void ComputePC2(bool subkey[48], bool roundKey[56]) {
  int i;

  for (i = 0; i < 48; i++)
    subkey[i] = roundKey[table_DES_PC2[i]];
}



/*
 *  ComputeExpansionE: Compute the E expansion to prepare to use S tables.
 */
static void ComputeExpansionE(bool expandedBlock[48], bool R[32]) {
  int i;

  for (i = 0; i < 48; i++)
    expandedBlock[i] = R[table_DES_E[i]];
}



/*
 *  Exchange_L_and_R:  Swap L and R
 */
static void Exchange_L_and_R(bool L[32], bool R[32]) {
  int i;

  for (i = 0; i < 32; i++)
    L[i] ^= R[i] ^= L[i] ^= R[i];                 /* exchanges L[i] and R[i] */
}



/*
 *  DumpBin: Display intermediate values if emableDumpBin is set.
 */
static void DumpBin(char *str, bool *b, int bits) {
  int i;

  if ((bits % 4)!=0 || bits>48) {
    printf("Bad call to DumpBin (bits > 48 or bit len not a multiple of 4\n");
    exit(1);
  }

  if (EnableDumpBin) {
    for (i = strlen(str); i < 14; i++)
      printf(" ");
    printf("%s: ", str);
    for (i = bits-1; i >= 0; i--)
      printf("%d", b[i]);
    printf(" ");
    for (i = bits; i < 48; i++)
      printf(" ");
    printf("(");
    for (i = bits-4; i >= 0; i-=4)
      printf("%X", b[i]+2*b[i+1]+4*b[i+2]+8*b[i+3]);
    printf(")\n");
  }
}

