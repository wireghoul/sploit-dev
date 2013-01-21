
#define MAX_PASSWORD_LENGTH 32


void searchForHashNTLM(uint a, uint b, uint c, uint d, uint vectorPosition,
        __local unsigned char *sharedBitmap,
        uint deviceNumberOfHashes, __global unsigned int *deviceHashArray,
        __global   unsigned char *foundPasswordArray, __global  unsigned char *deviceSuccessArray,
        uint b0, uint b1, uint b2, uint b3, uint b4, uint b5, uint b6);


inline void searchForHashNTLM(uint a, uint b, uint c, uint d, uint vectorPosition,
        __local unsigned char *sharedBitmap,
        uint deviceNumberOfHashes, __global unsigned int *deviceHashArray,
        __global   unsigned char *foundPasswordArray, __global  unsigned char *deviceSuccessArray,
        uint b0, uint b1, uint b2, uint b3, uint b4, uint b5, uint b6) {

    uint search_high, search_low, search_index, temp, hash_order_mem, hash_order_a;

#if CPU_DEBUG && 0
    printf("Searching: %c%c%c%c%c%c\n", b0 & 0xff, (b0 >> 8 ) & 0xff,
            (b0 >> 16) & 0xff, (b0 >> 24) & 0xff, b1 & 0xff, (b1 >> 8 ) & 0xff);
#endif

    if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) {
        //printf("Shared bitmap hit!\n");
        //printf("%08x %08x %08x %08x\n", a, b, c, d);

            {
                search_high = deviceNumberOfHashes;
                search_low = 0;
                search_index = 0;
                while (search_low < search_high) {
                    search_index = search_low + (search_high - search_low) / 2;
                    temp = deviceHashArray[4 * search_index];
                    hash_order_mem = (temp & 0xff) << 24 | ((temp >> 8) & 0xff) << 16 | ((temp >> 16) & 0xff) << 8 | ((temp >> 24) & 0xff);
                    hash_order_a = (a & 0xff) << 24 | ((a >> 8) & 0xff) << 16 | ((a >> 16) & 0xff) << 8 | ((a >> 24) & 0xff);
                    if (hash_order_mem < hash_order_a) {
                        search_low = search_index + 1;
                    } else {
                        search_high = search_index;
                    }
                    if ((hash_order_a == hash_order_mem) && (search_low < deviceNumberOfHashes)) {
                        break;
                    }
                }
                if (hash_order_a != hash_order_mem) {
                    goto next;
                }
                while (search_index && (a == deviceHashArray[(search_index - 1) * 4])) {
                    search_index--;
                }
                while ((a == deviceHashArray[search_index * 4])) {
                    {
                        if (b == deviceHashArray[search_index * 4 + 1]) {
                            if (c == deviceHashArray[search_index * 4 + 2]) {
                                if (d == deviceHashArray[search_index * 4 + 3]) {
                                    #if CPU_DEBUG
                                    printf("HASH FOUND!\n");
                                    printf("%08x %08x\n", b0, b1);
                                    #endif
                                    if (PASSWORD_LENGTH >= 1) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 0] = (b0 >> 0) & 0xff;
                                    if (PASSWORD_LENGTH >= 2) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 1] = (b0 >> 16) & 0xff;
                                    if (PASSWORD_LENGTH >= 3) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 2] = (b1 >> 0) & 0xff;
                                    if (PASSWORD_LENGTH >= 4) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 3] = (b1 >> 16) & 0xff;
                                    if (PASSWORD_LENGTH >= 5) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 4] = (b2 >> 0) & 0xff;
                                    if (PASSWORD_LENGTH >= 6) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 5] = (b2 >> 16) & 0xff;
                                    if (PASSWORD_LENGTH >= 7) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 6] = (b3 >> 0) & 0xff;
                                    if (PASSWORD_LENGTH >= 8) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 7] = (b3 >> 16) & 0xff;
                                    if (PASSWORD_LENGTH >= 9) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 8] = (b4 >> 0) & 0xff;
                                    if (PASSWORD_LENGTH >= 10) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 9] = (b4 >> 16) & 0xff;
                                    if (PASSWORD_LENGTH >= 11) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 10] = (b5 >> 0) & 0xff;
                                    if (PASSWORD_LENGTH >= 12) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 11] = (b5 >> 16) & 0xff;
                                    if (PASSWORD_LENGTH >= 13) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 12] = (b6 >> 0) & 0xff;
                                    if (PASSWORD_LENGTH >= 14) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 13] = (b6 >> 16) & 0xff;
                                    deviceSuccessArray[search_index] = (unsigned char) 1;

                                }
                            }
                        }
                    }
                    search_index++;
                }
            }
        }
next:
;
}


__kernel __attribute__((vec_type_hint(vector_type))) void RegenerateChainsNTLMAMD(
    __constant unsigned char *deviceCharset,/* 0 */
    __private unsigned int deviceCharsetLength,/* 1 */
    __private unsigned int deviceChainLength,/* 2 */
    __private unsigned int deviceTableIndex,/* 3 */
    __private unsigned int deviceNumberOfThreads,/* 4 */
    __private unsigned int deviceNumberOfChainsToRegen, /* 5 */
    __private unsigned int deviceNumberOfHashes, /* 6 */
    __constant  unsigned char *constantBitmap, /* 7 */
    __global   unsigned int *initialPasswordArray, /* 8 */
    __global   unsigned char *foundPasswordArray, /* 9 */
    __global   unsigned int *deviceHashArray, /* 10 */
    __global  unsigned char *deviceSuccessArray, /* 11 */
    __private unsigned int devicePasswordSpaceOffset, /* 12 */
    __private unsigned int deviceStartChainIndex, /* 13 */
    __private unsigned int deviceStepsToRun /* 14 */
) {

    // Start the kernel.
    __local unsigned char charset[512];
    __local unsigned char sharedBitmap[8192];

#if CPU_DEBUG
    printf("\n\n\n");
    printf("Kernel start, global id %d\n", get_global_id(0));
    printf("deviceCharsetLength: %d\n", deviceCharsetLength);
    printf("deviceCharset: %08x ...\n", deviceCharset);
    printf("deviceCharset: %c %c %c %c ...\n", deviceCharset[0], deviceCharset[1], deviceCharset[2], deviceCharset[3]);
    printf("deviceChainLength: %d\n", deviceChainLength);
    printf("deviceTableIndex: %d\n", deviceTableIndex);
    printf("deviceNumberOfThreads: %d\n", deviceNumberOfThreads);
    printf("deviceNumberOfChainsToRegen: %d\n", deviceNumberOfChainsToRegen);
    printf("deviceNumberOfHashes: %d\n", deviceNumberOfHashes);
    printf("devicePasswordSpaceOffset: %d\n", devicePasswordSpaceOffset);
    printf("deviceStartChainIndex: %d\n", deviceStartChainIndex);
    printf("deviceStepsToRun: %d\n", deviceStepsToRun);
    printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
#endif


    // Needed variables for generation
    uint CurrentStep, PassCount, password_index, charsetOffset;

    // Hash variables
    vector_type b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
    vector_type a,b,c,d;
    vector_type bitmap_index;

    // Generic "copy charset to shared memory" function
    copySingleCharsetToShared(charset, deviceCharset);
    //printf("Charset copied... %c %c %c %c ...\n", charset[0], charset[1], charset[2], charset[3]);

#if CPU_DEBUG
    for (int i = 0; i < 512; i++) {
        printf("Charset %d: %02x %c\n", i, charset[i], charset[i]);
    }
#endif
    
    copySingleBitmapToShared(sharedBitmap, constantBitmap);
    // Figure out which password we are working on.
    password_index = (get_global_id(0) + (devicePasswordSpaceOffset * deviceNumberOfThreads));
#if CPU_DEBUG
    printf("password index: %d\n", password_index);
    printf("startChainIndex: %d\n", deviceStartChainIndex);
#endif
    // Return if this thread is working on something beyond the end of the password space
    if (password_index >= (deviceNumberOfChainsToRegen / VECTOR_WIDTH)) {
#if CPU_DEBUG && 0
        printf("Returning: pass_index > deviceNumberOfChains / 4\n");
#endif
        return;
    }

    b0 = 0x00000000;
    b1 = 0x00000000;
    b2 = 0x00000000;
    b3 = 0x00000000;
    b4 = 0x00000000;
    b5 = 0x00000000;
    b6 = 0x00000000;
    b7 = 0x00000000;
    b8 = 0x00000000;
    b9 = 0x00000000;
    b10 = 0x00000000;
    b11 = 0x00000000;
    b12 = 0x00000000;
    b13 = 0x00000000;
    b14 = 0x00000000;
    b15 = 0x00000000;

    // Load b0/b1 out of memory
    b15 = vload_type(0 * deviceNumberOfChainsToRegen + password_index, initialPasswordArray);
    b0 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
#if PASSWORD_LENGTH > 2
    b1 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
#endif
#if PASSWORD_LENGTH > 4
    b15 = vload_type((1 * deviceNumberOfChainsToRegen / VECTOR_WIDTH) + password_index, initialPasswordArray);
    b2 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 6
    b3 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
#endif
#if PASSWORD_LENGTH > 8
    b15 = vload_type((2 * deviceNumberOfChainsToRegen / VECTOR_WIDTH) + password_index, initialPasswordArray);
    b4 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 10
    b5 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
#endif
    b15 = (vector_type)0x00000000;

#if CPU_DEBUG
    printf("Initial password s0: %c%c%c%c%c%c\n", b0.s0 & 0xff, (b0.s0 >> 16 ) & 0xff,
        (b1.s0) & 0xff, (b1.s0 >> 16) & 0xff, b2.s0 & 0xff, (b2.s0 >> 16 ) & 0xff);
    printf("Initial password s1: %c%c%c%c%c%c\n", b0.s1 & 0xff, (b0.s1 >> 16 ) & 0xff,
        (b1.s1) & 0xff, (b1.s1 >> 16) & 0xff, b2.s1 & 0xff, (b2.s1 >> 16 ) & 0xff);
    printf("Initial password s2: %c%c%c%c%c%c\n", b0.s2 & 0xff, (b0.s2 >> 16 ) & 0xff,
        (b1.s2) & 0xff, (b1.s2 >> 16) & 0xff, b2.s2 & 0xff, (b2.s2 >> 16 ) & 0xff);
    printf("Initial password s3: %c%c%c%c%c%c\n", b0.s3 & 0xff, (b0.s3 >> 16 ) & 0xff,
        (b1.s3) & 0xff, (b1.s3 >> 16) & 0xff, b2.s3 & 0xff, (b2.s3 >> 16 ) & 0xff);
#endif
    charsetOffset = deviceStartChainIndex % deviceCharsetLength;

    for (PassCount = 0; PassCount < deviceStepsToRun; PassCount++) {
        CurrentStep = PassCount + deviceStartChainIndex;
#if CPU_DEBUG
        printf("\nChain %d, step %d\n", password_index, PassCount);
#endif
        padMDHash(PASSWORD_LENGTH * 2, &b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        OpenCL_MD4(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, &a, &b, &c, &d);
#if CPU_DEBUG
        printf("\n\n\nNTLM result s0... %08x %08x %08x %08x\n", a.s0, b.s0, c.s0, d.s0);
        printf("NTLM result s1... %08x %08x %08x %08x\n", a.s1, b.s1, c.s1, d.s1);
        printf("NTLM result s2... %08x %08x %08x %08x\n", a.s2, b.s2, c.s2, d.s2);
        printf("NTLM result s3... %08x %08x %08x %08x\n", a.s3, b.s3, c.s3, d.s3);
#endif

        // Do all the searching
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
        searchForHashNTLM(a.s0, b.s0, c.s0, d.s0, 0, sharedBitmap, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                b0.s0, b1.s0, b2.s0, b3.s0, b4.s0, b5.s0, b6.s0);
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
        searchForHashNTLM(a.s1, b.s1, c.s1, d.s1, 1, sharedBitmap, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                b0.s1, b1.s1, b2.s1, b3.s1, b4.s1, b5.s1, b6.s1);
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
        searchForHashNTLM(a.s2, b.s2, c.s2, d.s2, 2, sharedBitmap, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                b0.s2, b1.s2, b2.s2, b3.s2, b4.s2, b5.s2, b6.s2);
        searchForHashNTLM(a.s3, b.s3, c.s3, d.s3, 3, sharedBitmap, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                b0.s3, b1.s3, b2.s3, b3.s3, b4.s3, b5.s3, b6.s3);
#endif
#if grt_vector_8 || grt_vector_16
        searchForHashNTLM(a.s4, b.s4, c.s4, d.s4, 4, sharedBitmap, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                b0.s4, b1.s4, b2.s4, b3.s4, b4.s4, b5.s4, b6.s4);
        searchForHashNTLM(a.s5, b.s5, c.s5, d.s5, 5, sharedBitmap, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                b0.s5, b1.s5, b2.s5, b3.s5, b4.s5, b5.s5, b6.s5);
        searchForHashNTLM(a.s6, b.s6, c.s6, d.s6, 6, sharedBitmap, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                b0.s6, b1.s6, b2.s6, b3.s6, b4.s6, b5.s6, b6.s6);
        searchForHashNTLM(a.s7, b.s7, c.s7, d.s7, 7, sharedBitmap, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                b0.s7, b1.s7, b2.s7, b3.s7, b4.s7, b5.s7, b6.s7);
#endif


        reduceSingleCharsetNTLM(&b0, &b1, &b2, &b3, &b4, &b5, a, b, c, d, CurrentStep, charset, charsetOffset, PASSWORD_LENGTH, deviceTableIndex);
#if CPU_DEBUG
        printf("New password s0: %08x %08x\n", b0.s0, b1.s0);
        printf("New password s1: %08x %08x\n", b0.s1, b1.s1);
        printf("New password s2: %08x %08x\n", b0.s2, b1.s2);
        printf("New password s3: %08x %08x\n", b0.s3, b1.s3);
#endif
#if CPU_DEBUG
    printf("New password s0: %c%c%c%c%c%c\n", b0.s0 & 0xff, (b0.s0 >> 16 ) & 0xff,
        (b1.s0) & 0xff, (b1.s0 >> 16) & 0xff, b2.s0 & 0xff, (b2.s0 >> 16 ) & 0xff);
    printf("New password s1: %c%c%c%c%c%c\n", b0.s1 & 0xff, (b0.s1 >> 16 ) & 0xff,
        (b1.s1) & 0xff, (b1.s1 >> 16) & 0xff, b2.s1 & 0xff, (b2.s1 >> 16 ) & 0xff);
    printf("New password s2: %c%c%c%c%c%c\n", b0.s2 & 0xff, (b0.s2 >> 16 ) & 0xff,
        (b1.s2) & 0xff, (b1.s2 >> 16) & 0xff, b2.s2 & 0xff, (b2.s2 >> 16 ) & 0xff);
    printf("New password s3: %c%c%c%c%c%c\n", b0.s3 & 0xff, (b0.s3 >> 16 ) & 0xff,
        (b1.s3) & 0xff, (b1.s3 >> 16) & 0xff, b2.s3 & 0xff, (b2.s3 >> 16 ) & 0xff);
#endif
        charsetOffset++;
        if (charsetOffset >= deviceCharsetLength) {
            charsetOffset = 0;
        }
#if CPU_DEBUG
        printf("charsetOffset: %d\n", charsetOffset);
#endif

    }
    // Done with the number of steps we need to run

    // If we are done (or have somehow overflowed), store the result
    if (CurrentStep >= (deviceChainLength - 1)) {
#if CPU_DEBUG
        printf("Out of chains, returning.\n");
        printf("final state s0: %08x %08x\n", a.s0, b.s0);
        printf("final state s1: %08x %08x\n", a.s1, b.s1);
        printf("final state s2: %08x %08x\n", a.s2, b.s2);
        printf("final state s3: %08x %08x\n", a.s3, b.s3);
#endif
        return;
    }
    // Else, store the b0/b1 values back to the initial array for the next loop
    else {
#if CPU_DEBUG
        printf("storing state s0: %08x %08x\n", b0.s0, b1.s0);
        printf("storing state s1: %08x %08x\n", b0.s1, b1.s1);
        printf("storing state s2: %08x %08x\n", b0.s2, b1.s2);
        printf("storing state s3: %08x %08x\n", b0.s3, b1.s3);
#endif

        b15 =  (b0 & 0xff) | ((b0 & 0xff0000) >> 8);
#if PASSWORD_LENGTH > 2
        b15 |= (b1 & 0xff) << 16 | ((b1 & 0xff0000) << 8);
#endif
#if CPU_DEBUG
        printf("storing state b15: %08x %08x %08x %08x\n", b15.s0, b15.s1, b15.s2, b15.s3);
#endif
        // Need to store this for all cases.
        vstore_type(b15, 0 * deviceNumberOfChainsToRegen + password_index, initialPasswordArray);

#if PASSWORD_LENGTH > 4
        b15 =  (b2 & 0xff) | ((b2 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 6
        b15 |= (b3 & 0xff) << 16 | ((b3 & 0xff0000) << 8);

    #endif
        vstore_type(b15, (1 * deviceNumberOfChainsToRegen / VECTOR_WIDTH) + password_index, initialPasswordArray);
#endif

#if PASSWORD_LENGTH > 8
        b15 =  (b4 & 0xff) | ((b4 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 10
        b15 |= (b5 & 0xff) << 16 | ((b5 & 0xff0000) << 8);

    #endif
        vstore_type(b15, (2 * deviceNumberOfChainsToRegen / VECTOR_WIDTH) + password_index, initialPasswordArray);
#endif

    }

}