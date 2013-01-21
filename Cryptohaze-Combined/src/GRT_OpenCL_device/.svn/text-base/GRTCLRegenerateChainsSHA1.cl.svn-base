
#if INTEL_OPENCL_HACK
#define LOCAL_MEMORY_TYPE __constant
#define CHARSET_NAME deviceCharset
#define BITMAP_NAME constantBitmap
#else
#define LOCAL_MEMORY_TYPE __local
#define CHARSET_NAME charset
#define BITMAP_NAME sharedBitmap
#endif

#define MAX_PASSWORD_LENGTH 32

// Apple OpenCL compiler quirks
void searchForHashSHA1(uint a, uint b, uint c, uint d, uint vectorPosition,
        LOCAL_MEMORY_TYPE unsigned char *sharedBitmap,
        uint deviceNumberOfHashes, __global unsigned int *deviceHashArray,
        __global   unsigned char *foundPasswordArray, __global  unsigned char *deviceSuccessArray,
        uint b0, uint b1, uint b2, uint b3);


inline void searchForHashSHA1(uint a, uint b, uint c, uint d, uint vectorPosition,
        LOCAL_MEMORY_TYPE unsigned char *sharedBitmap,
        uint deviceNumberOfHashes, __global unsigned int *deviceHashArray,
        __global   unsigned char *foundPasswordArray, __global  unsigned char *deviceSuccessArray,
        uint b0, uint b1, uint b2, uint b3) {

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
                    temp = deviceHashArray[5 * search_index];
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
                while (search_index && (a == deviceHashArray[(search_index - 1) * 5])) {
                    search_index--;
                }
                while ((a == deviceHashArray[search_index * 5])) {
                    {
                        if (b == deviceHashArray[search_index * 5 + 1]) {
                            if (c == deviceHashArray[search_index * 5 + 2]) {
                                if (d == deviceHashArray[search_index * 5 + 3]) {
                                    #if CPU_DEBUG
                                    printf("HASH FOUND!\n");
                                    printf("%08x %08x\n", b0, b1);
                                    #endif
                                    if (PASSWORD_LENGTH >= 1) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 0] = (b0 >> 0) & 0xff;
                                    if (PASSWORD_LENGTH >= 2) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 1] = (b0 >> 8) & 0xff;
                                    if (PASSWORD_LENGTH >= 3) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 2] = (b0 >> 16) & 0xff;
                                    if (PASSWORD_LENGTH >= 4) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 3] = (b0 >> 24) & 0xff;
                                    if (PASSWORD_LENGTH >= 5) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 4] = (b1 >> 0) & 0xff;
                                    if (PASSWORD_LENGTH >= 6) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 5] = (b1 >> 8) & 0xff;
                                    if (PASSWORD_LENGTH >= 7) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 6] = (b1 >> 16) & 0xff;
                                    if (PASSWORD_LENGTH >= 8) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 7] = (b1 >> 24) & 0xff;
                                    if (PASSWORD_LENGTH >= 9) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 8] = (b2 >> 0) & 0xff;
                                    if (PASSWORD_LENGTH >= 10) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 9] = (b2 >> 8) & 0xff;
                                    if (PASSWORD_LENGTH >= 11) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 10] = (b2 >> 16) & 0xff;
                                    if (PASSWORD_LENGTH >= 12) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 11] = (b2 >> 24) & 0xff;
                                    if (PASSWORD_LENGTH >= 13) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 12] = (b3 >> 0) & 0xff;
                                    if (PASSWORD_LENGTH >= 14) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 13] = (b3 >> 8) & 0xff;
                                    if (PASSWORD_LENGTH >= 15) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 14] = (b3 >> 16) & 0xff;
                                    if (PASSWORD_LENGTH >= 16) foundPasswordArray[search_index * MAX_PASSWORD_LENGTH + 15] = (b3 >> 24) & 0xff;
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


__kernel __attribute__((vec_type_hint(vector_type))) void RegenerateChainsSHA1AMD(
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

#ifndef INTEL_OPENCL_HACK
    // Start the kernel.
    __local unsigned char charset[512];
    __local unsigned char sharedBitmap[8192];
#endif
#if CPU_DEBUG && 0
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
    vector_type a,b,c,d,e;
    vector_type passb0, passb1, passb2, passb3;
    vector_type bitmap_index;
#ifndef INTEL_OPENCL_HACK
    // Generic "copy charset to shared memory" function
    copySingleCharsetToShared(charset, deviceCharset);
    //printf("Charset copied... %c %c %c %c ...\n", charset[0], charset[1], charset[2], charset[3]);

    copySingleBitmapToShared(sharedBitmap, constantBitmap);
#endif

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
    b0 = vload_type(0 * deviceNumberOfChainsToRegen + password_index, initialPasswordArray); // lengths 1-4
#if PASSWORD_LENGTH > 4
    b1 = vload_type((1 * deviceNumberOfChainsToRegen / VECTOR_WIDTH) + password_index, initialPasswordArray); // Len 5-8
#endif
#if PASSWORD_LENGTH > 8
    b2 = vload_type((2 * deviceNumberOfChainsToRegen / VECTOR_WIDTH) + password_index, initialPasswordArray);
#endif
#if PASSWORD_LENGTH > 12
    #error "Password lengths > 12 are not supported!"
#endif

#if CPU_DEBUG
    printf("Initial password s0: %c%c%c%c%c%c\n", b0.s0 & 0xff, (b0.s0 >> 8 ) & 0xff,
        (b0.s0 >> 16) & 0xff, (b0.s0 >> 24) & 0xff, b1.s0 & 0xff, (b1.s0 >> 8 ) & 0xff);
    printf("Initial password s0: %c%c%c%c%c%c\n", b0.s1 & 0xff, (b0.s1 >> 8 ) & 0xff,
        (b0.s1 >> 16) & 0xff, (b0.s1 >> 24) & 0xff, b1.s1 & 0xff, (b1.s1 >> 8 ) & 0xff);
    printf("Initial password s0: %c%c%c%c%c%c\n", b0.s2 & 0xff, (b0.s2 >> 8 ) & 0xff,
        (b0.s2 >> 16) & 0xff, (b0.s2 >> 24) & 0xff, b1.s2 & 0xff, (b1.s2 >> 8 ) & 0xff);
    printf("Initial password s0: %c%c%c%c%c%c\n", b0.s3 & 0xff, (b0.s3 >> 8 ) & 0xff,
        (b0.s3 >> 16) & 0xff, (b0.s3 >> 24) & 0xff, b1.s3 & 0xff, (b1.s3 >> 8 ) & 0xff);
#endif
    charsetOffset = deviceStartChainIndex % deviceCharsetLength;

    for (PassCount = 0; PassCount < deviceStepsToRun; PassCount++) {
        CurrentStep = PassCount + deviceStartChainIndex;
#if CPU_DEBUG
        printf("\nChain %d, step %d\n", password_index, PassCount);
#endif
        passb0 = b0; passb1 = b1; passb2 = b2; passb3 = b3;
        padSHAHash(PASSWORD_LENGTH, &b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        a = reverse(a);b = reverse(b);c = reverse(c);d = reverse(d);e = reverse(e);
#if CPU_DEBUG
        printf("\n\n\nMD5 result s0... %08x %08x %08x %08x\n", a.s0, b.s0, c.s0, d.s0);
        printf("MD5 result s1... %08x %08x %08x %08x\n", a.s1, b.s1, c.s1, d.s1);
        printf("MD5 result s2... %08x %08x %08x %08x\n", a.s2, b.s2, c.s2, d.s2);
        printf("MD5 result s3... %08x %08x %08x %08x\n", a.s3, b.s3, c.s3, d.s3);
#endif

        // Do all the searching
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
        searchForHashSHA1(a.s0, b.s0, c.s0, d.s0, 0, BITMAP_NAME, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                passb0.s0, passb1.s0, passb2.s0, passb3.s0);
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
        searchForHashSHA1(a.s1, b.s1, c.s1, d.s1, 1, BITMAP_NAME, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                passb0.s1, passb1.s1, passb2.s1, passb3.s1);
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
        searchForHashSHA1(a.s2, b.s2, c.s2, d.s2, 2, BITMAP_NAME, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                passb0.s2, passb1.s2, passb2.s2, passb3.s2);
        searchForHashSHA1(a.s3, b.s3, c.s3, d.s3, 3, BITMAP_NAME, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                passb0.s3, passb1.s3, passb2.s3, passb3.s3);
#endif
#if grt_vector_8 || grt_vector_16
        searchForHashSHA1(a.s4, b.s4, c.s4, d.s4, 4, BITMAP_NAME, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                passb0.s4, passb1.s4, passb2.s4, passb3.s4);
        searchForHashSHA1(a.s5, b.s5, c.s5, d.s5, 5, BITMAP_NAME, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                passb0.s5, passb1.s5, passb2.s5, passb3.s5);
        searchForHashSHA1(a.s6, b.s6, c.s6, d.s6, 6, BITMAP_NAME, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                passb0.s6, passb1.s6, passb2.s6, passb3.s6);
        searchForHashSHA1(a.s7, b.s7, c.s7, d.s7, 7, BITMAP_NAME, deviceNumberOfHashes, deviceHashArray, foundPasswordArray, deviceSuccessArray,
                passb0.s7, passb1.s7, passb2.s7, passb3.s7);
#endif

        ClearB0ToB15(&b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        reduceSingleCharsetNormal(&b0, &b1, &b2, a, b, c, d, CurrentStep, CHARSET_NAME, charsetOffset, PASSWORD_LENGTH, deviceTableIndex);
#if CPU_DEBUG
        printf("New password s0: %08x %08x\n", b0.s0, b1.s0);
        printf("New password s1: %08x %08x\n", b0.s1, b1.s1);
        printf("New password s2: %08x %08x\n", b0.s2, b1.s2);
        printf("New password s3: %08x %08x\n", b0.s3, b1.s3);
#endif
#if CPU_DEBUG
        printf("New password s0: %c%c%c%c%c%c\n", b0.s0 & 0xff, (b0.s0 >> 8 ) & 0xff,
            (b0.s0 >> 16) & 0xff, (b0.s0 >> 24) & 0xff, b1.s0 & 0xff, (b1.s0 >> 8 ) & 0xff);
        printf("New password s0: %c%c%c%c%c%c\n", b0.s1 & 0xff, (b0.s1 >> 8 ) & 0xff,
            (b0.s1 >> 16) & 0xff, (b0.s1 >> 24) & 0xff, b1.s1 & 0xff, (b1.s1 >> 8 ) & 0xff);
        printf("New password s0: %c%c%c%c%c%c\n", b0.s2 & 0xff, (b0.s2 >> 8 ) & 0xff,
            (b0.s2 >> 16) & 0xff, (b0.s2 >> 24) & 0xff, b1.s2 & 0xff, (b1.s2 >> 8 ) & 0xff);
        printf("New password s0: %c%c%c%c%c%c\n", b0.s3 & 0xff, (b0.s3 >> 8 ) & 0xff,
            (b0.s3 >> 16) & 0xff, (b0.s3 >> 24) & 0xff, b1.s3 & 0xff, (b1.s3 >> 8 ) & 0xff);
#endif
        charsetOffset++;
        if (charsetOffset >= deviceCharsetLength) {
            charsetOffset = 0;
        }
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
        vstore_type(b0, 0 * deviceNumberOfChainsToRegen + password_index, initialPasswordArray);
#if PASSWORD_LENGTH > 4
        vstore_type(b1, (1 * deviceNumberOfChainsToRegen / VECTOR_WIDTH) + password_index, initialPasswordArray);
#endif
#if PASSWORD_LENGTH > 8
        vstore_type(b2, (2 * deviceNumberOfChainsToRegen / VECTOR_WIDTH) + password_index, initialPasswordArray);
#endif
    }

}