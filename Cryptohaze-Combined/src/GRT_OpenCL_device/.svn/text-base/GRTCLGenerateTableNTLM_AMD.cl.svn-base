// Kernels for NTLM - vectorized for AMD...


__kernel __attribute__((vec_type_hint(vector_type))) void GenerateNTLMAMD(
    __constant unsigned char *deviceCharset,
    __private unsigned int deviceCharsetLength,
    __private unsigned int deviceChainLength,
    __private unsigned int deviceNumberOfChains,
    __private unsigned int deviceTableIndex,
    __private unsigned int deviceNumberOfThreads,
    __global   unsigned int *initialPasswordArray,
    __global   unsigned int *outputHashArray,
    __private   unsigned int passwordSpaceOffset,
    __private   unsigned int startChainIndex,
    __private   unsigned int stepsToRun,
    __private   unsigned int charsetOffset
) {

    // Start the kernel.
    __local unsigned char charset[512];

#if CPU_DEBUG
    printf("\n\n\n");
    printf("Kernel start, global id %d\n", get_global_id(0));
    printf("deviceCharsetLength: %d\n", deviceCharsetLength);
    printf("deviceCharset: %c %c %c %c ...\n", deviceCharset[0], deviceCharset[1], deviceCharset[2], deviceCharset[3]);
    printf("deviceChainLength: %d\n", deviceChainLength);
    printf("deviceNumberOfChains: %d\n", deviceNumberOfChains);
    printf("deviceTableIndex: %d\n", deviceTableIndex);
    printf("deviceNumberOfThreads: %d\n", deviceNumberOfThreads);
    printf("passwordSpaceOffset: %d\n", passwordSpaceOffset);
    printf("startChainIndex: %d\n", startChainIndex);
    printf("stepsToRun: %d\n", stepsToRun);
    printf("charsetOffset: %d\n", charsetOffset);
#endif

    // Needed variables for generation
    uint CurrentStep, PassCount, password_index;

    // Hash variables
    vector_type b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
    vector_type a,b,c,d;

    // Generic "copy charset to shared memory" function
    copySingleCharsetToShared(charset, deviceCharset);
    //printf("Charset copied... %c %c %c %c ...\n", charset[0], charset[1], charset[2], charset[3]);

    // Figure out which password we are working on.
    password_index = (get_global_id(0) + (passwordSpaceOffset * deviceNumberOfThreads));
#if CPU_DEBUG
    printf("password index: %d\n", password_index);
    printf("startChainIndex: %d\n", startChainIndex);
#endif
    // Return if this thread is working on something beyond the end of the password space
    if (password_index >= (deviceNumberOfChains / VECTOR_WIDTH)) {
#if CPU_DEBUG
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
    b15 = vload_type(0 * deviceNumberOfChains + password_index, initialPasswordArray);
    b0 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
#if PASSWORD_LENGTH > 2
    b1 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
#endif
#if PASSWORD_LENGTH > 4
    b15 = vload_type((1 * deviceNumberOfChains / VECTOR_WIDTH) + password_index, initialPasswordArray);
    b2 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 6
    b3 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
#endif
#if PASSWORD_LENGTH > 8
    b15 = vload_type((2 * deviceNumberOfChains / VECTOR_WIDTH) + password_index, initialPasswordArray);
    b4 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
#endif
#if PASSWORD_LENGTH > 10
    b5 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
#endif
    b15 = (vector_type)0x00000000;
    
#if CPU_DEBUG
    printf("Initial loaded password s0: %08x %08x %08x %08x %08x %08x\n", b0.s0, b1.s0, b2.s0, b3.s0, b4.s0, b5.s0);
    printf("Initial loaded password s1: %08x %08x %08x %08x\n", b0.s1, b1.s1, b2.s1, b3.s1);
    printf("Initial loaded password s2: %08x %08x %08x %08x\n", b0.s2, b1.s2, b2.s2, b3.s2);
    printf("Initial loaded password s3: %08x %08x %08x %08x\n", b0.s3, b1.s3, b2.s3, b3.s3);
#endif
    for (PassCount = 0; PassCount < stepsToRun; PassCount++) {
        CurrentStep = PassCount + startChainIndex;
#if CPU_DEBUG
        printf("\nChain %d, step %d\n", password_index, PassCount);
#endif
        padMDHash(PASSWORD_LENGTH * 2, &b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
#if CPU_DEBUG
    printf("To-run password s0: %08x %08x %08x %08x %08x %08x\n", b0.s0, b1.s0, b2.s0, b3.s0, b4.s0, b5.s0);
    printf("Initial loaded password s1: %08x %08x %08x %08x\n", b0.s1, b1.s1, b2.s1, b3.s1);
    printf("Initial loaded password s2: %08x %08x %08x %08x\n", b0.s2, b1.s2, b2.s2, b3.s2);
    printf("Initial loaded password s3: %08x %08x %08x %08x\n", b0.s3, b1.s3, b2.s3, b3.s3);
#endif
        OpenCL_MD4(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, &a, &b, &c, &d);
#if CPU_DEBUG
        printf("\n\n\nNTLM result s0... %08x %08x %08x %08x\n", a.s0, b.s0, c.s0, d.s0);
        printf("NTLM result s1... %08x %08x %08x %08x\n", a.s1, b.s1, c.s1, d.s1);
        printf("NTLM result s2... %08x %08x %08x %08x\n", a.s2, b.s2, c.s2, d.s2);
        printf("NTLM result s3... %08x %08x %08x %08x\n", a.s3, b.s3, c.s3, d.s3);
#endif
        reduceSingleCharsetNTLM(&b0, &b1, &b2, &b3, &b4, &b5, a, b, c, d, CurrentStep, charset, charsetOffset, PASSWORD_LENGTH, deviceTableIndex);
#if CPU_DEBUG
        printf("New password s0: %08x %08x %08x %08x %08x %08x\n", b0.s0, b1.s0, b2.s0, b3.s0, b4.s0, b5.s0);
        printf("New password s1: %08x %08x %08x %08x\n", b0.s1, b1.s1, b2.s1, b3.s1);
        printf("New password s2: %08x %08x %08x %08x\n", b0.s2, b1.s2, b2.s2, b3.s2);
        printf("New password s3: %08x %08x %08x %08x\n", b0.s3, b1.s3, b2.s3, b3.s3);
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
        printf("\nstoring output chain s0: %08x %08x %08x %08x\n", a.s0, b.s0, c.s0, d.s0);
        printf("\nstoring output chain s1: %08x %08x %08x %08x\n", a.s1, b.s1, c.s1, d.s1);
        printf("\nstoring output chain s2: %08x %08x %08x %08x\n", a.s2, b.s2, c.s2, d.s2);
        printf("\nstoring output chain s3: %08x %08x %08x %08x\n", a.s3, b.s3, c.s3, d.s3);
#endif
        vstore_type(a, 0 * deviceNumberOfChains + password_index, outputHashArray);
        vstore_type(b, (1 * deviceNumberOfChains / VECTOR_WIDTH) + password_index, outputHashArray);
        vstore_type(c, (2 * deviceNumberOfChains / VECTOR_WIDTH) + password_index, outputHashArray);
        vstore_type(d, (3 * deviceNumberOfChains / VECTOR_WIDTH) + password_index, outputHashArray);
    }
    // Else, store the b0/b1 values back to the initial array for the next loop
    else {
#if CPU_DEBUG
        printf("storing state s0: %08x %08x %08x %08x\n", b0.s0, b1.s0, b2.s0, b3.s0);
        printf("storing state s1: %08x %08x %08x %08x\n", b0.s1, b1.s1, b2.s1, b3.s1);
        printf("storing state s2: %08x %08x %08x %08x\n", b0.s2, b1.s2, b2.s2, b3.s2);
        printf("storing state s3: %08x %08x %08x %08x\n", b0.s3, b1.s3, b2.s3, b3.s3);
#endif

        b15 =  (b0 & 0xff) | ((b0 & 0xff0000) >> 8);
#if PASSWORD_LENGTH > 2
        b15 |= (b1 & 0xff) << 16 | ((b1 & 0xff0000) << 8);
#endif
#if CPU_DEBUG
        printf("storing state b15: %08x %08x %08x %08x\n", b15.s0, b15.s1, b15.s2, b15.s3);
#endif
        // Need to store this for all cases.
        vstore_type(b15, 0 * deviceNumberOfChains + password_index, initialPasswordArray);

#if PASSWORD_LENGTH > 4
        b15 =  (b2 & 0xff) | ((b2 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 6
        b15 |= (b3 & 0xff) << 16 | ((b3 & 0xff0000) << 8);

    #endif
        vstore_type(b15, (1 * deviceNumberOfChains / VECTOR_WIDTH) + password_index, initialPasswordArray);
#endif

#if PASSWORD_LENGTH > 8
        b15 =  (b4 & 0xff) | ((b4 & 0xff0000) >> 8);
    #if PASSWORD_LENGTH > 10
        b15 |= (b5 & 0xff) << 16 | ((b5 & 0xff0000) << 8);

    #endif
        vstore_type(b15, (2 * deviceNumberOfChains / VECTOR_WIDTH) + password_index, initialPasswordArray);
#endif
    }
}
