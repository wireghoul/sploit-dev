
//#define CPU_DEBUG 1

#ifdef CPU_DEBUG
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif



__kernel __attribute__((vec_type_hint(vector_type))) void CandidateHashSHA1AMD(
    __constant unsigned char *deviceCharset,/* 0 */
    __private unsigned int deviceCharsetLength,/* 1 */
    __private unsigned int deviceChainLength,/* 2 */
    __private unsigned int deviceTableIndex,/* 3 */
    __private unsigned int deviceNumberOfThreads,/* 4 */
    __constant unsigned int *SHA1_Candidate_Device_Hash,/* 5 */
    __global   unsigned int *outputCandidateHashArray, /* 6 */
    __private   unsigned int threadSpaceOffset,/* 7 */
    __private   unsigned int startStep,/* 8 */
    __private   unsigned int stepsToRun/* 9 */
) {

    // Start the kernel.
    __local unsigned char charset[512];

#if CPU_DEBUG
    printf("\n\n\n");
    printf("Kernel start, global id %d\n", get_global_id(0));
    printf("deviceCharsetLength: %d\n", deviceCharsetLength);
    printf("deviceCharset: %c %c %c %c ...\n", deviceCharset[0], deviceCharset[1], deviceCharset[2], deviceCharset[3]);
    printf("deviceChainLength: %d\n", deviceChainLength);
    printf("deviceTableIndex: %d\n", deviceTableIndex);
    printf("deviceNumberOfThreads: %d\n", deviceNumberOfThreads);
    printf("threadSpaceOffset: %d\n", threadSpaceOffset);
    printf("startStep: %d\n", startStep);
    printf("stepsToRun: %d\n", stepsToRun);
    printf("Candidate hash: %08x %08x %08x %08x\n", SHA1_Candidate_Device_Hash[0],
            SHA1_Candidate_Device_Hash[1], SHA1_Candidate_Device_Hash[2], SHA1_Candidate_Device_Hash[3]);
    printf("PASSWORD_LENGTH: %d\n", PASSWORD_LENGTH);
    for(int i = 0; i < 512; i++) {
        printf("%c.", deviceCharset[i]);
    }
#endif


    // Hash variables
    vector_type b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
    vector_type a,b,c,d,e;
    // Vector to hold the initial values until we are done with them.
    // This is to load up the vector space nicely.
    uint init_a, init_b, init_c, init_d, init_e;

    uint i, chain_index, step_to_calculate, charset_offset, last_step_for_iteration;

    // Generic "copy charset to shared memory" function
    copySingleCharsetToShared(charset, deviceCharset);
    //printf("Charset copied... %c %c %c %c ...\n", charset[0], charset[1], charset[2], charset[3]);

    chain_index = (get_global_id(0) * VECTOR_WIDTH + (threadSpaceOffset * deviceNumberOfThreads));

#if CPU_DEBUG
    printf("chain index: %d\n", chain_index);
#endif

    // Find out if we're done with work.
    // If our index + the startstep is greater than the chain length, this thread has nothing to do.
    if ((chain_index) >= deviceChainLength) {
#if CPU_DEBUG
        printf("returning: chain_index %d  start_step %d  chain length %d \n", chain_index, startStep, deviceChainLength);
#endif
        return;
    }



    // Figure out which step we're running.
    step_to_calculate = chain_index + startStep;
#if CPU_DEBUG
    printf("Step to calculate: %d \n", step_to_calculate);
#endif

    // Yes, modulus here is slow.  And this is a less-critical chunk of code.  So it stays here.
    charset_offset = step_to_calculate % deviceCharsetLength;

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

    // Load the initial hash.  This will either be from the constant or from the storage space.
    // If it is from the constant area, load it into the init_{a,b,c,d} vars so we can copy
    // it as needed.
    if (startStep == 0) {
        init_a = (uint)SHA1_Candidate_Device_Hash[0];
        init_b = (uint)SHA1_Candidate_Device_Hash[1];
        init_c = (uint)SHA1_Candidate_Device_Hash[2];
        init_d = (uint)SHA1_Candidate_Device_Hash[3];
        init_e = (uint)SHA1_Candidate_Device_Hash[4];
#if CPU_DEBUG
        printf("Loaded candidate %08x %08x %08x %08x %08x\n", init_a, init_b, init_c, init_d, init_e);
#endif
        a = (vector_type) 0;
        b = (vector_type) 0;
        c = (vector_type) 0;
        d = (vector_type) 0;
        e = (vector_type) 0;

    // Now step through the vectors to load them properly...
 #if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
        a.s0 = init_a;
        b.s0 = init_b;
        c.s0 = init_c;
        d.s0 = init_d;
        e.s0 = init_e;
        ClearB0ToB15(&b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        reduceSingleCharsetNormal(&b0, &b1, &b2, a, b, c, d, step_to_calculate, charset, charset_offset, PASSWORD_LENGTH, deviceTableIndex);
    #if CPU_DEBUG
        printf("Reduced password .s0: %08x %08x %08x\n", b0.s0, b1.s0, b2.s0);
    #endif
        charset_offset++;
        if (charset_offset >= deviceCharsetLength) {
            charset_offset = 0;
        }
        step_to_calculate++;
        stepsToRun--;
        padSHAHash(PASSWORD_LENGTH, &b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        a = reverse(a);b = reverse(b);c = reverse(c);d = reverse(d);e = reverse(e);
    #if CPU_DEBUG
        printf("SHA1 result .s0: %08x %08x %08x %08x \n", a.s0, b.s0, c.s0, d.s0);
        printf("SHA1 result .s1: %08x %08x %08x %08x \n", a.s1, b.s1, c.s1, d.s1);
    #endif
        a.s1 = init_a;
        b.s1 = init_b;
        c.s1 = init_c;
        d.s1 = init_d;
        e.s1 = init_e;
    #endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
        ClearB0ToB15(&b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        reduceSingleCharsetNormal(&b0, &b1, &b2, a, b, c, d, step_to_calculate, charset, charset_offset, PASSWORD_LENGTH, deviceTableIndex);
    #if CPU_DEBUG
        printf("Reduced password .s0: %08x %08x %08x\n", b0.s0, b1.s0, b2.s0);
    #endif
        charset_offset++;
        if (charset_offset >= deviceCharsetLength) {
            charset_offset = 0;
        }
        step_to_calculate++;
        stepsToRun--;
        padSHAHash(PASSWORD_LENGTH, &b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        a = reverse(a);b = reverse(b);c = reverse(c);d = reverse(d);e = reverse(e);
    #if CPU_DEBUG
        printf("SHA1 result .s0: %08x %08x %08x %08x \n", a.s0, b.s0, c.s0, d.s0);
        printf("SHA1 result .s1: %08x %08x %08x %08x \n", a.s1, b.s1, c.s1, d.s1);
    #endif
        a.s2 = init_a;
        b.s2 = init_b;
        c.s2 = init_c;
        d.s2 = init_d;
        e.s2 = init_e;
        ClearB0ToB15(&b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        reduceSingleCharsetNormal(&b0, &b1, &b2, a, b, c, d, step_to_calculate, charset, charset_offset, PASSWORD_LENGTH, deviceTableIndex);
    #if CPU_DEBUG
        printf("Reduced password .s0: %08x %08x %08x\n", b0.s0, b1.s0, b2.s0);
    #endif
        charset_offset++;
        if (charset_offset >= deviceCharsetLength) {
            charset_offset = 0;
        }
        step_to_calculate++;
        stepsToRun--;
        padSHAHash(PASSWORD_LENGTH, &b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        a = reverse(a);b = reverse(b);c = reverse(c);d = reverse(d);e = reverse(e);
    #if CPU_DEBUG
        printf("SHA1 result .s0: %08x %08x %08x %08x \n", a.s0, b.s0, c.s0, d.s0);
        printf("SHA1 result .s1: %08x %08x %08x %08x \n", a.s1, b.s1, c.s1, d.s1);
    #endif
        a.s3 = init_a;
        b.s3 = init_b;
        c.s3 = init_c;
        d.s3 = init_d;
        e.s3 = init_e;
    #endif

    } else {
        a = vload_type( 0 * deviceChainLength                 + chain_index / VECTOR_WIDTH, outputCandidateHashArray);
        b = vload_type((1 * deviceChainLength / VECTOR_WIDTH) + chain_index / VECTOR_WIDTH, outputCandidateHashArray);
        c = vload_type((2 * deviceChainLength / VECTOR_WIDTH) + chain_index / VECTOR_WIDTH, outputCandidateHashArray);
        d = vload_type((3 * deviceChainLength / VECTOR_WIDTH) + chain_index / VECTOR_WIDTH, outputCandidateHashArray);
        e = vload_type((4 * deviceChainLength / VECTOR_WIDTH) + chain_index / VECTOR_WIDTH, outputCandidateHashArray);
#if CPU_DEBUG
    printf("Loaded hash from location %d .s0:  %08x %08x %08x %08x %08x\n", chain_index,
            a.s0, b.s0, c.s0, d.s0, e.s0);
    printf("Loaded hash from location %d .s1:  %08x %08x %08x %08x %08x\n", chain_index,
            a.s1, b.s1, c.s1, d.s1, e.s1);
#endif

    }




#if CPU_DEBUG
    printf("Entering reduceCharset...\n");
    printf("Charset offset: %d\n", charset_offset);
#endif
    ClearB0ToB15(&b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
    reduceSingleCharsetNormal(&b0, &b1, &b2, a, b, c, d, step_to_calculate, charset, charset_offset, PASSWORD_LENGTH, deviceTableIndex);
#if CPU_DEBUG
    printf("Reduced password .s0: %08x %08x %08x\n", b0.s0, b1.s0, b2.s0);
    printf("Reduced password .s1: %08x %08x %08x\n", b0.s1, b1.s1, b2.s1);
#endif

    step_to_calculate++;
    charset_offset++;
    if (charset_offset >= deviceCharsetLength) {
        charset_offset = 0;
    }
    // Figure out the last step to run - either the chain length or
    // the number of specified steps.
    if ((step_to_calculate + stepsToRun) > deviceChainLength) {
        last_step_for_iteration = deviceChainLength - 1;
    } else {
        last_step_for_iteration = (step_to_calculate + stepsToRun - 1); // Already run one
    }
#if CPU_DEBUG
    printf("last_step_for_iteration: %d \n", last_step_for_iteration);
#endif


    // We now have our (step+1) charset.
    for (i = step_to_calculate; i <= last_step_for_iteration; i++) {
#if CPU_DEBUG
    printf("Chain step %d \n", i);
#endif
        padSHAHash(PASSWORD_LENGTH, &b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        a = reverse(a);b = reverse(b);c = reverse(c);d = reverse(d);e = reverse(e);
#if CPU_DEBUG
    printf("SHA1 result .s0: %08x %08x %08x %08x \n", a.s0, b.s0, c.s0, d.s0);
    printf("SHA1 result .s1: %08x %08x %08x %08x \n", a.s1, b.s1, c.s1, d.s1);
#endif
        ClearB0ToB15(&b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        reduceSingleCharsetNormal(&b0, &b1, &b2, a, b, c, d, i, charset, charset_offset, PASSWORD_LENGTH, deviceTableIndex);
#if CPU_DEBUG
    printf("Reduced password .s0: %08x %08x %08x\n", b0.s0, b1.s0, b2.s0);
    printf("Reduced password .s1: %08x %08x %08x\n", b0.s1, b1.s1, b2.s1);
#endif
        charset_offset++;
        if (charset_offset >= deviceCharsetLength) {
            charset_offset = 0;
        }
    }
    // Store the hash output.
#if CPU_DEBUG
    if (chain_index > deviceChainLength) {
        printf("ERROR: SHOULD NOT WRITE THIS!!!  Chain index to store: %d\n", chain_index);
    }
#endif
    vstore_type(a,  0 * deviceChainLength                 + chain_index / VECTOR_WIDTH, outputCandidateHashArray);
    vstore_type(b, (1 * deviceChainLength / VECTOR_WIDTH) + chain_index / VECTOR_WIDTH, outputCandidateHashArray);
    vstore_type(c, (2 * deviceChainLength / VECTOR_WIDTH) + chain_index / VECTOR_WIDTH, outputCandidateHashArray);
    vstore_type(d, (3 * deviceChainLength / VECTOR_WIDTH) + chain_index / VECTOR_WIDTH, outputCandidateHashArray);
    vstore_type(e, (4 * deviceChainLength / VECTOR_WIDTH) + chain_index / VECTOR_WIDTH, outputCandidateHashArray);
#if CPU_DEBUG
    printf("Stored candidate .s0 %08x %08x %08x %08x \n", a.s0, b.s0, c.s0, d.s0);
    printf("Stored candidate .s1 %08x %08x %08x %08x \n", a.s1, b.s1, c.s1, d.s1);
#endif
}


