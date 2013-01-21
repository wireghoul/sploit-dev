// Defines for debugging

#ifndef MFN_DEBUGGING_H
#define MFN_DEBUGGING_H

/*
#define TRACE_PRINTF 1
#define KERNEL_LAUNCH_PRINTF 1
#define MT_PRINTF 1
#define STATIC_PRINTF 1
#define MEMALLOC_PRINTF 1
#define NETWORK_PRINTF 1
*/

// Defines for the trace printfs: Showing flow through the code.
// Use for things like announcing on entering and exiting a function.
#ifndef TRACE_PRINTF
    #define TRACE_PRINTF 0
#endif

#if TRACE_PRINTF
#define trace_printf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define trace_printf(fmt, ...) do {} while (0)
#endif


// Kernel launch printfs - things like the thread/block count, etc.
#ifndef KERNEL_LAUNCH_PRINTF
    #define KERNEL_LAUNCH_PRINTF 0
#endif

#if KERNEL_LAUNCH_PRINTF
#define klaunch_printf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define klaunch_printf(fmt, ...) do {} while (0)
#endif



// Multithreaded debugging related printfs
#ifndef MT_PRINTF
    #define MT_PRINTF 0
#endif

#if MT_PRINTF
#define mt_printf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define mt_printf(fmt, ...) do {} while (0)
#endif


// Static data setup printfs
#ifndef STATIC_PRINTF
    #define STATIC_PRINTF 0
#endif

#if STATIC_PRINTF
#define static_printf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define static_printf(fmt, ...) do {} while (0)
#endif


// Memory allocation printfs
#ifndef MEMALLOC_PRINTF
    #define MEMALLOC_PRINTF 0
#endif

#if MEMALLOC_PRINTF
#define memalloc_printf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define memalloc_printf(fmt, ...) do {} while (0)
#endif

// Network communication printfs
#ifndef NETWORK_PRINTF
    #define NETWORK_PRINTF 0
#endif

#if NETWORK_PRINTF
#define network_printf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define network_printf(fmt, ...) do {} while (0)
#endif

#endif

