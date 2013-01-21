#ifndef __TIMER_H
#define __TIMER_H

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

using namespace std;

class Timer {
private:

    timeval startTime;

public:

    void start(){
        gettimeofday(&startTime, NULL);
    }

    double stop(){
        timeval endTime;
        long seconds, nseconds;
        double duration;

        gettimeofday(&endTime, NULL);

        seconds  = endTime.tv_sec  - startTime.tv_sec;
        nseconds = endTime.tv_usec - startTime.tv_usec;

        duration = seconds + nseconds/1000000.0;

        return duration;
    }

    static void printTime(double duration){
        printf("%5.6f seconds\n", duration);
    }
};

#endif