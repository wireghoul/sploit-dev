#!/bin/bash

g++ -I../../../inc/CH_Common/ -I../../../inc/GRT_Common/  -o GRTParallelSort testGRTParallelSort.cpp ../../CH_Common/CHHiresTimer.cpp
