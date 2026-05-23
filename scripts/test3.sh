#!/bin/bash

#================ FILES TO COMPILE ================#

FILES=(
    src/test.cpp
    cuda/eri_gpu.cu
)

#================ COMPILER SETTINGS ===============#

OUT="hf"

FLAGS=(
    -O3
    -std=c++20
    -Iinclude
    -I "/home/jc6224/lib/eigen-5.0.0"
)

#==================================================#

echo
echo "================ COMPILATION ================"

START_COMPILE=$(date +%s.%N)

nvcc "${FILES[@]}" "${FLAGS[@]}" -o $OUT

COMPILE_STATUS=$?

END_COMPILE=$(date +%s.%N)

COMPILE_TIME=$(echo "$END_COMPILE - $START_COMPILE" | bc)

if [ $COMPILE_STATUS -ne 0 ]; then
    echo
    echo "Compilation failed."
    exit 1
fi

echo "Compilation finished in $COMPILE_TIME seconds."

echo
echo "================ EXECUTION ================="

START_RUN=$(date +%s.%N)

./$OUT

END_RUN=$(date +%s.%N)

RUN_TIME=$(echo "$END_RUN - $START_RUN" | bc)

echo
echo "Execution finished in $RUN_TIME seconds."