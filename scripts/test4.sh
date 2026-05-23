#!/bin/bash

# Configuration
FILE="src/H2_2dslice.cpp"
OUT="app"
STD="c++20"

# 1. Compilation
echo "🛠️  Compiling..."
g++ -std=$STD -I "include" -I "/home/jc6224/lib/eigen-5.0.0" -O3 "$FILE" -o "$OUT"

if [ $? -eq 0 ]; then
    echo "🚀 Run Start:"
    echo "------------------------"
    
    # Start timer
    START_TIME=$(date +%s.%N)
    
    ./"$OUT"
    
    # End timer
    END_TIME=$(date +%s.%N)
    
    echo "------------------------"
    
    # Calculate duration
    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    
    echo "⏱️  Execution time: $ELAPSED seconds"
else
    echo "❌ Build failed."
    exit 1
fi

#tonight: H2O, CO2, HCl, MgO, N2
