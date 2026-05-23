#!/bin/bash

# Create a list of all .cpp files in the current folder
FILES=("src/LiH.cpp" "src/HF.cpp" "src/H2O_1d.cpp" "src/BeH2.cpp")

# Loop through each file
for f in "${FILES[@]}"

do
    # Extract the filename without the extension for the output binary
    FILENAME="${f%.*}"
    
    echo "------------------------------------------------"
    echo "Starting Task: $f"
    
    # Compile the file (using g++)
    # We check if compilation succeeds before running
    if g++ -O3 -I "include" -I "/home/jc6224/lib/eigen-5.0.0" "$f" -o "$FILENAME"; then
        
        # Capture the start time in nanoseconds
        START_TIME=$(date +%s%N)
        
        # Execute the binary
        ./"$FILENAME"
        
        # Capture the end time in nanoseconds
        END_TIME=$(date +%s%N)
        
        # Calculate duration in seconds (using bc for decimal math)
        # Note: 1 second = 1,000,000,000 nanoseconds
        DURATION=$(echo "scale=3; ($END_TIME - $START_TIME) / 1000000000" | bc)
        
        echo "Task $f finished in $DURATION seconds."
        
        # Clean up the binary after running (optional)
        rm "$FILENAME"
    else
        echo "Error: Compilation failed for $f"
    fi
done

echo "------------------------------------------------"
echo "All tasks complete."
#run LiH, HF, H2O, BeH2
