#!/bin/bash

# configure and compile the program
mkdir -p build
cd build
rm -rf ./*
cmake ..
make

# run the program
echo "==================== Running the program ===================="
./cudacase
