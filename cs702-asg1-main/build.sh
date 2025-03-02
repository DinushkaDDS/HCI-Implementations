#!/bin/bash
python setup.py build_ext --inplace
rm -rf build/ # Clean up build directory
mv q3.cpython-312-aarch64-linux-gnu.so q3