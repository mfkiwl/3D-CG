#!/bin/bash

logfile=test_results.log
echo "Running test cases"
python3 tests.py 2> $logfile