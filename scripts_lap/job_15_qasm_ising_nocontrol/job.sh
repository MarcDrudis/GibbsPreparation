#!/bin/bash
parallel python3 script_qasm.py ::: {1..3}