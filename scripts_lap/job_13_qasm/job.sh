#!/bin/bash
parallel /home/drudis/python_environements/test_easy_instalation/bin/python3.10 script_qasm.py ::: {1..10}