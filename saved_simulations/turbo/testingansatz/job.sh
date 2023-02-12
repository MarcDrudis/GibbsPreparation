#!/bin/bash
for i in $(seq -0.25 0.05 0.25)
do
   /home/drudis/python_environements/test_easy_instalation/bin/python3.10 control_fields_classic.py $i YYII $i IYYI $i IIYY
done
