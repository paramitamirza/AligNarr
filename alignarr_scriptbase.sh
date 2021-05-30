#!/bin/bash

END=913
for i in $(seq 1 $END); 
do 
	python -u -W ignore alignarr.py scriptbase /GW/D5data-14/scriptbase_extracted/ $i; 
done