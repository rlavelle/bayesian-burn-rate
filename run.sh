#!/usr/bin/env bash
set -euo pipefail

funds=$(wc -l < data/funds.txt)
spend=$(wc -l < data/spend.txt)

echo "$spend"

if [ "$funds" -eq "$spend" ]; then
	gcc -o bayesian_main ccore/*.c ccore/distributions/*.c ccore/samplers/*.c ccore/simulation/*.c -lm
	./bayesian_main	
	python3 python/visualization.py --max_idx="$spend"
else
	echo "Line counts differ: funds=$funds, spend=$spend"
fi
