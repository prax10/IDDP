#!/bin/bash
cd /home/claude/cap
python3 -u task_h3_confound.py > reports/h3_run_log.txt 2>&1
python3 -u task_h1_percheckpoint_tuning.py > reports/h1_run_log.txt 2>&1
python3 -u task_h2_text_pca_sweep.py > reports/h2_run_log.txt 2>&1
echo ALLDONE > reports/done.flag
