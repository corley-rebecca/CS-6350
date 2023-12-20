#!/bin/sh

python make_dagman.py 
condor_submit_dag dagman_job_submission.dag
