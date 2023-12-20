#!/bin/sh

infile=${1}
gcdfile=${2}
outfile=${3}
echo ${infile} ${gcdfile} ${outfile}

cd /home/yashida/MLtrack/rf_energy

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`
bash /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/RHEL_7_x86_64/metaprojects/icetray/v1.3.3/env-shell.sh python prep.py -i ${infile} -g ${gcdfile} -o ${outfile}
