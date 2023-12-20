#!/usr/bin/env python
"""
 train a random forest with optional k-fold cross-validation
"""

import os
from os import listdir
from os.path import isfile, join, splitext
from glob import glob
from optparse import OptionParser
import tables
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib


## inputs 
parser = OptionParser()
parser.allow_interspersed_args = True
parser.add_option("-j", "--jobs", default=2, type=int,
                  dest="jobs", help="number of jobs to run in random forest")
parser.add_option("-o", "--outfile", default="mese", type=str,
                  dest="outfile", help="output names, includes numpy arrays and pickle of random forest")

(opts, args) = parser.parse_args()


## sort input files for train and test dataset 
data_dir =  "./MLtrack_data/rf_data"
infiles_all = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and ".npy" in f and "nugen_numu_" in f]
infiles_all = [data_dir + "/" + s for s in infiles_all]

nfiles = int(len(infiles_all)/2)  # half for training, half for testing  
print(f"nall={len(infiles_all)}")

infiles_train = infiles_all[:nfiles] 
infiles_test = infiles_all[nfiles:]
print(f"ntrain={len(infiles_train)}, ntest={len(infiles_test)}")

input_train = [] 
for infile in infiles_train: 
    input_train.append(np.load(infile, allow_pickle=True))

input_train = np.concatenate(input_train)
#print(input_train.shape)

input_test = [] 
for infile in infiles_test: 
    input_test.append(np.load(infile, allow_pickle=True))

input_test = np.concatenate(input_test)
#print(input_test.shape)


## load event weights
if "mese" in opts.outfile: 
    w_index = 0
elif "hese" in opts.outfile: 
    w_index = 1
elif "numu" in opts.outfile: 
    w_index = 2
else: 
    w_index = 3

if w_index > 2:
    sample_weight_train = np.ones(len(np.stack(input_train[:,2],axis=0)[:,0]))
    sample_weight_test = np.ones(len(np.stack(input_test[:,2],axis=0)[:,0]))
else:
    sample_weight_train = np.stack(input_train[:,2],axis=0)[:,w_index]
    sample_weight_test = np.stack(input_test[:,2],axis=0)[:,w_index]


## load features array, shape = (n_events,n_features)
x_train = np.stack(input_train[:,0],axis=0)
x_test = np.stack(input_test[:,0],axis=0)


## load target array, shape = (n_events,n_targets) if n_targets > 1
y_train = np.stack(input_train[:,1],axis=0)
y_test = np.stack(input_test[:,1],axis=0)


## create random forest and specify your additional options here
print("training RF")
rf = RandomForestRegressor(n_estimators=400,
                          max_depth=20,
                          min_samples_leaf=1,
                          max_features=1.0,
                           n_jobs=opts.jobs)

## train with full sample and save pickle
rf.fit(x_train, y_train, sample_weight=sample_weight_train)
joblib.dump(rf, "./MLtrack_data/rf_data/rf_"+opts.outfile+".pkl")


## save numpy arrays 
print("saving arrays with rf")
full_array =  []
part_i = 0
length_sample = len(x_test)
for i in range(length_sample):
    weight = sample_weight_test[i]
    true_array = y_test[i]; pred_array = x_test[i]

    ta = 10**true_array
    pa = 10**rf.predict([pred_array])[0]
    full_array.append([ta[0], ta[1], ta[0]+ta[1],  # ehad, emu, enu (true)
                       ta[2], ta[3], ta[2]+ta[3],  # edephad, edepmu, edepnu (true)
                       pa[0], pa[1], pa[0]+pa[1],  # ehad, emu, enu (pred)
                       pa[2], pa[3], pa[2]+pa[3],  # edephad, edepmu, edepnu (pred)
                       weight])
    part_i +=1  # for quick checks
    if part_i%1e4 < 1:
        print("event %1.0f"%part_i)
        #break

print("saving np")
np.save("./MLtrack_data/rf_data/%s_saved_arrays_with_predictions_n400_md_20.npy"%opts.outfile, full_array)
