#!/usr/bin/env python
"""
 collect random forest features and targets and write to a numpy file
"""

import os
from glob import glob
from optparse import OptionParser
import numpy as np
import matplotlib.path as mpltPath
from icecube import dataclasses, icetray, MuonGun
from icecube.rootwriter import I3ROOTWriter
from icecube.hdfwriter import I3HDFWriter
from icecube.simprod import segments
from I3Tray import *
from utils import *


## inputs
parser = OptionParser()
parser.allow_interspersed_args = True
parser.add_option("-i", "--infile", 
                  default="/data/ana/Muon/ESTES/ESTES_2019/ESTES_data/nugen_numu_21813_p0=0.0_p1=0.0/nugen_numu_21813_p0=0.0_p1=0.0_0_step4.i3.bz2", 
                  type=str,  dest="infile", help="input .i3 file")
parser.add_option("-g", "--gcdfile", 
                  default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz", 
                  type=str, dest="gcdfile", help="GCD file")
parser.add_option("-o", "--outfile", 
                  default="./MLtrack_data/out.npy", 
                  type=str,  dest="outfile", help="output .npy file")

(opts, args) = parser.parse_args()
infiles = [opts.gcdfile, opts.infile] 
print(infiles)

surface_det = MuonGun.ExtrudedPolygon.from_file(opts.gcdfile, padding=50)


tray = I3Tray()
tray.AddModule("I3Reader", "reader", FilenameList=infiles)


## apply precut 
def pre_cut(frame):
    # numu CC only
    wd = frame["I3MCWeightDict"]
    if wd['InteractionType'] != 1 or abs(wd["PrimaryNeutrinoType"]) != 14:
        return False

    # find in-ice neutrino, muon
    tree = frame["I3MCTree_preMuonProp"]
    nu = tree.primaries[0]
    if not nu.is_neutrino:
        return False
    
    child = tree.first_child(nu)
    while child.is_neutrino:
        nu = child
        child = tree.first_child(nu)
    
    muon = child

    # find intersection points of muon track with detector volume
    ip = phys_services.Surface.intersection(surface_det, muon.pos, muon.dir)
    ip = [ip.first, ip.second]
    if len(ip) > 0:
        length_before = ip[0]
        length_after = ip[-1]        
        # remove if vertex outside detector
        if (length_before > 0) or (length_after < 0):
            return False
    else:
        return False

    # remove badly reconstructed events 
    track = frame["Millipede_Free_Best_ESTES_Fit_1"]
    if np.degrees(np.arccos(track.dir*muon.dir)) > 5:
        return False
    
    return True    


tray.AddModule(pre_cut, "pre_cut")


bound_2D = []
x = [(surface_det.x[i],surface_det.y[i]) for i in range(len(surface_det.x))]
bound_2D = mpltPath.Path(x) # projection of detector on x,y plane

def boundary_check(particle):
    inlimit = False
    if ((particle.pos.z <= max(surface_det.z)) and (particle.pos.z >= min(surface_det.z))):
        if bound_2D.contains_points([(particle.pos.x, particle.pos.y)]):
            inlimit = True
    
    return inlimit


## find deposited energy
def dep_energy(frame):
    dep_casc_energy = 0
    dep_muon_energy = 0
    
    # find energy deposited due to secondaries
    tree = frame["I3MCTree"]
    for primary in tree.primaries:
        if primary.is_neutrino:
            nu = primary
    
    child = tree.first_child(nu)
    while child.is_neutrino:
        if tree.number_of_children(child)>0:
            nu = child
            child = tree.first_child(nu)
        else:
            break

    # first extract cascade dep energy
    neut_children = tree.get_daughters(nu)
    if abs(neut_children[0].type) == 13:
        hadron = neut_children[1]
    else:
        hadron = neut_children[0]

    had_daughters = tree.get_daughters(hadron)
    for particle in had_daughters:
        if (particle.shape.name != "Dark") and (particle.shape.name != "Primary"):
            if (particle.type.name != "MuPlus") and (particle.type.name != "MuMinus"):
                if boundary_check(particle):
                    dep_casc_energy += particle.energy
    
    # find energy deposited from track only
    for track in MuonGun.Track.harvest(frame["I3MCTree"], frame["MMCTrackList"]):
        intersections = surface_det.intersection(track.pos, track.dir)
        e0, e1 = track.get_energy(intersections.first), track.get_energy(intersections.second)
        dep_muon_energy +=  (e0-e1)
    
    frame["Deposited_Cascade_Energy"] = dataclasses.I3Double(dep_casc_energy)
    frame["Deposited_Muon_Energy"] = dataclasses.I3Double(dep_muon_energy)


tray.Add(dep_energy, Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics])

tray.AddModule("Delete", "delete", Keys=["RandomForestFeatures","RandomForestEnergyEstimates","RandomForestTarget"])
tray.AddModule(RandomForestCollect, "collect",
               MillipedeName="Millipede_Free_SplineMPE_SPEFit2_Free_1_10",
               NQuantiles=26,
               FeaturesName="RandomForestFeatures",
               TargetName="RandomForestTarget",
               IsStartingTrack=True,
               Cleanup=True)

    
## apply weighting 
def simple_weight(frame):
    ow = frame["I3MCWeightDict"]["OneWeight"]
    e = frame["I3MCWeightDict"]["PrimaryNeutrinoEnergy"]
    nev = frame["I3MCWeightDict"]["NEvents"]
    meseflux = (2.06*10**-18)*((e/(100000.))**-2.46)
    frame["MESEWeight"] = dataclasses.I3Double(ow*meseflux/(nev/2))
    heseflux = (2.15*10**-18)*((e/(100000.))**-2.89)
    frame["HESEWeight"] = dataclasses.I3Double(ow*heseflux/(nev/2))
    numuflux = (1.44*10**-18)*((e/(100000.))**-2.28)
    frame["numuWeight"] = dataclasses.I3Double(ow*numuflux/(nev/2))
    return True


tray.AddModule(simple_weight, "weight")


## save into numpy files 
global output_array; output_array = [];
def prepare_numpy_arrays(frame):
    RFF = frame["RandomForestFeatures"]
    RFT = frame["RandomForestTarget"]
    weights = [frame["MESEWeight"].value,frame["HESEWeight"].value,frame["numuWeight"].value]
    output_array.append([RFF,RFT,weights])
    if len(output_array)%1000 == 0: 
        print(len(output_array))
    
    return True


tray.AddModule(prepare_numpy_arrays, Streams=[icetray.I3Frame.Physics])
tray.Execute()
tray.Finish()

outfile = opts.outfile
if (len(output_array) > 0):
    np.save(outfile, output_array)
