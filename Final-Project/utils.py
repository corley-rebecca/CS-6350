import numpy as np
from numpy import diff
from icecube import dataclasses, icetray, phys_services, simclasses, MuonGun
from I3Tray import *


"""
 collect features and targets for random forest training
"""
class RandomForestCollect(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox("OutBox")
        self.AddParameter("MillipedeName", "Name of millipede loss vector", "SplineMPE_MillipedeHighEnergyMIE")
        self.AddParameter("FeaturesName", "Name of output feature vector", "RandomForestFeatures")
        self.AddParameter("TargetName", "Name of target vector", "RandomForestTarget")
        self.AddParameter("NQuantiles", "Number of quantiles of energy loss distribution to use", 101)
        self.AddParameter("IsStartingTrack", "Learn neutrino energy/inelasticity of starting tracks? Otherwise learn most energetic muon energy", True)
        self.AddParameter("DoInelasticity", "Learn hadronic and muon energies of starting tracks", True)
        self.AddParameter("Cleanup", "Remove features/targets that are nan/inf", True)

    def Configure(self):
        self.MillipedeName = self.GetParameter("MillipedeName")
        self.nQuantiles = self.GetParameter("NQuantiles")
        self.isStartingTrack = self.GetParameter("IsStartingTrack")
        self.doInelasticity = self.GetParameter("DoInelasticity")

    def Geometry(self, frame):
        self.PushFrame(frame)
        return True

    def Calibration(self, frame):
        self.PushFrame(frame)
        pass

    def DetectorStatus(self, frame):
        self.PushFrame(frame)
        pass

    def Physics(self, frame):
        track = frame[self.MillipedeName]
        if (np.sqrt(track.pos.x**2 + track.pos.y**2 + track.pos.z**2) > 1e3):
            IC0 = dataclasses.I3Position(0, 0, 0)
            CAP = track.pos + track.dir*(track.dir*(IC0 - track.pos))
            CAT = (phys_services.I3Calculator.distance_along_track(track,CAP))/track.speed + track.time
            track.pos = CAP; track.time = CAT;
        
        reco_zen = track.dir.zenith
        # find detector entrance and exit times
        gcdname = "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
        surface_det = MuonGun.ExtrudedPolygon.from_file(gcdname, padding=50)
        ip = phys_services.Surface.intersection(surface_det, track.pos, track.dir)
        ip = [ip.first, ip.second]

        if len(ip)>0:
            t_start = (ip[0])/track.speed + track.time
            t_stop  = (ip[-1])/track.speed + track.time
        else:
            t_start = float("nan")
            t_stop = float("nan")

        if (self.MillipedeName + "Params" not in frame): 
            print("no good millipede found, evacuate")
            return False
        
        losses = frame[self.MillipedeName + "Params"]
        
        # exclude losses outside detector
        dist = []
        energy = []
        for loss in losses:
            if (loss.time > t_start) and (loss.time < t_stop):
                dist.append((loss.time - t_start)*track.speed)
                energy.append(loss.energy)

        dist = np.array(dist)
        energy = np.array(energy)

        if len(dist) < 1:
            print("no losses found within detector volume")
            return False
        
        length = (t_stop - t_start)*track.speed

        # compute quantiles of energy loss distribution
        quantiles1 = length - np.interp(np.linspace(0,1,self.nQuantiles),np.cumsum(energy)/np.sum(energy),dist)        
        quantiles2 = np.sum(energy) - np.interp(np.linspace(0,1,self.nQuantiles),np.cumsum(energy)/np.sum(energy),np.cumsum(energy))
        dedx = diff(quantiles2)/diff(quantiles1)

        # make features used for training [log10(contained energy),contained track length, recon zen, 0th quantile, 1st quantile, ...]
        features = [np.cos(reco_zen),length,np.sum(energy)] + quantiles1.tolist() + quantiles2.tolist() + dedx.tolist()

        # cleanup nans and infs since scikit-learn can't handle this
        if self.GetParameter("Cleanup"):
            if (np.isnan(features)|np.isinf(features)).sum() > 0:
                return False
        
        frame[self.GetParameter("FeaturesName")] = dataclasses.I3VectorDouble(features)

        # no targets created if using data
        if self.GetParameter("TargetName") is None:
            self.PushFrame(frame)
            return True

        # add target values to frame if this is MC; target is in-ice neutrino energy for starting tracks
        if self.isStartingTrack:
            nu_inice_energy = 0.
            if frame.Has("I3MCTree") or frame.Has("I3MCTree_preMuonProp"):
                # find in-ice neutrino
                if frame.Has("I3MCTree"):
                    tree = frame["I3MCTree"]
                else:
                    tree = frame["I3MCTree_preMuonProp"]

                nu = tree.primaries[0]
                child = tree.first_child(nu)
                while child.is_neutrino:
                    if tree.number_of_children(child) > 0:                        
                        nu = child
                        child = tree.first_child(nu)
                    else:
                        break
                
                nu_inice_energy = nu.energy

                # get muon energy and cascade energy if learning inelasticity
                if self.doInelasticity:
                    muon_energy = 0.
                    if abs(child.type) == 13:
                        muon_energy = child.energy
                    
                    had_energy = nu_inice_energy - muon_energy
                    targets = [np.log10(had_energy),np.log10(muon_energy),np.log10(frame["Deposited_Cascade_Energy"].value),np.log10(frame["Deposited_Muon_Energy"].value)]
                else:
                    targets = [np.log10(nu_inice_energy),np.log10(frame["Deposited_Cascade_Energy"].value+frame["Deposited_Muon_Energy"].value)]

        # otherwise target is most enegetic muon energy at point of closest approach to detector center
        else:
            muon_center_energy = 0.
            if frame.Has("MMCTrackList"):
                for mmc_track in frame["MMCTrackList"]:
                    if mmc_track.Ec > muon_center_energy:
                        muon_center_energy = mmc_track.Ec
            
            targets = [np.log10(muon_center_energy)]

        # cleanup nans and infs
        if self.GetParameter("Cleanup"):
            if (np.isnan(targets)|np.isinf(targets)).sum() > 0:
                return False
        
        frame[self.GetParameter("TargetName")] = dataclasses.I3VectorDouble(targets)

        self.PushFrame(frame)
        return True


"""
 predict output from an already trained random forest
"""
class RandomForestPredict(icetray.I3Module):
    def __init__(self,context):
        icetray.I3Module.__init__(self,context)
        self.AddOutBox("OutBox")
        self.AddParameter("FeaturesName", "Name of random forest features in frame", "RandomForestFeatures")
        self.AddParameter("OutputName", "Name of random forest output to put in frame", "RandomForestEnergyEstimates") 
        self.AddParameter("RandomForestPickle", "Location of pickled random forest regressor", "rf.pkl")

    def Configure(self):
        import joblib
        self.rf = joblib.load(self.GetParameter("RandomForestPickle"))

    def Geometry(self, frame):
        self.PushFrame(frame)
        pass

    def Calibration(self, frame):
        self.PushFrame(frame)
        pass

    def DetectorStatus(self, frame):
        self.PushFrame(frame)
        pass

    def Physics(self, frame):
        features = np.array(frame[self.GetParameter("FeaturesName")])

        # return nan if any feature is nan or inf
        if (np.isnan(features)|np.isinf(features)).sum() > 0:
            output = np.zeros(self.rf.n_outputs_)*np.nan
        else:
            if self.rf.n_outputs_>1:
                output = self.rf.predict(features[None,:])[0]
            else:
                output = self.rf.predict(features[None,:])
        
        output = 10**output
        frame[self.GetParameter("OutputName")] = dataclasses.I3MapStringDouble({
                                                                "Cascade Energy": output[0],
                                                                "Muon Energy": output[1],
                                                                "Neutrino Energy": output[0]+output[1],
                                                                "Cascade Deposited Energy": output[2],
                                                                "Muon Deposited Energy": output[3],
                                                                "Neutrino Deposited Energy": output[2]+output[3]        
                                                               })
        self.PushFrame(frame)
        return True
