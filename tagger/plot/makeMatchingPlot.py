from utils.imports import *
from utils.dataset import *
import argparse
from train.models import *
import tensorflow_model_optimization as tfmot
from array import array
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
import json
import glob
import pdb
import pandas
import numpy
from histbook import *


recoTauFile = "../recoTauNTuples/extendedAll200.root"
genTauFile = "../genTauNTuples/extendedAll200.root"

outFolder = "matchingPlots/extendedTRK/"
if not os.path.exists(outFolder):
    os.makedirs(outFolder)



# start from gen taus
filter = "/(jet)_(eta|phi|pt|pt_log|pt_raw|mass|energy|px|py|pz|bjetscore|tauscore|taupt|pt_corr|tauflav|muflav|elflav|taudecaymode|lepflav|taucharge|genmatch_pt|genmatch_eta|genmatch_phi|genmatch_mass|genmatch_hflav|genmatch_lep_vis_pt|genmatch_lep_pt|genmatch_pflav|npfcand|pfcand_pt|pfcand_pt_rel|pfcand_pt_rel_log|pfcand_px|pfcand_py|pfcand_pz|pfcand_pt_log|pfcand_eta|pfcand_phi|pfcand_mass|pfcand_energy|pfcand_energy_log|pfcand_puppiweight|pfcand_z0|pfcand_dxy|pfcand_dxy_custom|pfcand_id|pfcand_charge|pfcand_pperp_ratio|pfcand_ppara_ratio|pfcand_deta|pfcand_dphi|pfcand_etarel|pfcand_track_chi2|pfcand_track_chi2norm|pfcand_track_qual|pfcand_track_npar|pfcand_track_nstubs|pfcand_track_vx|pfcand_track_vy|pfcand_track_vz|pfcand_track_pterror|pfcand_cluster_hovere|pfcand_cluster_sigmarr|pfcand_cluster_abszbarycenter|pfcand_cluster_emet|pfcand_cluster_egvspion|pfcand_cluster_egvspu)/"
f = uproot.open(genTauFile)
data = f["gentauntuple/GenTaus"].arrays(
    # filter_name = filter, 
    how = "zip")
tau_ptmin =   (data['gentau_pt'] > 5.) & (np.abs(data['gentau_eta']) < 2.4)
data = data[tau_ptmin]

# data["hasTauMatch"] = data["recotau_match_dR"] < 0.4
# data["hasJetMatch"] = data["recojet_match_dR"] < 0.4
data["hasTauMatch"] = data["recotau_match_dR"] < 0.2
data["hasJetMatch"] = data["recojet_match_dR"] < 0.2


# efficiency vs pT
# x_bins_pt = np.array([10., 15., 20., 25., 30., 40., 50., 75., 100., 150., 200., 500., 1000.])
# x_bins_pt = np.array([0., 15., 20., 30., 40., 50., 75., 100., 125., 150., 175., 200., 300., 500., 750., 1000.])
# x_bins_pt = np.array([10., 15., 20., 30., 40., 50., 60., 70., 80., 90., 100., 125., 150., 175., 200., 250., 300., 400., 500., 750., 1000.])
x_bins_pt = np.array([5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 125., 150., 175., 200., 225., 250., 275., 300., 350., 400., 450., 500., 750.])

data_ = ak.to_pandas(data)

h_hasJetMatch = Hist(split("gentau_pt", x_bins_pt), cut("hasJetMatch"))
h_hasJetMatch.fill(data_)
h_hasTauMatch = Hist(split("gentau_pt", x_bins_pt), cut("hasTauMatch"))
h_hasTauMatch.fill(data_)

df_hasJetMatch = h_hasJetMatch.pandas("hasJetMatch", error="normal")
df_hasTauMatch = h_hasTauMatch.pandas("hasTauMatch", error="normal")

df_hasJetMatch["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_hasJetMatch.index]
df_hasTauMatch["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_hasTauMatch.index]

plt.cla()
# now scatter plot with error bars
ax = df_hasJetMatch.plot.line(x="midpoints", y="hasJetMatch", yerr="err(hasJetMatch)", label = "SCJet matching eff.", color="blue", linestyle='-')
df_hasTauMatch.plot.line(x="midpoints", y="hasTauMatch", yerr="err(hasTauMatch)", label = "NNPuppiTau matching eff.", ax = ax, color="orange", linestyle='--')
plt.xlabel(r'Gen. tau $p_T$ [GeV]')
plt.ylabel('Matching efficiency')
plt.ylim(0., 1.3)
plt.xlim(10., 750.)
# plt.semilogx()
plt.legend(prop={'size': 10})
plt.legend(loc='upper right')
# hep.cms.label("Private Work", data = False, com = 14)
hep.cms.label("Private Work", data = False, rlabel = "14 TeV (PU 200)")
plt.savefig(outFolder+"/matcheff_vs_pt"+".png")
plt.savefig(outFolder+"/matcheff_vs_pt"+".pdf")
plt.cla()

# efficiency vs pT 2
# x_bins_pt = np.array([10., 15., 20., 25., 30., 40., 50., 75., 100., 150., 200., 500., 1000.])
# x_bins_pt = np.array([0., 15., 20., 30., 40., 50., 75., 100., 125., 150., 175., 200., 300., 500., 750., 1000.])
# x_bins_pt = np.array([10., 15., 20., 30., 40., 50., 60., 70., 80., 90., 100., 125., 150., 175., 200., 250., 300., 400., 500., 750., 1000.])
x_bins_pt = np.array([5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100.])

data_ = ak.to_pandas(data)

h_hasJetMatch = Hist(split("gentau_pt", x_bins_pt), cut("hasJetMatch"))
h_hasJetMatch.fill(data_)
h_hasTauMatch = Hist(split("gentau_pt", x_bins_pt), cut("hasTauMatch"))
h_hasTauMatch.fill(data_)

df_hasJetMatch = h_hasJetMatch.pandas("hasJetMatch", error="normal")
df_hasTauMatch = h_hasTauMatch.pandas("hasTauMatch", error="normal")

df_hasJetMatch["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_hasJetMatch.index]
df_hasTauMatch["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_hasTauMatch.index]

plt.cla()
# now scatter plot with error bars
ax = df_hasJetMatch.plot.line(x="midpoints", y="hasJetMatch", yerr="err(hasJetMatch)", label = "SCJet matching eff.", color="blue", linestyle='-')
df_hasTauMatch.plot.line(x="midpoints", y="hasTauMatch", yerr="err(hasTauMatch)", label = "NNPuppiTau matching eff.", ax = ax, color="orange", linestyle='--')
plt.xlabel(r'Gen. tau $p_T$ [GeV]')
plt.ylabel('Matching efficiency')
plt.ylim(0., 1.3)
plt.xlim(5., 100.)
# plt.semilogx()
plt.legend(prop={'size': 10})
plt.legend(loc='upper right')
hep.cms.label("Private Work", data = False, rlabel = "14 TeV (PU 200)")
plt.savefig(outFolder+"/matcheff_vs_pt_low"+".png")
plt.savefig(outFolder+"/matcheff_vs_pt_low"+".pdf")
plt.cla()

# efficiency vs eta
# x_bins_eta = np.array([-2.4, -2.2, -2., -1.8, -1.6 , -1.4, -1.2, -1., -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4])
x_bins_eta = np.linspace(-2.4, 2.4, 48)

h_hasJetMatch_eta = Hist(split("gentau_eta ", x_bins_eta), cut("hasJetMatch"))
h_hasJetMatch_eta.fill(data_)
h_hasTauMatch_eta = Hist(split("gentau_eta", x_bins_eta), cut("hasTauMatch"))
h_hasTauMatch_eta.fill(data_)

df_hasJetMatch_eta = h_hasJetMatch_eta.pandas("hasJetMatch", error="normal")
df_hasTauMatch_eta = h_hasTauMatch_eta.pandas("hasTauMatch", error="normal")

df_hasJetMatch_eta["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_hasJetMatch_eta.index]
df_hasTauMatch_eta["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_hasTauMatch_eta.index]

plt.cla()
# now scatter plot with error bars
ax = df_hasJetMatch_eta.plot.line(x="midpoints", y="hasJetMatch", yerr="err(hasJetMatch)", label = "SCJet matching eff.", color="blue", linestyle='-')
df_hasTauMatch_eta.plot.line(x="midpoints", y="hasTauMatch", yerr="err(hasTauMatch)", label = "NNPuppiTau matching eff.", ax = ax, color="orange", linestyle='--')
plt.xlabel(r'Gen. tau $\eta$')
plt.ylabel('Matching efficiency')
plt.ylim(0., 1.3)
plt.xlim(-2.4, 2.4)
plt.legend(prop={'size': 10})
plt.legend(loc='upper right')
hep.cms.label("Private Work", data = False, rlabel = "14 TeV (PU 200)")
plt.savefig(outFolder+"/matcheff_vs_eta"+".png")
plt.savefig(outFolder+"/matcheff_vs_eta"+".pdf")
plt.cla()





# now reco taus to jets
    
filter = "/(jet)_(eta|phi|pt|pt_log|pt_raw|mass|energy|px|py|pz|bjetscore|tauscore|pt_corr|tauflav|muflav|elflav|taudecaymode|lepflav|taucharge|genmatch_pt|genmatch_eta|genmatch_phi|genmatch_mass|genmatch_hflav|genmatch_lep_vis_pt|genmatch_lep_pt|genmatch_pflav|npfcand|pfcand_pt|pfcand_pt_rel|pfcand_pt_rel_log|pfcand_px|pfcand_py|pfcand_pz|pfcand_pt_log|pfcand_eta|pfcand_phi|pfcand_mass|pfcand_energy|pfcand_energy_log|pfcand_puppiweight|pfcand_z0|pfcand_dxy|pfcand_dxy_custom|pfcand_id|pfcand_charge|pfcand_pperp_ratio|pfcand_ppara_ratio|pfcand_deta|pfcand_dphi|pfcand_etarel|pfcand_track_chi2|pfcand_track_chi2norm|pfcand_track_qual|pfcand_track_npar|pfcand_track_nstubs|pfcand_track_vx|pfcand_track_vy|pfcand_track_vz|pfcand_track_pterror|pfcand_cluster_hovere|pfcand_cluster_sigmarr|pfcand_cluster_abszbarycenter|pfcand_cluster_emet|pfcand_cluster_egvspion|pfcand_cluster_egvspu)/"
f = uproot.open(recoTauFile)
data = f["tauntuple/Taus"].arrays(
    # filter_name = filter, 
    how = "zip")
tau_ptmin =   (data['tau_pt'] > 5.) & (np.abs(data['tau_eta']) < 2.4)
data = data[tau_ptmin]

data_genmatched = data[data["tau_tauflav"] > 0]

data_genmatched["hasGenTauMatchAndJetMatch"] =  data_genmatched["tau_jetmatch_dR"] <= 0.4
data["hasJetMatch"] = data["tau_jetmatch_dR"] <= 0.4

# efficiency vs pT
x_bins_pt = np.array([5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 125., 150., 175., 200., 225., 250., 275., 300., 350., 400., 450., 500., 750., 1000.])

data_ = ak.to_pandas(data)
data_genmatched_ = ak.to_pandas(data_genmatched)

h_hasJetMatch = Hist(split("tau_pt", x_bins_pt), cut("hasJetMatch"))
h_hasJetMatch.fill(data_)
h_hasTauMatch = Hist(split("tau_pt", x_bins_pt), cut("hasGenTauMatchAndJetMatch"))
h_hasTauMatch.fill(data_genmatched_)

df_hasJetMatch = h_hasJetMatch.pandas("hasJetMatch", error="normal")
df_hasTauMatch = h_hasTauMatch.pandas("hasGenTauMatchAndJetMatch", error="normal")

df_hasJetMatch["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_hasJetMatch.index]
df_hasTauMatch["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_hasTauMatch.index]

plt.cla()
# now scatter plot with error bars
ax = df_hasJetMatch.plot.line(x="midpoints", y="hasJetMatch", yerr="err(hasJetMatch)", label = "All reco. taus to jets", color="blue", linestyle='-')
df_hasTauMatch.plot.line(x="midpoints", y="hasGenTauMatchAndJetMatch", yerr="err(hasGenTauMatchAndJetMatch)", label = "Gen. matched reco. taus to jets", ax = ax, color="orange", linestyle='--')
plt.xlabel(r'Gen. tau $p_T$ [GeV]')
plt.ylabel('Matching efficiency')
plt.ylim(0.7, 1.3)
plt.xlim(10., 1000.)
# plt.semilogx()
plt.legend(prop={'size': 10})
plt.legend(loc='upper right')
hep.cms.label("Private Work", data = False, rlabel = "14 TeV (PU 200)")
plt.savefig(outFolder+"/recotau_matcheff_vs_pt"+".png")
plt.savefig(outFolder+"/recotau_matcheff_vs_pt"+".pdf")
plt.cla()

# efficiency vs eta
x_bins_eta = np.linspace(-2.4, 2.4, 48)

data_ = ak.to_pandas(data)
data_genmatched_ = ak.to_pandas(data_genmatched)

h_hasJetMatch = Hist(split("tau_eta", x_bins_eta), cut("hasJetMatch"))
h_hasJetMatch.fill(data_)
h_hasTauMatch = Hist(split("tau_eta", x_bins_eta), cut("hasGenTauMatchAndJetMatch"))
h_hasTauMatch.fill(data_genmatched_)

df_hasJetMatch = h_hasJetMatch.pandas("hasJetMatch", error="normal")
df_hasTauMatch = h_hasTauMatch.pandas("hasGenTauMatchAndJetMatch", error="normal")

df_hasJetMatch["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_hasJetMatch.index]
df_hasTauMatch["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_hasTauMatch.index]

plt.cla()
# now scatter plot with error bars
ax = df_hasJetMatch.plot.line(x="midpoints", y="hasJetMatch", yerr="err(hasJetMatch)", label = "All reco taus to jets", color="blue", linestyle='-')
df_hasTauMatch.plot.line(x="midpoints", y="hasGenTauMatchAndJetMatch", yerr="err(hasGenTauMatchAndJetMatch)", label = "Gen matched taus to jets", ax = ax, color="orange", linestyle='--')
plt.xlabel(r'Gen. tau $\eta$')
plt.ylabel('Matching efficiency')
plt.ylim(0.7, 1.3)
plt.xlim(-2.4, 2.4)
# plt.semilogx()
plt.legend(prop={'size': 10})
plt.legend(loc='upper right')
hep.cms.label("Private Work", data = False, rlabel = "14 TeV (PU 200)")
plt.savefig(outFolder+"/recotau_matcheff_vs_eta"+".png")
plt.savefig(outFolder+"/recotau_matcheff_vs_eta"+".pdf")
plt.cla()