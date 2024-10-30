#Configuration for data sets creation
FILTER_PATTERN = "/(jet)_(reject|eta|eta_phys|phi|phi_phys|pt|pt_phys|pt_raw|bjetscore|tauscore|taupt|pt_corr|tauflav|muflav|elflav|taudecaymode|lepflav|taucharge|genmatch_pt|genmatch_eta|genmatch_phi|genmatch_mass|genmatch_hflav|genmatch_lep_vis_pt|genmatch_lep_pt|genmatch_pflav|npfcand|pfcand_pt|pfcand_pt_rel|pfcand_pt_rel_log|pfcand_pt_log|pfcand_eta|pfcand_phi|pfcand_puppiweight|pfcand_emid|pfcand_pt_rel_phys|pfcand_pt_phys|pfcand_eta_phys|pfcand_phi_phys|pfcand_dphi_phys|pfcand_deta_phys|pfcand_quality|pfcand_tkquality|pfcand_z0|pfcand_dxy|pfcand_dxy_custom|pfcand_dxy_phys|pfcand_dxy_physSquared|pfcand_id|pfcand_charge|pfcand_pperp_ratio|pfcand_ppara_ratio|pfcand_deta|pfcand_dphi|pfcand_etarel|pfcand_track_valid|pfcand_track_rinv|pfcand_track_phizero|pfcand_track_tanl|pfcand_track_z0|pfcand_track_d0|pfcand_track_chi2rphi|pfcand_track_chi2rz|pfcand_track_bendchi2|pfcand_track_hitpattern|pfcand_track_mvaquality|pfcand_track_mvaother|pfcand_track_chi2|pfcand_track_chi2norm|pfcand_track_qual|pfcand_track_npar|pfcand_track_nstubs|pfcand_track_vx|pfcand_track_vy|pfcand_track_vz|pfcand_track_pterror|pfcand_cluster_hovere|pfcand_cluster_sigmarr|pfcand_cluster_abszbarycenter|pfcand_cluster_emet|pfcand_cluster_egvspion|pfcand_cluster_egvspu|pfcand_isPhoton|pfcand_isElectronPlus|pfcand_isElectronMinus|pfcand_isMuonPlus|pfcand_isMuonMinus|pfcand_isNeutralHadron|pfcand_isChargedHadronPlus|pfcand_isChargedHadronMinus|pfcand_isfilled|pfcand_energy|pfcand_mass)/"
N_PARTICLES = 16 #Number of particle constituents used in a jet for tagging
INPUT_TAG = "baseline_hardware_inputs" #Defined in pfcand_fields.yml