#include "L1Trigger/Phase2L1ParticleFlow/interface/MultiJetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include <cmath>

MultiJetId::MultiJetId(const std::shared_ptr<hls4mlEmulator::Model> model,
                       int iNParticles)
    : modelRef_(model) {
  NNvectorVar_.clear();
  fNParticles_ = iNParticles;

  fPt_rel_phys_ = std::make_unique<float[]>(fNParticles_);
  fDEta_phys_ = std::make_unique<float[]>(fNParticles_);
  fDPhi_phys_ = std::make_unique<float[]>(fNParticles_);
  fPt_log_ = std::make_unique<float[]>(fNParticles_);
  fEta_phys_ = std::make_unique<float[]>(fNParticles_);
  fPhi_phys_ = std::make_unique<float[]>(fNParticles_);
  fMass_ = std::make_unique<float[]>(fNParticles_);
  fIs_photon_ = std::make_unique<int[]>(fNParticles_);
  fIs_electron_plus_ = std::make_unique<int[]>(fNParticles_);
  fIs_electron_minus_ = std::make_unique<int[]>(fNParticles_);
  fIs_muon_plus_ = std::make_unique<int[]>(fNParticles_);
  fIs_muon_minus_ = std::make_unique<int[]>(fNParticles_);
  fIs_neutral_hadron_ = std::make_unique<int[]>(fNParticles_);
  fIs_charged_hadron_plus_ = std::make_unique<int[]>(fNParticles_);
  fIs_charged_hadron_minus_ = std::make_unique<int[]>(fNParticles_);
  fZ0_ = std::make_unique<float[]>(fNParticles_);
  fDxy_phys_ = std::make_unique<float[]>(fNParticles_);
  fIs_filled_ = std::make_unique<int[]>(fNParticles_);
  fPuppi_weight_ = std::make_unique<float[]>(fNParticles_);
  fEmID_ = std::make_unique<int[]>(fNParticles_);
  fQuality_ = std::make_unique<int[]>(fNParticles_);

}


void MultiJetId::setNNVectorVar() {
  NNvectorVar_.clear();
  for (int i0 = 0; i0 < fNParticles_; i0++) {
    NNvectorVar_.push_back(fPt_rel_phys_.get()[i0]); //pT as a fraction of jet pT
    NNvectorVar_.push_back(fDEta_phys_.get()[i0]);  //dEta from jet axis
    NNvectorVar_.push_back(fDPhi_phys_.get()[i0]);  //dPhi from jet axis
    NNvectorVar_.push_back(fPt_log_.get()[i0]);  // pt log 
    NNvectorVar_.push_back(fEta_phys_.get()[i0]);  // eta
    NNvectorVar_.push_back(fPhi_phys_.get()[i0]);  // phi
    NNvectorVar_.push_back(fMass_.get()[i0]);  // Mass
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Photon);  // Photon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Electron && fCharge_.get()[i0] > 0);       // Positron
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Electron && fCharge_.get()[i0] < 0);       // Electron
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Muon && fCharge_.get()[i0] > 0);           // Anti-muon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Muon && fCharge_.get()[i0] < 0);           // Muon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::NeutralHadron);                            // Neutral Had
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::ChargedHadron && fCharge_.get()[i0] > 0);  // Anti-Pion
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::ChargedHadron && fCharge_.get()[i0] < 0);  // Pion 
    NNvectorVar_.push_back((fZ0_.get())[i0]);  // z0  
    NNvectorVar_.push_back((fDxy_phys_.get())[i0]);  // dxy
    NNvectorVar_.push_back((fIs_filled_.get())[i0]);  // isfilled
    NNvectorVar_.push_back((fPuppi_weight_.get())[i0]);  // puppi weight
    NNvectorVar_.push_back((fEmID_.get())[i0]);  // emID
    NNvectorVar_.push_back((fQuality_.get())[i0]);  // quality
  }
}

std::vector<ap_fixed<20, 9, AP_RND, AP_SAT>> MultiJetId::EvaluateNNFixed() {
  ap_fixed<20, 9, AP_RND, AP_SAT> modelInput[336] = {};   // Do something
  for (unsigned int i = 0; i < NNvectorVar_.size(); i++) {
    modelInput[i] = NNvectorVar_[i];
  }
  
  ap_fixed<20, 9, AP_RND, AP_SAT> modelResult[9] = {-1,-1,-1,-1,-1,-1,-1,-1,-1};

  modelRef_->prepare_input(modelInput);
  modelRef_->predict();
  modelRef_->read_result(modelResult);
  std::vector<ap_fixed<20, 9, AP_RND, AP_SAT>> modelResult_;
  for (unsigned int i = 0; i < 9; i++) {
    modelResult_.push_back(modelResult[i]);
  }
  // std::cout << "ID 1:" << modelResult[0]
  //           << "ID 2:" << modelResult[1] 
  //           << "ID 3:" << modelResult[2] 
  //           << "ID 4:" << modelResult[3] 
  //           << "ID 5:" << modelResult[4] 
  //           << "ID 6:" << modelResult[5] 
  //           << "ID 7:" << modelResult[6] 
  //           << "ID 8:" << modelResult[7] 
  //           << "PT Regression:" << modelResult[8] 
  //           << std::endl;
  return modelResult_;
}  //end EvaluateNNFixed


std::vector<ap_fixed<20, 9, AP_RND, AP_SAT>> MultiJetId::computeFixed(const l1t::PFJet &iJet, float vz, bool useRawPt) {
  for (int i0 = 0; i0 < fNParticles_; i0++) {
    fPt_rel_phys_.get()[i0] = 0;
    fDEta_phys_.get()[i0] = 0;
    fDPhi_phys_.get()[i0] = 0;
    fPt_log_.get()[i0] = 0;
    fEta_phys_.get()[i0] = 0;
    fPhi_phys_.get()[i0] = 0;
    fMass_.get()[i0] = 0;
    fIs_photon_.get()[i0] = 0;
    fIs_electron_plus_.get()[i0] = 0;
    fIs_electron_minus_.get()[i0] = 0;
    fIs_muon_plus_.get()[i0] = 0;
    fIs_neutral_hadron_.get()[i0] = 0;
    fIs_charged_hadron_plus_.get()[i0] = 0;
    fIs_charged_hadron_minus_.get()[i0] = 0;
    fZ0_.get()[i0] = 0;
    fDxy_phys_.get()[i0] = 0;
    fIs_filled_.get()[i0] = 0;
    fPuppi_weight_.get()[i0] = 0;
    fEmID_.get()[i0] = 0;
    fQuality_.get()[i0] = 0;

  }
  auto iParts = iJet.constituents();
  std::sort(iParts.begin(), iParts.end(), [](edm::Ptr<l1t::PFCandidate> i, edm::Ptr<l1t::PFCandidate> j) {
    return (i->pt() > j->pt());
  });
  float jetpt = useRawPt ? iJet.rawPt() : iJet.pt();
  for (unsigned int i0 = 0; i0 < iParts.size(); i0++) {
    if (i0 >= (unsigned int)fNParticles_)
      break;
    fPt_rel_phys_.get()[i0] = iParts[i0]->pt() / jetpt;
    fDEta_phys_.get()[i0] = iParts[i0]->eta() - iJet.eta();
    fDPhi_phys_.get()[i0] = deltaPhi(iParts[i0]->phi(), iJet.phi());

    fPt_log_.get()[i0] = std::log(iParts[i0]->pt());

    fEta_phys_.get()[i0] = iParts[i0]->eta();
    fPhi_phys_.get()[i0] = iParts[i0]->phi();

    float massCand = 0.13f;
    if (abs(iParts[i0]->charge())) {
      if ((iParts[i0]->id() == l1t::PFCandidate::Muon)) {
        massCand = 0.105;
      } else if ((iParts[i0]->id() == l1t::PFCandidate::Electron)) {
        massCand = 0.005;
      } 
    } else {
      massCand = iParts[i0]->id() == l1t::PFCandidate::Photon ? 0.0 : 0.5;
    }

    fMass_.get()[i0] = massCand;
    fZ0_.get()[i0] = iParts[i0]->z0();
    fDxy_phys_.get()[i0] = iParts[i0]->dxy();
    fIs_filled_.get()[i0] = iParts[i0]->pfTrack().isNonnull();
    fPuppi_weight_.get()[i0] = iParts[i0]->puppiWeight();
    fEmID_.get()[i0] = iParts[i0]->hwEmID();
    fQuality_.get()[i0] = iParts[i0]->hwTkQuality();

    fCharge_.get()[i0] = iParts[i0]->charge();
    fId_.get()[i0] = iParts[i0]->id();

  }
  setNNVectorVar();
  return EvaluateNNFixed();
}
