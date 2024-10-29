#include "L1Trigger/Phase2L1ParticleFlow/interface/MultiJetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include <cmath>

MultiJetId::MultiJetId(const std::shared_ptr<hls4mlEmulator::Model> model,
                       int iNParticles)
    : modelRef_(model) {
  NNvectorVar_.clear();
  fNParticles_ = iNParticles;

  fPt_ = std::make_unique<float[]>(fNParticles_);
  fEta_ = std::make_unique<float[]>(fNParticles_);
  fPhi_ = std::make_unique<float[]>(fNParticles_);
  fId_ = std::make_unique<float[]>(fNParticles_);
  fCharge_ = std::make_unique<int[]>(fNParticles_);
  fDZ_ = std::make_unique<float[]>(fNParticles_);
  fDX_ = std::make_unique<float[]>(fNParticles_);
  fDY_ = std::make_unique<float[]>(fNParticles_);

}


void MultiJetId::setNNVectorVar() {
  NNvectorVar_.clear();
  for (int i0 = 0; i0 < fNParticles_; i0++) {
    NNvectorVar_.push_back(fPt_.get()[i0]); //pT as a fraction of jet pT
    NNvectorVar_.push_back(fEta_.get()[i0]);  //dEta from jet axis
    NNvectorVar_.push_back(fPhi_.get()[i0]);  //dPhi from jet axis
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Photon);  // Photon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Electron && fCharge_.get()[i0] > 0);       // Positron
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Electron && fCharge_.get()[i0] < 0);       // Electron
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Muon && fCharge_.get()[i0] > 0);           // Anti-muon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Muon && fCharge_.get()[i0] < 0);           // Muon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::NeutralHadron);                            // Neutral Had
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::ChargedHadron && fCharge_.get()[i0] > 0);  // Anti-Pion
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::ChargedHadron && fCharge_.get()[i0] < 0);  // Pion    
  }
}

std::vector<ap_fixed<20, 9, AP_RND, AP_SAT>> MultiJetId::EvaluateNNFixed() {
  ap_fixed<20, 9, AP_RND, AP_SAT> modelInput[176] = {};   // Do something
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
    fPt_.get()[i0] = 0;
    fEta_.get()[i0] = 0;
    fPhi_.get()[i0] = 0;
    fId_.get()[i0] = 0;
    fCharge_.get()[i0] = 0;
    fDZ_.get()[i0] = 0;
    fDX_.get()[i0] = 0;
    fDY_.get()[i0] = 0;
  }
  auto iParts = iJet.constituents();
  std::sort(iParts.begin(), iParts.end(), [](edm::Ptr<l1t::PFCandidate> i, edm::Ptr<l1t::PFCandidate> j) {
    return (i->pt() > j->pt());
  });
  float jetpt = useRawPt ? iJet.rawPt() : iJet.pt();
  for (unsigned int i0 = 0; i0 < iParts.size(); i0++) {
    if (i0 >= (unsigned int)fNParticles_)
      break;
    fPt_.get()[i0] = iParts[i0]->pt() / jetpt;
    fEta_.get()[i0] = iParts[i0]->eta() - iJet.eta();
    fPhi_.get()[i0] = deltaPhi(iParts[i0]->phi(), iJet.phi());
    fId_.get()[i0] = iParts[i0]->id();
    fCharge_.get()[i0] = iParts[i0]->charge();
    if (iParts[i0]->pfTrack().isNonnull()) {
      fDX_.get()[i0] = iParts[i0]->pfTrack()->vx();
      fDY_.get()[i0] = iParts[i0]->pfTrack()->vy();
      fDZ_.get()[i0] = iParts[i0]->pfTrack()->vz() - vz;
    }
  }
  setNNVectorVar();
  return EvaluateNNFixed();
}
