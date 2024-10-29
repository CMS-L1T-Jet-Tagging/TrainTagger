#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOWS_MULTIJETID_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOWS_MULTIJETID_H

#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"

//HLS4ML compiled emulator modeling
#include <string>
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

class MultiJetId {
public:
  MultiJetId(const std::shared_ptr<hls4mlEmulator::Model> model,
             int iNParticles);
  ~MultiJetId() = default;

  void setNNVectorVar();
  std::vector<ap_fixed<20, 9, AP_RND, AP_SAT>> EvaluateNNFixed();
  std::vector<ap_fixed<20, 9, AP_RND, AP_SAT>> computeFixed(const l1t::PFJet &iJet, float vz, bool useRawPt);

private:
  std::vector<float> NNvectorVar_;
  int fNParticles_;
  unique_ptr<float[]> fPt_;
  unique_ptr<float[]> fEta_;
  unique_ptr<float[]> fPhi_;
  unique_ptr<float[]> fId_;
  unique_ptr<int[]> fCharge_;
  unique_ptr<float[]> fDZ_;
  unique_ptr<float[]> fDX_;
  unique_ptr<float[]> fDY_;
  std::shared_ptr<hls4mlEmulator::Model> modelRef_;
};
#endif
