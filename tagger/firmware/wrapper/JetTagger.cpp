#include "JetTagger.h"
#include "../JetTaggerNN/firmware/JetTaggerNN.h"
#include "data.h"
#include "hls_math.h"

input_t candidate_mass(Particle candidate) {
    
    input_t massCand = input_t(0.13); // pion mass

    if (candidate.hwId.bits== l1ct::ParticleID::PHOTON) {
      massCand = input_t(0.0);
    }
    else if (candidate.hwId.bits == l1ct::ParticleID::ELEPLUS || l1ct::ParticleID::ELEMINUS)
    {
      massCand = input_t(0.005);
    }
    else if (candidate.hwId.bits == l1ct::ParticleID::MUMINUS || l1ct::ParticleID::MUPLUS)
    {
      massCand = input_t(0.105);
    }
    else if (candidate.hwId.bits == l1ct::ParticleID::HADZERO)
    {
      massCand = input_t(0.5); //K mass
    }

    return massCand;
}

template<unsigned int N_PARTICLES, unsigned int N_FEATURES>
void prepare_inputs(const Particle candidates[N_PARTICLES], input_t (&tagger_ingput)[N_PARTICLES*N_FEATURES]) {

  //Based on latest PUPPI specifications
  #pragma HLS INLINE OFF

  //Loop through each particle and its features
  for (int i_particle = 0; i_particle < N_PARTICLES; i_particle++) {

      Particle candidate = candidates[i_particle];
      int isFilled = candidate.hwPt == 0 ? 0:1;

      if (isFilled == 0) {

        // If the particle is not filled, then we can skip the rest of the features
        for (int i_feature = 0; i_feature < N_FEATURES; i_feature++) {
          tagger_ingput[i_particle*N_FEATURES + i_feature] = 0;
        }
        continue;

      }
      else{
        // The order is based on the emulator here
        // https://github.com/CMS-L1T-Jet-Tagging/TrainTagger/blob/e590960ddd5a516fa33e869017d6b1ccbba4ede1/emulation/CMSSW/MultiJetId.cc#L35-L55

        // TODO: Check if these are the same objects in the emulator
        tagger_ingput[i_particle*N_FEATURES + 0] = input_t(candidate.hwPt); //pT as a fraction of jet pT
        tagger_ingput[i_particle*N_FEATURES + 1] = input_t(candidate.hwEta); //dEta from jet axis
        tagger_ingput[i_particle*N_FEATURES + 2] = input_t(candidate.hwPhi); //dPhi from jet axis
        tagger_ingput[i_particle*N_FEATURES + 3] = input_t(log(candidate.hwPt)); //log(pT)
        tagger_ingput[i_particle*N_FEATURES + 4] = input_t(candidate.hwEta * l1ct::Scales::ETAPHI_LSB); //Eta
        tagger_ingput[i_particle*N_FEATURES + 5] = input_t(candidate.hwPhi * l1ct::Scales::ETAPHI_LSB); //Phi
        tagger_ingput[i_particle*N_FEATURES + 6] = candidate_mass(candidate); //Candidate mass

        //One hot encoding for the particle type
        tagger_ingput[i_particle*N_FEATURES + 7] = input_t(candidate.hwId.bits == l1ct::ParticleID::PHOTON ? 1 : 0); //photon
        tagger_ingput[i_particle*N_FEATURES + 8] = input_t(candidate.hwId.bits == l1ct::ParticleID::ELEPLUS ? 1 : 0); //e+
        tagger_ingput[i_particle*N_FEATURES + 9] = input_t(candidate.hwId.bits == l1ct::ParticleID::ELEMINUS ? 1 : 0); //e-
        tagger_ingput[i_particle*N_FEATURES + 10] = input_t(candidate.hwId.bits == l1ct::ParticleID::MUPLUS ? 1 : 0); //mu+
        tagger_ingput[i_particle*N_FEATURES + 11] = input_t(candidate.hwId.bits == l1ct::ParticleID::MUMINUS ? 1 : 0); //mu-
        tagger_ingput[i_particle*N_FEATURES + 12] = input_t(candidate.hwId.bits == l1ct::ParticleID::HADZERO ? 1 : 0); //Neutral hadron
        tagger_ingput[i_particle*N_FEATURES + 13] = input_t(candidate.hwId.bits == l1ct::ParticleID::HADPLUS ? 1 : 0); //pi+
        tagger_ingput[i_particle*N_FEATURES + 14] = input_t(candidate.hwId.bits == l1ct::ParticleID::HADMINUS ? 1 : 0); //pi-

        //Other variables
        tagger_ingput[i_particle*N_FEATURES + 15] = input_t(candidate.hwZ0()*l1ct::Scales::Z0_LSB); //z0
        tagger_ingput[i_particle*N_FEATURES + 16] = input_t(candidate.hwDxy()*l1ct::Scales::DXY_LSB); //DXY
        tagger_ingput[i_particle*N_FEATURES + 17] = input_t(isFilled); //isfilled
        tagger_ingput[i_particle*N_FEATURES + 18] = input_t(candidate.hwPuppiW()); //PuppiWeight
        tagger_ingput[i_particle*N_FEATURES + 19] = input_t(candidate.hwEmID()); //emID
        tagger_ingput[i_particle*N_FEATURES + 20] = input_t(candidate.hwTkQuality()); //quality
      }

  }

} 

void JetTagger(const Particle input[N_TAGGER_PARTICLES], // Input 
                Jet out_scores[1], pt_reg_t pt_correction[1], // Output
                const JetCtrlToken& token_d, JetCtrlToken& token_q) { // Tokens

  #pragma HLS ARRAY_PARTITION variable=input complete
  #pragma HLS ARRAY_PARTITION variable=out_scores complete
  #pragma HLS ARRAY_PARTITION variable=pt_correction complete

  #pragma HLS INTERFACE mode=ap_none port=out_scores,pt_correction,token_q
  #pragma HLS PIPELINE II=1

  // Pass on the token
  token_q = token_d;
  
  // Take the inputs and aggregate it for the model
  input_t  tagger_input[N_TAGGER_PARTICLES*N_FEATURES_PARTICLES];
  nn_tagger_score_t nn_output_scores[N_TAGGER_SCORES];
  pt_reg_t nn_output_pt_reg[N_PT_HEAD];
  #pragma HLS ARRAY_PARTITION variable=tagger_input complete
  #pragma HLS ARRAY_PARTITION variable=nn_output_scores complete
  #pragma HLS ARRAY_PARTITION variable=nn_output_pt_reg complete

  prepare_inputs<N_TAGGER_PARTICLES, N_FEATURES_PARTICLES>(input, tagger_input);
  JetTaggerNN(tagger_input, nn_output_scores, nn_output_pt_reg); // Run it through the network
  
  //clear the output and assign the score
  clear(out_scores[0]);
  out_scores[0].hwBkgScore = tagger_score_t(nn_output_scores[0]); //background
  out_scores[0].hwBtagScore = tagger_score_t(nn_output_scores[1]); // btag 
  out_scores[0].hwTauPScore = tagger_score_t(nn_output_scores[2]); // Tau Plus
  out_scores[0].hwTauMScore = tagger_score_t(nn_output_scores[3]); // Tau Minus
  out_scores[0].hwGluonScore = tagger_score_t(nn_output_scores[4]); // Gluon
  out_scores[0].hwCharmScore = tagger_score_t(nn_output_scores[5]); // Charm
  out_scores[0].hwMuonScore = tagger_score_t(nn_output_scores[6]); // Muon
  out_scores[0].hwElectronScore = tagger_score_t(nn_output_scores[7]); // Electron
  pt_correction[0] = nn_output_pt_reg[0]; //pt correction
}