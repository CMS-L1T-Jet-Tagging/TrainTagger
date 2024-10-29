#ifndef SCJ_ALGO_DATA_H
#define SCJ_ALGO_DATA_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "DataFormats/L1TParticleFlow/interface/puppi.h"
#include "DataFormats/L1TParticleFlow/interface/jets.h"

typedef l1ct::pt_t pt_t;
typedef l1ct::glbeta_t etaphi_t;     // Type for eta & phi
typedef ap_int<13> detaphi_t;        // Type for deta & dphi
typedef ap_fixed<18,23> detaphi2_t;  // Type for deta^2 & dphi^2
typedef ap_fixed<22,22> pt_etaphi_t; // Type for product of pt with deta & dphi
typedef l1ct::PuppiObj Particle;
typedef l1ct::Jet Jet;

//For the tagger
typedef ap_ufixed<8,0,AP_RND,AP_SAT> tagger_score_t; //This does not include 1, however 1 is rounded to the maximum positive number so still safe
typedef ap_fixed<16,6> nn_tagger_score_t;
typedef ap_fixed<16,6> pt_reg_t;

// constants for the axis update
typedef ap_ufixed<18,-2> inv_pt_t;
static constexpr int N_table_inv_pt = 1024;

static const detaphi_t TWOPI = detaphi_t(l1ct::Scales::INTPHI_TWOPI);
static const detaphi_t PI = detaphi_t(l1ct::Scales::INTPHI_PI);
//static const detaphi_t RCONE = 0.4 / l1ct::Scales::ETAPHI_LSB;
//static const detaphi2_t R2CONE = 0.4 * 0.4 / l1ct::Scales::ETAPHI_LSB / l1ct::Scales::ETAPHI_LSB;
static const detaphi2_t R2CONE4 = 0.4 * 0.4 / l1ct::Scales::ETAPHI_LSB / l1ct::Scales::ETAPHI_LSB;
static const detaphi2_t R2CONE8 = 0.8 * 0.8 / l1ct::Scales::ETAPHI_LSB / l1ct::Scales::ETAPHI_LSB;

static const etaphi_t FIDUCIAL_ETA_PHI = 5.11 / l1ct::Scales::ETAPHI_LSB;
static const pt_t JET_PT_CUT = 0.25;

// Class for the data type sent between the JetLoop & JetCompute IPs
// Instead of the particle (eta, phi), we send (pt * deta, pt * dphi)
// ready for accumulation in JetCompute so that nothing is recomputed
class PartialParticle {
public:
    pt_t hwPt;
    pt_etaphi_t hwEta; 
    pt_etaphi_t hwPhi; 

    bool operator >= (const PartialParticle &b){
        return hwPt >= b.hwPt;
    }
};

inline void clear(PartialParticle &p) {
    p.hwPt = 0;
    p.hwEta = 0;
    p.hwPhi = 0;
}

struct JetCtrlToken{
    ap_uint<3> n_event;
    ap_uint<5> n_iter;
    ap_uint<3> n_wait;
    bool largeRadius;
    ap_uint<1> start_of_orbit;
    bool dataValid;
    bool frameValid;
};

inline void clear(JetCtrlToken &p) {
    p.n_event = 0;
    p.n_iter = 0;
    p.n_wait = 0;
    p.largeRadius = false;
    p.start_of_orbit = 0;
    p.dataValid = false;
    p.frameValid = false;
}

#define NPARTICLES 128
#define NJETS 16
#define NJETS_TO_GT 12
#define CTRL_REPLICATION 8
#define N_TAGGER_PARTICLES  16 //How many particles in a jet to use for the tagger
#define N_FEATURES_PARTICLES  21 // How many features are used in a particle
#define N_TAGGER_SCORES 8
#define N_PT_HEAD 1
#define NOUTPUT 1 //Number of jet ID output

#endif