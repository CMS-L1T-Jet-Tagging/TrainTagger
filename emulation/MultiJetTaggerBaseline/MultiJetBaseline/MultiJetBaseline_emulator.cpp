#include "NN/MultiJetBaseline.h" //include of the top level of HLS model
#include "emulator.h" //include of emulator modeling
#include <any>
#include "ap_fixed.h"

using namespace MultiJetTagger_v1;

class MultiJetBaseline_emulator : public hls4mlEmulator::Model{
    private:
        input_t _input[N_INPUT_1_1*N_INPUT_2_1];
        layer25_t _layer25_out[N_LAYER_22];
        layer24_t _layer24_out[N_LAYER_20];
    public:


        virtual void prepare_input(std::any input)
        {
            //std::cout << "INPUTS: ";
            input_t* input_p = std::any_cast<input_t*>(input);
            for(int i = 0; i < N_INPUT_1_1*N_INPUT_2_1; ++i){
                _input[i] = std::any_cast<input_t>(input_p[i]);
                //std::cout << _input[i] << ", "; 
            }
            //std::cout << std::endl;
        }



        virtual void predict()
        {
            MultiJetBaseline(_input, _layer24_out, _layer25_out);
            //for (int i = 0; i < N_LAYER_20; ++i ){
                //std::cout << _layer24_out[i] << " | ";
            //}

            //std::cout << std::endl; 
        }

        virtual void read_result(std::any result)
        { 
            std::pair<std::array<layer25_t,N_LAYER_22>,std::array<layer24_t,N_LAYER_20>> *result_p = std::any_cast<std::pair<std::array<layer25_t,N_LAYER_22>,std::array<layer24_t,N_LAYER_20>>*>(result);
            //std::cout << "OUTPUTS REG: ";
            for (int i = 0; i < N_LAYER_22; ++i ){
                result_p->first[i] = _layer25_out[i];  
                //std::cout <<  _layer22_out[i] << ", ";
            }
            //std::cout << std::endl;
            //std::cout << "OUTPUTS CLASS: ";
            for (int i = 0; i < N_LAYER_20; ++i ){
                result_p->second[i] = _layer24_out[i];
                //std::cout <<  _layer24_out[i] << ", ";
            }
            //std::cout << std::endl;
            //std::cout << "===========" << std::endl;
        }

};

extern "C" hls4mlEmulator::Model* create_model()
{
    return new MultiJetBaseline_emulator;
}

extern "C" void destroy_model(hls4mlEmulator::Model* m)
{
    delete m;
}