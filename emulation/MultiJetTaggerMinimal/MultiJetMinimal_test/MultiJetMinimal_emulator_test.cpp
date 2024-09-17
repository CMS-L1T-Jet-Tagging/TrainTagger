#include "NN/MultiJetMinimal_test.h" //include of the top level of HLS model
#include "emulator.h" //include of emulator modeling
#include <any>
#include "ap_fixed.h"

class MultiJetMinimal_emulator_test : public hls4mlEmulator::Model{
    private:
        input_t _input[N_INPUT_1_1*N_INPUT_2_1];
        layer24_t _layer24_out[N_LAYER_20];
        layer25_t _layer25_out[N_LAYER_22];
    public:
        virtual void prepare_input(std::any input)
        {
            input_t* input_p = std::any_cast<input_t*>(input);
            for(int i = 0; i < N_INPUT_1_1*N_INPUT_2_1; ++i)
                _input[i] = std::any_cast<input_t>(input_p[i]);
        }

        virtual void predict()
        {
            MultiJetMinimal_test(_input, _layer24_out, _layer25_out);
        }

        virtual void read_result(std::any result)
        { 
            layer24_t *result_p1 = std::any_cast<layer24_t*>(result);
            layer25_t *result_p2 = std::any_cast<layer25_t*>(result);
            for (int i = 0; i < N_LAYER_20; ++i)  // CHECK 20 is hardcoded
                result_p1[i] = _layer24_out[i];
            for (int i = 0; i < N_LAYER_22; ++i)  // CHECK 22 is hardcoded
                result_p2[i] = _layer25_out[i];
        }

};

extern "C" hls4mlEmulator::Model* create_model()
{
    return new MultiJetMinimal_emulator_test;
}

extern "C" void destroy_model(hls4mlEmulator::Model* m)
{
    delete m;
}