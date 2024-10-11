mkdir -p MultiJetBaseline/NN
SRC=MultiJetBaseline/firmware

cp -r $SRC/defines.h   MultiJetBaseline/NN/
sed -i "s/#include <cstdio>/#include <cstdio>\n\nnamespace MultiJetTagger_v1 {/" MultiJetBaseline/NN/defines.h
echo '}' >> MultiJetBaseline/NN/defines.h

cp    $SRC/myproject.cpp   MultiJetBaseline/NN/MultiJetBaseline.cpp
sed -i "s/void myproject(/namespace MultiJetTagger_v1 {\nvoid myproject(/" MultiJetBaseline/NN/MultiJetBaseline.cpp
echo '}' >> MultiJetBaseline/NN/MultiJetBaseline.cpp

cp    $SRC/myproject.h   MultiJetBaseline/NN/MultiJetBaseline.h
sed -i "s/void myproject(/namespace MultiJetTagger_v1 {\nvoid myproject(/" MultiJetBaseline/NN/MultiJetBaseline.h
sed -i "s/#endif/}\n\n#endif/" MultiJetBaseline/NN/MultiJetBaseline.h

cp -r $SRC/nnet_utils   MultiJetBaseline/NN/

cp    $SRC/parameters.h   MultiJetBaseline/NN/
sed -i "s,// hls-fpga-machine-learning insert layer-config,namespace MultiJetTagger_v1 {\n// hls-fpga-machine-learning insert layer-config," MultiJetBaseline/NN/parameters.h
sed -i "s/#endif/}\n\n#endif/" MultiJetBaseline/NN/parameters.h

cp -r $SRC/weights    MultiJetBaseline/NN/
for filename in MultiJetBaseline/NN/weights/*.h; do
    sed -i '8 a namespace MultiJetTagger_v1 {' $filename
    echo '}' >> $filename
done

rm MultiJetBaseline/NN/weights/*.txt

# change the way the weights are read
find ./MultiJetBaseline/NN \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/#ifndef __SYNTHESIS__/#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__/'
# rename project (could be done in hlsm4l as well)
find ./MultiJetBaseline/NN \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/myproject/MultiJetBaseline/g'
find ./MultiJetBaseline/NN \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/MYPROJECT_H_/MULTIJETBaseline_H_/g'