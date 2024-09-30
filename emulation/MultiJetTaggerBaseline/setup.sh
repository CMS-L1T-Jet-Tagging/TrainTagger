mkdir -p MultiJetBaseline/NN
SRC=MultiJetBaseline/firmware

cp -r $SRC/defines.h   MultiJetBaseline/NN/
cp    $SRC/myproject.cpp   MultiJetBaseline/NN/MultiJetBaseline.cpp
cp    $SRC/myproject.h   MultiJetBaseline/NN/MultiJetBaseline.h
cp -r $SRC/nnet_utils   MultiJetBaseline/NN/
cp    $SRC/parameters.h   MultiJetBaseline/NN/
cp -r $SRC/weights    MultiJetBaseline/NN/
rm MultiJetBaseline/NN/weights/*.txt

# change the way the weights are read
find ./MultiJetBaseline/NN \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/#ifndef __SYNTHESIS__/#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__/'
# rename project (could be done in hlsm4l as well)
find ./MultiJetBaseline/NN \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/myproject/MultiJetBaseline/g'
find ./MultiJetBaseline/NN \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/MYPROJECT_H_/MULTIJETBaseline_H_/g'