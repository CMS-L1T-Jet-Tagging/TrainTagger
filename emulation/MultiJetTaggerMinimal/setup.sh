mkdir -p MultiJetMinimal_test/NN
SRC=OutputSynthesis/regression/Training_2024_22_08_vTEST/firmware/

cp -r $SRC/defines.h   MultiJetMinimal_test/NN/
cp    $SRC/myproject.cpp   MultiJetMinimal_test/NN/MultiJetMinimal_test.cpp
cp    $SRC/myproject.h   MultiJetMinimal_test/NN/MultiJetMinimal_test.h
cp -r $SRC/nnet_utils   MultiJetMinimal_test/NN/
cp    $SRC/parameters.h   MultiJetMinimal_test/NN/
cp -r $SRC/weights    MultiJetMinimal_test/NN/
rm MultiJetMinimal_test/NN/weights/*.txt

# change the way the weights are read
find ./MultiJetMinimal_test/NN \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/#ifndef __SYNTHESIS__/#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__/'
# rename project (could be done in hlsm4l as well)
find ./MultiJetMinimal_test/NN \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/myproject/MultiJetMinimal_test/g'
find ./MultiJetMinimal_test/NN \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/MYPROJECT_H_/MULTIJETMINIMAL_TEST_H_/g'