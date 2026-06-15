./setup.sh


#for p in 0.01 0.1 1 10 50; do
#     base="output/ds_hgq_300000ebops_${p}_data"

#     python tagger/train/train.py -p $p -e 300000 -y tagger/model/configs/baseline_HGQ2.yaml -o $base &> ${base%_data}_training_out.txt
#     python tagger/train/train.py -p $p -e 300000 -y tagger/model/configs/baseline_HGQ2.yaml -o $base --plot-basic > ${base%_data}_testing_out.txt
#     rm -r output/ds_hgq_300000ebops_${p}_data/testing_data
#done

#for p in 0.01 0.1 1 10 50; do
#     base="output/mlpm_hgq_300000ebops_${p}_data"

#     python tagger/train/train.py -p $p -e 300000 -y tagger/model/configs/MLPmixer_HGQ2.yaml -o $base > ${base%_data}_training_out.txt
#     python tagger/train/train.py -p $p -e 300000 -y tagger/model/configs/MLPmixer_HGQ2.yaml -o $base --plot-basic > ${base%_data}_testing_out.txt
#     rm -r output/mlpm_hgq_300000ebops_${p}_data/testing_data
#done

#for p in 0.01 0.1 1 10 20 30 40 50; do
#     base="output/jl_hgq_300000ebops_${p}_data"

#     python tagger/train/train.py -p $p -e 300000 -y tagger/model/configs/JEDIlinear_HGQ2.yaml -o $base > ${base%_data}_training_out.txt
#     python tagger/train/train.py -p $p -e 300000 -y tagger/model/configs/JEDIlinear_HGQ2.yaml -o $base --plot-basic > ${base%_data}_testing_out.txt
#     rm -r output/jl_hgq_300000ebops_${p}_data/testing_data
#done

# for p in 0.01 0.1 1 10 20 30 40; do
#      base="output/linf_hgq_300000ebops_${p}_data"

#      python tagger/train/train.py -p $p -e 300000 -y tagger/model/configs/linformer_HGQ2.yaml -o $base > ${base%_data}_training_out.txt
#      python tagger/train/train.py -p $p -e 300000 -y tagger/model/configs/linformer_HGQ2.yaml -o $base --plot-basic > ${base%_data}_testing_out.txt
#      rm -r output/linf_hgq_300000ebops_${p}_data/testing_data
# done

# for p in 0.01 0.1 1 10 50 60 80; do
#      base="output/ds_bigger_hgq_300000ebops_${p}_data"

#      python tagger/train/train.py -p $p -e 300000 -y tagger/model/configs/baseline_larger_HGQ2.yaml -o $base > ${base%_data}_training_out.txt
#      python tagger/train/train.py -p $p -e 300000 -y tagger/model/configs/baseline_larger_HGQ2.yaml -o $base --plot-basic > ${base%_data}_testing_out.txt
#      rm -r output/ds_bigger_hgq_300000ebops_${p}_data/testing_data
# done



# for p in 50000 100000 500000 1000000 5000000 ; do
#     base="output/mlpm_hgq_${p}_ebops_10_data"

#     python tagger/train/train.py -p 10 -e ${p} -y tagger/model/configs/MLPmixer_HGQ2.yaml -o $base > ${base%_data}_training_out.txt
#     python tagger/train/train.py -p 10 -e ${p} -y tagger/model/configs/MLPmixer_HGQ2.yaml -o $base --plot-basic > ${base%_data}_testing_out.txt
#     rm -r output/mlpm_hgq_${p}_ebops_10_data/testing_data
# done

# for p in 10000 50000 100000 500000 1000000 5000000 ; do
#     base="output/ds_very_small_hgq_${p}_ebops_10_data"

#     python tagger/train/train.py -p 10 -e ${p} -y tagger/model/configs/baseline_larger_HGQ2.yaml -o $base > ${base%_data}_training_out.txt
#     python tagger/train/train.py -p 10 -e ${p} -y tagger/model/configs/baseline_larger_HGQ2.yaml -o $base --plot-basic > ${base%_data}_testing_out.txt
#     rm -r output/ds_very_small_hgq_${p}_ebops_10_data/testing_data
# done

# for p in 10000 50000 100000 500000 1000000 5000000 ; do
#     base="output/linf_hgq_${p}_ebops_10_data"

#     python tagger/train/train.py -p 10 -e ${p} -y tagger/model/configs/linformer_HGQ2.yaml -o $base > ${base%_data}_training_out.txt
#     python tagger/train/train.py -p 10 -e ${p} -y tagger/model/configs/linformer_HGQ2.yaml -o $base --plot-basic > ${base%_data}_testing_out.txt
#     rm -r output/mlpm_hgq_${p}_ebops_10_data/testing_data
# done

# for p in 10000 50000 100000 500000 1000000 5000000 ; do
#     base="output/ds_hgq_${p}_ebops_10_data"

#     python tagger/train/train.py -p 10 -e ${p} -y tagger/model/configs/baseline_HGQ2.yaml -o $base > ${base%_data}_training_out.txt
#     python tagger/train/train.py -p 10 -e ${p} -y tagger/model/configs/baseline_HGQ2.yaml -o $base --plot-basic > ${base%_data}_testing_out.txt
#     rm -r output/mlpm_hgq_${p}_ebops_10_data/testing_data
# done

for p in 50000 100000 500000 1000000 5000000 ; do
    base="output/ds_bigger_hgq_${p}_ebops_10_data"
    python tagger/train/train.py -p 100 -e ${p} -y tagger/model/configs/baseline_larger_HGQ2.yaml -o $base > ${base%_data}_training_out.txt
    python tagger/train/train.py -p 100 -e ${p} -y tagger/model/configs/baseline_larger_HGQ2.yaml -o $base --plot-basic > ${base%_data}_testing_out.txt
    rm -r output/ds_bigger_hgq_${p}_ebops_10_data/testing_data
    alkaid convert output/ds_bigger_hgq_${p}_ebops_10_data/model/saved_model.keras output/ds_bigger_hgq_${p}_ebops_10_data/model/rtl_output --flavor vhdl --clock-period 2.7 
    alkaid report output/ds_bigger_hgq_${p}_ebops_10_data/model/rtl_output -f > ${base%_data}_fw_out.txt
done
