target=(MHEALTH HHAR DSA PAMAP2 selfBACK GOTOV C24)
for target_index in "${target[@]}" 
do
    python -u evaluate.py --N_WORKERS 1 --data_dir ./dataset/ --task cross_people --test_envs 1 --dataset ${target_index} --algorithm Fixed --mixupalpha 0.1 --alpha 0.5 --mixup_ld_margin 10 --top_k 5 --output ./results/cv_${target_index} --class_balanced 1 --mode cv
    python -u evaluate.py --N_WORKERS 1 --data_dir ./dataset/ --task cross_people --test_envs 1 --dataset ${target_index} --algorithm Fixed --mixupalpha 0.1 --alpha 0.5 --mixup_ld_margin 10 --top_k 5 --output ./results/d2d_${target_index} --class_balanced 1 --mode d2d_test
done