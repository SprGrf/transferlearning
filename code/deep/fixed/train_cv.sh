target=(MHEALTH HHAR DSA PAMAP2 selfBACK GOTOV C24)
for target_index in "${target[@]}"    
do
    python -u train.py --N_WORKERS 1 --data_dir ../../../../LIMU-BERT-Public/dataset/ --task cross_people --test_envs 1 --dataset ${target_index} --algorithm Fixed --mixupalpha 0.1 --alpha 0.5 --mixup_ld_margin 10 --top_k 5 --output ./results/cv_${target_index} --class_balanced 1
done