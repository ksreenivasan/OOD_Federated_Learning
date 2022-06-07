python simulated_averaging.py \
--lr 0.02 \
--gamma 0.998 \
--num_nets 200 \
--fl_round 1500 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg9 \
--fl_mode fixed-pool \
--attacker_pool_size 100 \
--defense_method flame \
--attack_method pgd \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.025 \
--eps 2 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type southwest \
--norm_bound 2 \
--attacker_percent 0.25 \
--instance benchmark-FLAME-25-percent \
--wandb_group VGG9-CIFAR10-SOUTHWEST-PGD \
--device=cuda