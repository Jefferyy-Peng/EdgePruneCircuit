EDGE_SPARSITIES=(0.96 0.98 1.05)

for i in "${!EDGE_SPARSITIES[@]}"; do

EDGE_SPARSITY=${EDGE_SPARSITIES[i]}
NODE_SPARSITY=0.72
ELR=0.8
LLR=0.8
RELR=0.8
RLLR=0.8
TOTAL=3000
WARMUP=2500

EXTRA="--disable_node_loss"
TAG="wo_node_loss"
ABLATE_MODE="mean"

# Uncomment this if you want to run with node loss
# EXTRA=""
# TAG="w_node_loss"

train_split="train" # "train_400", "train_100k"
N_TRAIN=1000000 # Set to a large value so all of the (200 / 400 / 100000) examples are used
N_VAL=200 # The val split size

# You can wrap the following in an sbatch script if you use SLURM
# Activate your environment etc

# If you want to always keep embedding nodes, remove the --with_embedding_nodes flag
# That flag, when set, also models masks over the embedding nodes

accelerate launch src/prune/clip_vit_imagenet.py \
    --report_to wandb \
    --do_train \
    --do_eval \
    --dataset_path ./data/datasets/gt/ \
    --train_split $train_split \
    --max_seq_length 64 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 16 \
    --edge_learning_rate $ELR \
    --layer_learning_rate $LLR \
    --reg_edge_learning_rate $RELR \
    --reg_layer_learning_rate $RLLR \
    --max_steps $TOTAL \
    --warmup_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 64 \
    --save_steps 64 \
    --logging_steps 8 \
    --save_total_limit 1 \
    --start_edge_sparsity 0.00 \
    --target_edge_sparsity $EDGE_SPARSITY \
    --start_layer_sparsity 0.00 \
    --target_layer_sparsity $NODE_SPARSITY \
    --num_sparsity_warmup_steps $WARMUP \
    --max_train_samples $N_TRAIN \
    --max_eval_samples $N_VAL \
    --output_dir ./data/runs/${ABLATE_MODE}_ablate-target_loss-clip-imagenet-${TAG}-elr${ELR}-llr${LLR}-relr${RELR}-rllr${RLLR}-es${EDGE_SPARSITY}-ns${NODE_SPARSITY}-t${TOTAL}/ \
    --remove_unused_columns false \
    --dataloader_num_workers 0 \
    --warmup_type linear \
    --with_embedding_nodes \
    --corr_mode $ABLATE_MODE \
    --ddp_find_unused_parameters False \
    $EXTRA

done