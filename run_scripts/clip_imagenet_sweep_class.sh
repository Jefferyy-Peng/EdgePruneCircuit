#TARGET_CLASSES=("n02002075" 0 127 128 407 436 468 511 609 627 656 661 751 817)
TARGET_CLASSES=("all")
CKPT_IDS=(70 10 20 480)
for i in "${!TARGET_CLASSES[@]}"; do
TARGET_CLASS=${TARGET_CLASSES[i]}

for CKPT_ID in "${CKPT_IDS[@]}"; do
EDGE_SPARSITY=1.05
NODE_SPARSITY=0.1
ELR=0.9
LLR=0.9
RELR=0.9
RLLR=0.9
TOTAL=960
WARMUP=800
LR_WARMUP=70
T_BATCHSIZE=8
ALPHA=3.0

LOSS_TYPE="kl"
EXTRA="--disable_node_loss"
TAG="wo_node_loss"
ABLATE_MODE="ood_failure"
INCLUDE_QKV=False
FT_METHOD="IN21k-ERM-WaterBird"
DATASET="waterbirds"
OOD_DATASET="v2"
#OOD_DATASET=""

# Uncomment this if you want to run with node loss
# EXTRA=""
# TAG="w_node_loss"

EMBEDDING="--with_embedding_nodes"
EMB_TAG="w_embedding"

# Uncomment this if you want to run without embedding nodes
EMBEDDING=""
EMB_TAG="wo_embedding"

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
    --per_device_train_batch_size $T_BATCHSIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 16 \
    --edge_learning_rate $ELR \
    --layer_learning_rate $LLR \
    --reg_edge_learning_rate $RELR \
    --reg_layer_learning_rate $RLLR \
    --max_steps $TOTAL \
    --warmup_steps $LR_WARMUP \
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
    --output_dir ./data/runs/${ABLATE_MODE}_ablate-qkv_${INCLUDE_QKV}-${FT_METHOD}_${CKPT_ID}-class-${TARGET_CLASS}-${LOSS_TYPE}_loss-${DATASET}-${TAG}-${EMB_TAG}-elr${ELR}-llr${LLR}-relr${RELR}-rllr${RLLR}-lrw${LR_WARMUP}-sw${WARMUP}-tbs${T_BATCHSIZE}-es${EDGE_SPARSITY}-ns${NODE_SPARSITY}-alpha${ALPHA}-t${TOTAL}/ \
    --remove_unused_columns false \
    --dataloader_num_workers 0 \
    --warmup_type linear \
    $EMBEDDING \
    --corr_mode $ABLATE_MODE \
    --ddp_find_unused_parameters False \
    --include_qkv=$INCLUDE_QKV \
    --ft_method=$FT_METHOD \
    --target_class=$TARGET_CLASS \
    --loss_type=$LOSS_TYPE \
    --dataset=$DATASET \
    --ood_dataset=$OOD_DATASET \
    --ckpt_id=$CKPT_ID \
    --alpha=$ALPHA \
    $EXTRA
done
done