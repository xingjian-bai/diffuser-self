# accelerate launch train_unconditional.py \
#   --dataset_name="huggan/flowers-102-categories" \
#   --resolution=64 --center_crop --random_flip \
#   --output_dir="ddpm-ema-flowers-64" \
#   --train_batch_size=16 \
#   --num_epochs=100 \
#   --gradient_accumulation_steps=1 \
#   --use_ema \
#   --learning_rate=1e-4 \
#   --lr_warmup_steps=500 \
#   --mixed_precision=no \
#   --push_to_hub


accelerate launch train_unconditional.py \
  --model_type="transformer" \
  --dataset_name="huggan/flowers-102-categories" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="transformer_not_work" \
  --train_batch_size=32 \
  --num_epochs=500 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=3e-4 \
  --lr_warmup_steps=50 \
  --mixed_precision=no \
  --save_images_epochs=2 \
  --save_model_epochs=100 \
  --eval_batch_size=16 \
  --dataset_dir="./../../../flowers102/" \
  --num_attention_heads=16 \
  --num_layers=28 \
  --in_channels=4 \
  --out_channels=4 \
  --attention_head_dim=72 \
  --sample_size=8 \
  

  