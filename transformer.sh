accelerate launch train_unconditional.py \
  --model_type="transformer" \
  --dataset_name="huggan/flowers-102-categories" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="transformer_feb_22" --dataset_dir="./flowers102/" \
  --train_batch_size=16 --eval_batch_size=16 \
  --num_epochs=500 \
  --gradient_accumulation_steps=1 --use_ema --lr_warmup_steps=50 --mixed_precision=no \
  --learning_rate=1e-4 \
  --save_images_epochs=5 --save_model_epochs=100 \
  --num_attention_heads=32 --num_layers=16 --attention_head_dim=72 \
  --in_channels=3 --out_channels=3 \
  --patch_size=1 --sample_size=64 \
  --use_wandb
#   --vae_model="stabilityai/sd-vae-ft-ema"
  

  