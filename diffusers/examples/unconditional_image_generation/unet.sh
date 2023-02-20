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


accelerate launch train_uncond.py \
  --model_type="unet" \
  --dataset_name="huggan/flowers-102-categories" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="init_baseline_test" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=50 \
  --mixed_precision=no \
  --save_images_epochs=5 \
  --save_model_epochs=100 \
  --eval_batch_size=256 \
  --push_to_hub \
  

  