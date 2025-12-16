python infer_phase1.py \
  --ckpt_dir "outputs/FontDiffuser/global_step_240000" \
  --content "data_examples/train/ContentImage/俺.jpg" \
  --style "data_examples/train/TargetImage/FZBaiZYHTJW/卑.jpg" \
  --output "outputs/FontDiffuser/phase1.png" \
  --steps 20 \
  --seed 123 \
  --device cuda:0