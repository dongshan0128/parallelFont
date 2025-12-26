python infer_phase1.py \
  --ckpt_dir "outputs/the_forth_try/global_step_280000" \
  --config_path "outputs/the_forth_try/FontDiffuser_training_phase_1_config.yaml" \
  --content_image_path "data_examples/sampling/1166.png" \
  --style_image_path "data_examples/sampling/1_4_3_年_异体字.png" \
  --save_image \
  --save_image_dir "outputs/phase1_inference" \
  --sampler "dpm-solver++" \
  --device cuda:0 \
  --num_inference_steps 20 \