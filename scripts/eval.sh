python evaluation.py \
    --type="SFUC_step30" \
    --style_content="" \
    --style_ttf_path="data_examples/test/SFUC/style" \
    --generated_image_path="outputs_Image/parallel+CBAM+ECA/SFUC_step30" \
    --content_path="data_examples/test/SFUC/unseen_240.json" \
    --style_path="data_examples/test/SFUC/styles.json" \
    --save_result_dir="outputs_metrics/parallel+CBAM+ECA" \
    --image_size=96 \
    --device="cuda:0"