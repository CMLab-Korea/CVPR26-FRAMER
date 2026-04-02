CUDA_VISIBLE_DEVICES=0,1 python test/test_wllava.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
--transformer_model_name_or_path="preset/models/framer_d" \
--image_path preset/datasets/test_datasets \
--output_dir results/ \