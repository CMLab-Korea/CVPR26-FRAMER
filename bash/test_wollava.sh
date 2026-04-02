CUDA_VISIBLE_DEVICES=0 python test/test_wollava.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
--transformer_model_name_or_path="preset/models/framer_d" \
--image_path preset/datasets/test_datasets \
--output_dir results/ \
--prompt_path preset/prompts \