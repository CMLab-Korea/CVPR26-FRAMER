import torch
import os
import glob
from tqdm import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import sys
sys.path.append(os.getcwd())

# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

# Copied from dreambooth sd3 example
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# Copied from dreambooth sd3 example
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


# Copied from dreambooth sd3 example
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default='preset/datasets/train_datasets/training', help='the dataset you want to tag.') # 
parser.add_argument("--start_gpu", type=int, default=0, help='if you have 5 GPUs, you can set it to 0/1/2/3/4 when using different GPU for parallel processing. It will save your time.') 
parser.add_argument("--all_gpu", type=int, default=1, help='if you set --start_gpu max to 5, please set it to 5') 
parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
parser.add_argument(
    "--variant",
    type=str,
    default=None,
    help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
)
parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
parser.add_argument("--end_num", type=int, default=1)
parser.add_argument("--start_num", type=int, default=1)
args = parser.parse_args()

tag_path = os.path.join(args.root_path, 'prompt')
prompt_embeds_path = os.path.join(args.root_path, 'prompt_embeds')
pooled_prompt_embeds_path = os.path.join(args.root_path, 'pooled_prompt_embeds')
os.makedirs(prompt_embeds_path, exist_ok=True)
os.makedirs(pooled_prompt_embeds_path, exist_ok=True)

tag_lists = glob.glob(os.path.join(tag_path, '*.txt'))
print(f'There are {len(tag_lists)} tags' )

# Load the tokenizer
tokenizer_one = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer",
    revision=args.revision,
)
tokenizer_two = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer_2",
    revision=args.revision,
)
tokenizer_three = T5TokenizerFast.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer_3",
    revision=args.revision,
)

# import correct text encoder class
text_encoder_cls_one = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision
)
text_encoder_cls_two = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
)
text_encoder_cls_three = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
)

text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args
    )

text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
text_encoder_three.requires_grad_(False)
text_encoder_one.eval()
text_encoder_two.eval()
text_encoder_three.eval()
text_encoder_one.to('cuda', dtype=torch.float16)
text_encoder_two.to('cuda', dtype=torch.float16)
text_encoder_three.to('cuda', dtype=torch.float16)

tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

def compute_text_embeddings(prompt, text_encoders, tokenizers):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, args.max_sequence_length
        )
        prompt_embeds = prompt_embeds
        pooled_prompt_embeds = pooled_prompt_embeds
    return prompt_embeds, pooled_prompt_embeds

# start_num = args.start_gpu * len(tag_lists)//args.all_gpu
# end_num = (args.start_gpu+1) * len(tag_lists)//args.all_gpu
start_num = args.start_num
end_num = args.end_num

print(f'===== process [{start_num}   {end_num}] =====')
batch_size = 1
total_tags = len(tag_lists)
for i in tqdm(range(0, total_tags, batch_size)):
    # 获取当前批次的标签路径
    batch_tag_paths = tag_lists[i:i + batch_size]
    batch_prompts = []
    valid_tag_paths = []
    save_paths = []
    for tag_path in batch_tag_paths:
        basename = os.path.basename(tag_path).split('.')[0]
        prompt_embeds_save_path = os.path.join(prompt_embeds_path, f'{basename}.pt')
        pooled_prompt_embeds_save_path = os.path.join(pooled_prompt_embeds_path, f'{basename}.pt')

        # 如果嵌入已存在，跳过
        if os.path.exists(prompt_embeds_save_path) and os.path.exists(pooled_prompt_embeds_save_path):
            continue

        with open(tag_path, "r") as f:
            prompt = f.read()
            batch_prompts.append(prompt)
            valid_tag_paths.append(tag_path)
            save_paths.append((prompt_embeds_save_path, pooled_prompt_embeds_save_path))
    if not batch_prompts:  # 如果当前批次没有需要处理的标签
        continue
    batch_prompt_embeddings = compute_text_embeddings(batch_prompts, text_encoders, tokenizers)
    for prompt, (prompt_embeds_save_path, pooled_prompt_embeds_save_path), p_embeds, p_pooled_embeds in zip(batch_prompts, save_paths, *batch_prompt_embeddings):
        torch.save(p_embeds.clone(), prompt_embeds_save_path)
        torch.save(p_pooled_embeds.clone(), pooled_prompt_embeds_save_path)
        print(f"Saved embeddings for {os.path.basename(prompt_embeds_save_path)}")
        












        
