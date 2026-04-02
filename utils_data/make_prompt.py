import torch
from PIL import Image
import os
from tqdm import tqdm
import re
import sys
sys.path.append(os.getcwd())
from llava.llm_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, default='preset/datasets/train_datasets/training_for', help='the dataset you want to tag.') # 
parser.add_argument("--save_dir", type=str, default='preset/datasets/train_datasets/training_for', help='the dataset you want to tag.') # 
parser.add_argument("--stop_num", type=int, default=-1)
parser.add_argument("--start_num", type=int, default=0)
args = parser.parse_args()

def remove_focus_sentences(text):
    # 使用正则表达式按照 . ? ! 分割，并且保留分隔符本身
    # re.split(pattern, string) 如果 pattern 中带有捕获组()，分隔符也会保留在结果列表中
    prohibited_words = ['focus', 'focal', 'prominent', 'close-up', 'black and white', 'blur', 'depth', 'dense', 'locate', 'position']
    parts = re.split(r'([.?!])', text)
    
    filtered_sentences = []
    i = 0
    while i < len(parts):
        # sentence 可能是句子主体，punctuation 是该句子结尾的标点
        sentence = parts[i]
        punctuation = parts[i+1] if (i+1 < len(parts)) else ''

        # 组合为完整句子，避免漏掉结尾标点
        full_sentence = sentence + punctuation
        
        full_sentence_lower = full_sentence.lower()
        skip = False
        for word in prohibited_words:
            if word.lower() in full_sentence_lower:
                skip = True
                break
        
        # 如果该句子不包含任何禁用词，则保留
        if not skip:
            filtered_sentences.append(full_sentence)
        
        # 跳过已经处理的句子和标点
        i += 2
    
    # 根据需要选择如何重新拼接；这里去掉多余空格并直接拼接
    return "".join(filtered_sentences).strip()

@torch.no_grad()
def process_llava(
    input_image):
    llama_prompt = llava_agent.gen_image_caption([input_image])[0]
    llama_prompt = remove_focus_sentences(llama_prompt)
    return llama_prompt

def PrintInfo(x):
    if not isinstance(x,list):
        x=[x]
    for i in x:
        print('shape : {} ; dtype : {} ; max : {} ; min : {}'.format(i.shape,i.dtype,i.max(),i.min())  )

img_folder = args.img_dir
prompt_save_folder = args.save_dir
os.makedirs(prompt_save_folder, exist_ok=True)
img_name_list = os.listdir(img_folder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device, load_8bit=True, load_4bit=False)

for img_name in tqdm(img_name_list[args.start_num:args.stop_num]):
    if os.path.exists(os.path.join(prompt_save_folder, img_name.replace('png', 'txt'))):
        continue
    img_path = os.path.join(img_folder, img_name)
    img = Image.open(img_path).convert('RGB')
    prompt = process_llava(img)
    with open(os.path.join(prompt_save_folder, img_name.replace('png', 'txt')), 'w', encoding="utf-8") as f:
        f.write(prompt)