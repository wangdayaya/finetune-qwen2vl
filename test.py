import json
import os

import torch
from modelscope import AutoTokenizer
from peft import LoraConfig, TaskType, PeftModel
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

output_dir = "output/Qwen2-VL-2B-LatexOCR"
prompt = "你是一个LaText OCR助手,目标是读取用户输入的照片，转换成LaTex公式。"
local_model_path = "Qwen2-VL-2B-Instruct"
val_dataset_json_path = "latex_ocr_val.json"
MAX_LENGTH = 8192

# 读取测试数据
with open(val_dataset_json_path, "r") as f:
    test_dataset = json.load(f)


def predict(type, model):
    test_image_list = []
    for item in test_dataset:
        image_file_path = item["conversations"][0]["value"]
        label = item["conversations"][1]["value"]
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file_path,
                    "resized_height": 100,
                    "resized_width": 500,
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ]}]
        # 准备推理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")

        # 生成输出
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        response = output_text[0]
        print(f"{type}  ----->  predict  ----->  {response}，gt  ----->  {label}，{response == label}")
    return test_image_list


# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(local_model_path)
origin_model = Qwen2VLForConditionalGeneration.from_pretrained(local_model_path, device_map="auto",
                                                               torch_dtype=torch.bfloat16, trust_remote_code=True, )


# ----------原模型---------
print("origin_model")
test_image_list = predict(type, origin_model)
# ------------lora------------

# 配置测试参数
# origin_model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
# val_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     inference_mode=True,  # 训练模式
#     r=64,  # Lora 秩
#     lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
#     lora_dropout=0.05,  # Dropout 比例
#     bias="none",
# )
# load_model_path = f"{output_dir}/checkpoint-{max([int(d.split('-')[-1]) for d in os.listdir(output_dir) if d.startswith('checkpoint-')])}"
# print(f"load_model_path: {load_model_path}")
# val_peft_model = PeftModel.from_pretrained(origin_model, model_id=load_model_path, config=val_config)
# type = "lora"
print("lora ")
lora_model_path = r"/output/Qwen2-VL-2B-LatexOCR/checkpoint-148"
model = PeftModel.from_pretrained(origin_model, lora_model_path)
test_image_list = predict(type, origin_model)

# origin_model  ----->  predict  ----->  \[(x^3 - x^2 - x)(2x - 7)\]，gt  ----->  \left( x ^ { 3 } - x ^ { 2 } - x \right) \left( 2 x - 7 \right)，False
# origin_model  ----->  predict  ----->  The LaTeX code for the given expression is:
#
# ```latex
# x_1 - x_2 + y_1 - y_2 + z_1 - z_2
# ```，gt  ----->  x _ { 1 } - x _ { 2 } + y _ { 1 } - y _ { 2 } + z _ { 1 } - z _ { 2 }，False
# origin_model  ----->  predict  ----->  The LaTeX code for the given handwritten expression is:
#
# ```latex
# a + b + c + d + e
# ```，gt  ----->  a + b + c + d + e，False
# origin_model  ----->  predict  ----->  $$x^{i} + 2 j \times k^{3} - 2 \frac{j}{i}$$，gt  ----->  x ^ { i + 2 j \times k ^ { 3 } - 2 \frac { j } { i } }，False
# load_model_path: ./output/Qwen2-VL-2B-LatexOCR/checkpoint-148
# lora  ----->  predict  ----->  \left( x ^ { 3 } - x ^ { 2 } - x \right) \left( 2 x - 7 \right)，gt  ----->  \left( x ^ { 3 } - x ^ { 2 } - x \right) \left( 2 x - 7 \right)，True
# lora  ----->  predict  ----->  x _ { 1 } - x _ { 2 } + y _ { 1 } - y _ { 2 } + z _ { 1 } - z _ { 2 }，gt  ----->  x _ { 1 } - x _ { 2 } + y _ { 1 } - y _ { 2 } + z _ { 1 } - z _ { 2 }，True
# lora  ----->  predict  ----->  a + b + c + d + e，gt  ----->  a + b + c + d + e，True
# lora  ----->  predict  ----->  x ^ { i + 2 j \times k ^ { 3 } - 2 \frac { j } { i } }，gt  ----->  x ^ { i + 2 j \times k ^ { 3 } - 2 \frac { j } { i } }，True


# https://www.latexlive.com/