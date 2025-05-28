import swanlab
import torch
from datasets import Dataset
from modelscope import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
from swanlab.integration.transformers import SwanLabCallback
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor, Qwen2_5_VLForConditionalGeneration,
)

prompt = "你是一个LaText OCR助手,目标是读取用户输入的照片，转换成LaTex公式。"
local_model_path = "Qwen2-VL-2B-Instruct"  # 需要自己在本地下载好
train_dataset_json_path = "latex_ocr_train.json"
val_dataset_json_path = "latex_ocr_val.json"
output_dir = "output/Qwen2-VL-2B-LatexOCR"
MAX_LENGTH = 8192


def process_func(example):
    """
    将数据集进行预处理
    """
    conversation = example["conversations"]
    image_file_path = conversation[0]["value"]
    output_content = conversation[1]["value"]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_file_path}",
                    "resized_height": 500,
                    "resized_width": 100,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list,为了方便拼接
    instruction = inputs
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])  # [h*w/14/14, 图像特征大小（大小不固定，但是后续会进行线性变化转为和大模型隐向量大小一样）]
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  #  因为图片的 t=time 都是 1 ，所以由（t,h,w) 变换为（h,w）
    return {
            "input_ids": input_ids,   # 其中的 <|image_pad|> 对应的 token 为图像占位符，个数为被图片 patch_num / 4 ，因为后面的图像特征会进行线性压缩，邻近的四个 patch 特征压缩到一个
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": inputs['pixel_values'],
            "image_grid_thw": inputs['image_grid_thw']
            }


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,  clean_up_tokenization_spaces=False)
    return output_text[0]


# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(local_model_path)
origin_model = Qwen2VLForConditionalGeneration.from_pretrained(local_model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
origin_model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
# 5.8G - 1.3G
# 处理数据集：读取json文件
train_ds = Dataset.from_json(train_dataset_json_path)
train_dataset = train_ds.map(process_func)

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
train_peft_model = get_peft_model(origin_model, config)

# 配置训练参数
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen2-VL-2B-ft-latexocr",
    experiment_name="1k_data",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct",
        "dataset": "https://modelscope.cn/datasets/AI-ModelScope/LaTeX_OCR/summary",
        "train_dataset_json_path": train_dataset_json_path,
        "val_dataset_json_path": val_dataset_json_path,
        "output_dir": output_dir,
        "prompt": prompt,
        "train_data_number": len(train_ds),
        "token_max_length": MAX_LENGTH,
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

# 配置Trainer
trainer = Trainer(
    model=train_peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

# 开启模型训练
trainer.train()  # 7.7G-5.8
# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()
# {'train_runtime': 385.66, 'train_samples_per_second': 6.202, 'train_steps_per_second': 0.384, 'train_loss': 0.7554919067266825, 'epoch': 1.98}