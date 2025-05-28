# 导入所需的库
from modelscope.msdatasets import MsDataset
import os
import pandas as pd

dataset_id = 'data'
dataset_dir = 'latex_ocr'
csv_path = 'latex_ocr_train.csv'

# 从modelscope下载COCO 2014图像描述数据集
ds = MsDataset.load(dataset_id)
print(len(ds))

# 创建保存图片的目录
os.makedirs(dataset_dir, exist_ok=True)

# 初始化存储图片路径和描述的列表
image_paths = []
texts = []
total = len(ds)
for i in range(total):
    # 获取每个样本的信息
    item = ds[i]
    text = item['text']
    image = item['image']

    # 保存图片并记录路径
    image_path = os.path.abspath(f'{dataset_dir}/{i}.jpg')
    image.convert("RGB").save(image_path)

    # 将路径和描述添加到列表中
    image_paths.append(image_path)
    texts.append(text)

    # 每处理50张图片打印一次进度
    if (i + 1) % 50 == 0:
        print(f'Processing {i + 1}/{total} images ({(i + 1) / total * 100:.1f}%)')

# 将图片路径和描述保存为CSV文件
df = pd.DataFrame({
    'image_path': image_paths,
    'text': texts,
})

# 将数据保存为CSV文件
df.to_csv(csv_path, index=False)

print(f'数据处理完成，共处理了{total}张图片')
