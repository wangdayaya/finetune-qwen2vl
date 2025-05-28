
from modelscope import snapshot_download
# 指定下载目录
download_path = 'D:/'

# 下载入模型
# model_dir = snapshot_download('Qwen/Qwen-14B-Chat', cache_dir=download_path)
#模型下载
# model_dir = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct',  cache_dir=download_path)
#模型下载
model_dir = snapshot_download('Qwen/Qwen2-VL-7B-Instruct',  cache_dir=download_path)