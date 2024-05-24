import os
import git
from transformers import AutoProcessor, AutoModelForCTC

# Lấy đường dẫn của file code hiện tại
current_file_path = os.path.abspath(__file__) if '__file__' in globals() else os.path.abspath(os.getcwd())

# Xây dựng đường dẫn tương đối cho cache_dir
cache_dir = os.path.join(os.path.dirname(current_file_path), "models", "wav2vec2", "pinyin")

processor = AutoProcessor.from_pretrained("DuyTa/ZH_pinyn")
model = AutoModelForCTC.from_pretrained("DuyTa/ZH_pinyn", cache_dir=cache_dir)



# Xây dựng đường dẫn tương đối cho cache_dir
cache_dir_hanzi = os.path.join(os.path.dirname(current_file_path), "models", "wav2vec2", "hanzi")

processor = AutoProcessor.from_pretrained("wbbbbb/wav2vec2-large-chinese-zh-cn")
model = AutoModelForCTC.from_pretrained("wbbbbb/wav2vec2-large-chinese-zh-cn", cache_dir=cache_dir_hanzi)


repo_url = "https://huggingface.co/DuyTa/Whisper_ZH"


destination_folder = os.path.join(os.path.dirname(current_file_path), "models", "whisper")

git.Repo.clone_from(repo_url, destination_folder)
