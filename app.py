import os

####################### Whisper ##########################
from transformers import AutoModelForSpeechSeq2Seq,AutoModelForCTC, AutoProcessor, pipeline
import torch
from operator import itemgetter


current_file_path = os.path.abspath(__file__) if '__file__' in globals() else os.path.abspath(os.getcwd())

w2v_pinyin_dir = os.path.join(os.path.dirname(current_file_path), "models", "wav2vec2", "pinyin")
w2v_hanzi_dir = os.path.join(os.path.dirname(current_file_path), "models", "wav2vec2", "hanzi")
whisper_pinyin_dir = os.path.join(os.path.dirname(current_file_path), "models", "whisper", "pinyin")
whisper_hanzi_dir = os.path.join(os.path.dirname(current_file_path), "models", "whisper", "chinese/checkpoint-7322")

class Whisper_Model:
    def __init__(self, model_type) :
        self.model_type = model_type
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if self.model_type == "Whisper_Pinyin" :
         
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                whisper_pinyin_dir, torch_dtype=self.torch_dtype
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large",cache_dir=whisper_pinyin_dir)
        elif self.model_type == "Whisper_Hanzi" :
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                whisper_hanzi_dir, torch_dtype=self.torch_dtype
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large",cache_dir=whisper_hanzi_dir)
            
    def infer(self,audiopath:str) -> str:
        pipe = pipeline(
        "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=20,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
            generate_kwargs={"language": "en","task": "transcribe"}
        )
        prediction = pipe(audiopath)
        result_string = " ".join(map(itemgetter('text'), prediction["chunks"]))
        return result_string


########################### Wav2vec2 ############################ 
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from operator import itemgetter

class Wav2vec2_Model:
    def __init__(self, model_type) :
        self.model_type = model_type
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if self.model_type == "W2V_Pinyin" :
         
            
            self.model = AutoModelForCTC.from_pretrained(
                "DuyTa/ZH_pinyn", torch_dtype=self.torch_dtype,cache_dir = w2v_pinyin_dir
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained("DuyTa/ZH_pinyn",cache_dir=w2v_pinyin_dir)
        elif self.model_type == "W2V_Hanzi" :
            self.model = AutoModelForCTC.from_pretrained(
                "wbbbbb/wav2vec2-large-chinese-zh-cn", torch_dtype=self.torch_dtype,cache_dir=w2v_hanzi_dir
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained("wbbbbb/wav2vec2-large-chinese-zh-cn",cache_dir=w2v_hanzi_dir)
        
    def infer(self,audiopath:str) -> str:
        pipe = pipeline(
        "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=20,
            batch_size=16,
            return_timestamps="word",
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        prediction = pipe(audiopath)
        result_string = " ".join(map(itemgetter('text'), prediction["chunks"]))
        return result_string


import streamlit as st
from pydub import AudioSegment

st.set_page_config(
    page_title="Chinese ASR",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

audio_tags = {'comments': 'Converted using pydub!'}

import os

upload_path = "uploads/"
download_path = "downloads/"
transcript_path = "transcripts/"


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


create_directory(upload_path)
create_directory(download_path)
create_directory(transcript_path)

@st.cache_resource(show_spinner=True)
def to_wav(audio_file, output_audio_file, upload_path, download_path):
    try:
        audio_data = AudioSegment.from_file(os.path.join(upload_path, audio_file.name))
        audio_data.export(os.path.join(download_path, output_audio_file), format="wav", tags=audio_tags)
        print(os.path.join(download_path, output_audio_file))
        return os.path.join(download_path, output_audio_file)
    except Exception as e:
        print(f"Error: Failed to convert audio file to WAV: {e}")
        return None




@st.cache_resource(show_spinner=True)
def process_audio(filename, model_type):
    """
    Use the model class for inferences !
    """
    if model_type == "W2V_Hanzi" :
        model = Wav2vec2_Model(model_type)
        result = model.infer(filename)
    elif  model_type == "W2V_Pinyin" : 
        model = Wav2vec2_Model(model_type)
        result = model.infer(filename)
    elif  model_type == "Whisper_Hanzi" : 
        model = Whisper_Model(model_type)
        result = model.infer(filename)
    elif  model_type == "Whisper_Pinyin" : 
        model = Whisper_Model(model_type)
        result = model.infer(filename)
    else  :
        raise ValueError(f"Invalid model_type: {model_type}. Please choose from 'W2V_Hanzi', 'W2V_Pinyin', 'Whisper_Hanzi', or 'Whisper_Pinyin'.")
    return result

@st.cache_resource(show_spinner=True)
def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file),"w") as f:
        f.write(transcript_data)

st.title("🗣 Chuyển đổi giọng nói tiếng trung ✨")
st.info('✨ Hỗ trợ các loại tệp - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV 😉')
uploaded_file = st.file_uploader("Upload audio file", type=["wav","mp3","ogg","wma","aac","flac","mp4","flv"])

audio_file = None

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())
    with st.spinner(f"Đang xử lí ... 💫"):
        output_audio_file = uploaded_file.name.split('.')[0] + '.wav'
        output_audio_file = to_wav(uploaded_file, output_audio_file, upload_path, download_path)
        audio_file = open(output_audio_file, 'rb')
        audio_bytes = audio_file.read()
    print("Opening ",audio_file)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Hãy upload bất cứ thứ gì liên quan đến tiếng Trung lên đây 🎼")
        st.audio(audio_bytes)
    with col2:
        whisper_model_type = st.radio("Please choose your model type", ('W2V_Hanzi', 'W2V_Pinyin', 'Whisper_Hanzi', 'Whisper_Pinyin'))

    if st.button("Hiển thị văn bản"):
        with st.spinner(f"Vui lòng đợi trong giây lát ... 💫"):
            transcript = process_audio(audio_bytes, whisper_model_type)
            st.write(transcript)
            output_txt_file = str(output_audio_file.split('/')[-1].replace('.wav', '.txt'))
            save_transcript(transcript, output_txt_file)
            output_file = open(os.path.join(transcript_path,output_txt_file),"r")
            output_file_data = output_file.read()

        if st.download_button(
                             label="Tải tệp văn bản 📝",
                             data=output_file_data,
                             file_name=output_txt_file,
                             mime='text/plain'
                         ):
            st.balloons()
            st.success('✅ Tải xuống thành công  !!')

else:
    st.warning('⚠ Vui lòng chọn một tệp âm thanh 😯')

st.markdown("<br><hr><center>Made by ZD Lab, ASR WebApp with the help of <a href='https://github.com/huggingface/transformers'><strong>Transformers</strong></a> built by <a href='https://github.com/openai'><strong>HuggingFace</strong></a> ✨</center><hr>", unsafe_allow_html=True)


