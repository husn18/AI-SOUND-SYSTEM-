
import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import librosa
import torch
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import json
import plotly.graph_objects as go
import tempfile
import time
from pushbullet import Pushbullet
from transformers import ASTForAudioClassification, ASTFeatureExtractor


ONNX_MODEL_PATH = "ast_model.onnx"
CONFIG_PATH = "ast_model_config.json"
API_KEY = "o.G9LlnANFHXPh51eEwFYqaoqc0klmIyoW"
pb = Pushbullet(API_KEY)

TARGET_CLASSES = {
    'doorbell': {'keywords': ['doorbell', 'door knock', 'door', 'knock'], 'emoji': '🔔'},
    'dog': {'keywords': ['dog', 'dog bark', 'barking'], 'emoji': '🐕'},
    'cat': {'keywords': ['cat', 'meow', 'cat meowing'], 'emoji': '🐱'},
    'bird': {'keywords': ['bird', 'chirping birds', 'bird vocalization'], 'emoji': '🐦'},
    'car_horn': {'keywords': ['car horn', 'vehicle horn', 'horn'], 'emoji': '🚗'},
    'baby_crying': {'keywords': ['baby cry', 'crying baby', 'infant cry'], 'emoji': '👶'},
    'alarm': {'keywords': ['alarm clock', 'alarm', 'clock alarm'], 'emoji': '⏰'},
    'siren': {'keywords': ['siren', 'ambulance', 'police car'], 'emoji': '🚨'},
    'phone': {'keywords': ['telephone', 'phone ringing', 'telephone bell ringing'], 'emoji': '📞'},
    'glass_breaking': {'keywords': ['glass breaking', 'breaking glass'], 'emoji': '💥'}
}

def send_alert(predicted_class,confidence):
    title = f"Sound Alert • {predicted_class.upper()} Detected!"
    print("HI")
    
    body = (
        f"Detected Sound: {predicted_class}\n"
        f"  Confidence: {confidence:.1%}\n"
    )
    push = pb.push_note(title,body)

class AudioTransformerClassifier:
    """Sound classifier with ONNX support"""
    
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", use_onnx=True):
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.onnx_session = None
        self.pytorch_model = None
        
     
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        
       
        if use_onnx and os.path.exists(ONNX_MODEL_PATH) and os.path.exists(CONFIG_PATH):
            try:
                import onnxruntime as ort
                
               
                with open(CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                self.labels = {int(k): v for k, v in config['labels'].items()}
                
                
                providers = ['CPUExecutionProvider']
                self.onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
                
            except Exception as e:
                self.onnx_session = None
        
      
        if self.onnx_session is None:
            self.pytorch_model = ASTForAudioClassification.from_pretrained(model_name)
            self.pytorch_model.eval()
            self.labels = self.pytorch_model.config.id2label
        
      
        self._create_class_mapping()
    
    def _create_class_mapping(self):
        """Create mapping from AudioSet classes to target classes"""
        self.class_mapping = {}
        
        for target_class, info in TARGET_CLASSES.items():
            keywords = info['keywords']
            for label_id, label in self.labels.items():
                label_lower = label.lower()
                for keyword in keywords:
                    if keyword in label_lower:
                        if target_class not in self.class_mapping:
                            self.class_mapping[target_class] = []
                        self.class_mapping[target_class].append(label_id)
                        break
    
    def load_audio(self, audio_path, sr=16000, duration=7):
        """Load and preprocess audio file"""
        waveform, _ = librosa.load(audio_path, sr=sr, duration=duration)
        target_length = sr * duration
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        else:
            waveform = waveform[:target_length]
        return waveform
    
    def predict(self, audio_path, top_k=10):
        """Predict sound class for an audio file"""
       
        waveform = self.load_audio(audio_path)
        
    
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt" if self.pytorch_model else "np"
        )
        
      
        if self.onnx_session:
            input_values = inputs['input_values']
            if isinstance(input_values, torch.Tensor):
                input_values = input_values.numpy()
            
            ort_inputs = {self.onnx_session.get_inputs()[0].name: input_values}
            logits = self.onnx_session.run(None, ort_inputs)[0]
            probs = self._softmax(logits[0])
        else:
            with torch.no_grad():
                outputs = self.pytorch_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs = probs.cpu().numpy()[0]
        
      
        target_scores = {}
        for target_class, label_ids in self.class_mapping.items():
            score = sum(probs[label_id] for label_id in label_ids)
            target_scores[target_class] = score
        
      
        sorted_classes = sorted(target_scores.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:top_k]
        
        predictions = [
            {'class': cls, 'confidence': float(score), 'emoji': TARGET_CLASSES[cls]['emoji']}
            for cls, score in sorted_classes
        ]
        
        return {
            'audio_file': os.path.basename(audio_path),
            'predicted_class': predictions[0]['class'],
            'confidence': predictions[0]['confidence'],
            'emoji': predictions[0]['emoji'],
            'top_predictions': predictions,
            'using_onnx': self.onnx_session is not None
        }
    
    @staticmethod
    def _softmax(x):
        """Numpy softmax implementation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()



st.set_page_config(
    page_title="🎵 Sound Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: center;
        color: #8e9aaf;
        font-size: 1.4rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Prediction Box - Gradient Style */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .predicted-class {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .confidence {
        font-size: 2rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* Status Box */
    .status-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(240, 147, 251, 0.3);
    }
    
    .status-text {
        font-size: 2rem;
        font-weight: 600;
        color: white;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.3rem;
        font-weight: 600;
        padding: 1rem 2rem;
        color: #8e9aaf;
    }
    
    .stTabs [aria-selected="true"] {
        color: #667eea;
        border-bottom: 3px solid #667eea;
    }
    
    /* Buttons */
    .stButton > button {
        font-size: 1.3rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* File Uploader */
    .uploadedFile {
        font-size: 1.2rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Headings */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    h2 {
        font-size: 2rem !important;
        font-weight: 600 !important;
    }
    
    h3 {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    
    /* Info boxes */
    .stAlert {
        font-size: 1.2rem;
        padding: 1.5rem;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('<p class="main-header">🎵 Sound Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload or record audio to identify sounds using AI</p>', unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    with st.spinner("🔮 Loading AI model..."):
        return AudioTransformerClassifier(use_onnx=True)

try:
    classifier = load_classifier()
    

    engine = "ONNX (Fast)" if classifier.onnx_session else "PyTorch"
    st.info(f"✅ **Model Loaded Successfully** | Engine: **{engine}** | Recording Duration: **5 seconds**")
    
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()


tab1, tab2, tab3 = st.tabs(["📤 Upload Audio", "🎤 Record Audio", "🔴 Real-Time Monitor"])


with tab1:
    st.markdown("### 📁 Upload an audio file")
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV, MP3, FLAC, OGG, M4A)",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_path = tmp_file.name
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🔍 Analyze Sound", key="analyze_upload", use_container_width=True):
                with st.spinner("🎯 Analyzing audio..."):
                    try:
                       
                        result = classifier.predict(audio_path, top_k=10)
                        
                        
                        send_alert(result["predicted_class"],result["confidence"])
                        st.markdown(f'''
                        <div class="prediction-box">
                            <p class="predicted-class">{result["emoji"]} {result["predicted_class"].upper()}</p>
                            <p class="confidence">Confidence: {result["confidence"]:.1%}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                    
                        st.markdown("### 📊 Top 10 Predictions")
                        
                        classes = [f"{pred['emoji']} {pred['class']}" for pred in result['top_predictions']]
                        confidences = [pred['confidence'] * 100 for pred in result['top_predictions']]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                y=classes[::-1],
                                x=confidences[::-1],
                                orientation='h',
                                marker=dict(
                                    color=confidences[::-1],
                                    colorscale='Purples',
                                    showscale=False,
                                    line=dict(color='rgba(102, 126, 234, 0.6)', width=2)
                                ),
                                text=[f'{c:.1f}%' for c in confidences[::-1]],
                                textposition='auto',
                                textfont=dict(size=14, color='white', family='Poppins')
                            )
                        ])
                        
                        fig.update_layout(
                            title=dict(text="Prediction Confidence Scores", font=dict(size=24, family='Poppins', color='#667eea')),
                            xaxis_title="Confidence (%)",
                            yaxis_title="Sound Class",
                            height=600,
                            showlegend=False,
                            xaxis=dict(range=[0, 100], gridcolor='rgba(0,0,0,0.1)'),
                            yaxis=dict(tickfont=dict(size=14)),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=14, family='Poppins')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        
                        with st.expander("📋 View Detailed Predictions"):
                            for i, pred in enumerate(result['top_predictions'], 1):
                                st.write(f"**{i}. {pred['emoji']} {pred['class']}**: {pred['confidence']:.2%}")
                        
                    except Exception as e:
                        st.error(f"❌ Error analyzing audio: {e}")
                    finally:
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)

with tab2:
    st.markdown("### 🎙️ Record audio from your microphone")
    st.info("⏱️ Recording will capture **5 seconds** of audio automatically")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔴 Start Recording", key="record_btn", use_container_width=True):
            try:
                duration = 5
                sr = 16000
             
                countdown_placeholder = st.empty()
                for i in [3, 2, 1]:
                    countdown_placeholder.markdown(f'''
                    <div class="status-box">
                        <p class="status-text">Starting in... {i}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    time.sleep(1)
                
                countdown_placeholder.markdown('''
                <div class="status-box">
                    <p class="status-text">🔴 RECORDING NOW!</p>
                </div>
                ''', unsafe_allow_html=True)
         
                progress_bar = st.progress(0)
                recording = sd.rec(int(duration * sr), 
                                 samplerate=sr, 
                                 channels=1, 
                                 dtype='float32')
                
                for i in range(duration * 10):
                    time.sleep(0.1)
                    progress_bar.progress((i + 1) / (duration * 10))
                
                sd.wait()
                countdown_placeholder.markdown('''
                <div class="status-box">
                    <p class="status-text">✅ Recording Complete!</p>
                </div>
                ''', unsafe_allow_html=True)
                progress_bar.empty()
                
               
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_path = f"recorded_{timestamp}.wav"
                sf.write(audio_path, recording, sr)
                
             
                st.audio(audio_path, format='audio/wav')
                
               
                with st.spinner("🎯 Analyzing audio..."):
                    result = classifier.predict(audio_path, top_k=10)
                    
                   
                    send_alert(result["predicted_class"],result["confidence"])
                    st.markdown(f'''
                    <div class="prediction-box">
                        <p class="predicted-class">{result["emoji"]} {result["predicted_class"].upper()}</p>
                        <p class="confidence">Confidence: {result["confidence"]:.1%}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    
                    st.markdown("### 📊 Top 10 Predictions")
                    
                    classes = [f"{pred['emoji']} {pred['class']}" for pred in result['top_predictions']]
                    confidences = [pred['confidence'] * 100 for pred in result['top_predictions']]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            y=classes[::-1],
                            x=confidences[::-1],
                            orientation='h',
                            marker=dict(
                                color=confidences[::-1],
                                colorscale='Purples',
                                showscale=False,
                                line=dict(color='rgba(102, 126, 234, 0.6)', width=2)
                            ),
                            text=[f'{c:.1f}%' for c in confidences[::-1]],
                            textposition='auto',
                            textfont=dict(size=14, color='white', family='Poppins')
                        )
                    ])
                    
                    fig.update_layout(
                        title=dict(text="Prediction Confidence Scores", font=dict(size=24, family='Poppins', color='#667eea')),
                        xaxis_title="Confidence (%)",
                        yaxis_title="Sound Class",
                        height=600,
                        showlegend=False,
                        xaxis=dict(range=[0, 100], gridcolor='rgba(0,0,0,0.1)'),
                        yaxis=dict(tickfont=dict(size=14)),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=14, family='Poppins')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📋 View Detailed Predictions"):
                        for i, pred in enumerate(result['top_predictions'], 1):
                            st.write(f"**{i}. {pred['emoji']} {pred['class']}**: {pred['confidence']:.2%}")
                
            except Exception as e:
                st.error(f"❌ Error during recording: {e}")

with tab3:
    st.markdown("### 🔴 Real-Time Sound Monitoring")
    st.info("⏱️ Continuously monitors audio in **5-second intervals**")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
     
        status_placeholder = st.empty()
        status_placeholder.markdown('''
        <div class="status-box">
            <p class="status-text">Monitoring Stopped</p>
        </div>
        ''', unsafe_allow_html=True)
        
      
        timer_col1, timer_col2 = st.columns(2)
        with timer_col1:
            st.markdown("**Status:**")
            status_text = st.empty()
            status_text.markdown("Stopped")
        with timer_col2:
            st.markdown("**Duration:**")
            duration_text = st.empty()
            duration_text.markdown("5s")
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_monitoring = st.button("▶️ START", key="start_monitor", use_container_width=True)
        with btn_col2:
            stop_monitoring = st.button("⏹️ STOP", key="stop_monitor", use_container_width=True)
    
    
    results_placeholder = st.empty()
    
   
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    
    if start_monitoring:
        st.session_state.monitoring = True
    
    if stop_monitoring:
        st.session_state.monitoring = False
        status_placeholder.markdown('''
        <div class="status-box">
            <p class="status-text">Monitoring Stopped</p>
        </div>
        ''', unsafe_allow_html=True)
        status_text.markdown("Stopped")
    
    if st.session_state.monitoring:
        try:
            duration = 5
            sr = 16000
            
            while st.session_state.monitoring:
               
                status_placeholder.markdown('''
                <div class="status-box">
                    <p class="status-text">🔴 Monitoring Active</p>
                </div>
                ''', unsafe_allow_html=True)
                status_text.markdown("**Recording...**")
                
              
                recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
                sd.wait()
                
            
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_path = f"monitor_{timestamp}.wav"
                sf.write(audio_path, recording, sr)
                
              
                status_text.markdown("**Analyzing...**")
                result = classifier.predict(audio_path, top_k=10)
                
               
                send_alert(result["predicted_class"],result["confidence"])
                
                with results_placeholder.container():
                    st.markdown(f'''
                    <div class="prediction-box">
                        <p class="predicted-class">{result["emoji"]} {result["predicted_class"].upper()}</p>
                        <p class="confidence">Confidence: {result["confidence"]:.1%}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                 
                    classes = [f"{pred['emoji']} {pred['class']}" for pred in result['top_predictions']]
                    confidences = [pred['confidence'] * 100 for pred in result['top_predictions']]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            y=classes[::-1],
                            x=confidences[::-1],
                            orientation='h',
                            marker=dict(color=confidences[::-1], colorscale='Purples', showscale=False),
                            text=[f'{c:.1f}%' for c in confidences[::-1]],
                            textposition='auto',
                            textfont=dict(size=14, color='white', family='Poppins')
                        )
                    ])
                    
                    fig.update_layout(
                        height=500,
                        showlegend=False,
                        xaxis=dict(range=[0, 100]),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
              
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                
                status_text.markdown("**Waiting...**")
                
                if not st.session_state.monitoring:
                    break
                    
        except Exception as e:
            st.error(f"❌ Monitoring error: {e}")
            st.session_state.monitoring = False


with st.sidebar:
    st.markdown("## 🎵 About")
    st.markdown("""
    This app uses a pre-trained **Audio Spectrogram Transformer (AST)** 
    to classify sounds in real-time.
    """)
    
    st.markdown("---")
    st.markdown("## 🎯 Supported Classes")
    
    for cls, info in TARGET_CLASSES.items():
        st.markdown(f"**{info['emoji']} {cls.replace('_', ' ').title()}**")
    
    st.markdown("---")
    st.markdown("## ⚙️ Configuration")
    st.markdown(f"""
    - **Model**: AST (MIT)  
    - **Sample Rate**: 16 kHz  
    - **Duration**: 5 seconds  
    - **Engine**: {engine}
    """)
    
    st.markdown("---")
    st.markdown("## 💡 Tips")
    st.markdown("""
    ✅ Use clear, isolated sounds  
    ✅ Minimize background noise  
    ✅ Keep steady during recording  
    ✅ Upload high-quality files  
    """)