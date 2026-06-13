# 🔊 AI Alert System for Deaf People

> An AI-powered Environmental Sound Recognition and IoT Alert System designed to help deaf and hard-of-hearing individuals stay aware of their surroundings through real-time visual and mobile notifications.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![IoT](https://img.shields.io/badge/IoT-Enabled-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Overview

People with hearing impairments often miss important environmental sounds such as:

- 🚪 Doorbells
- 🚨 Fire Alarms
- 🚓 Sirens
- 🚗 Vehicle Horns
- 🐶 Dog Barking
- ⏰ Alarms

Missing these sounds can affect safety, communication, and independence.

The **AI Alert System for Deaf People** addresses this challenge by combining **Artificial Intelligence**, **Environmental Sound Recognition**, and **IoT-based notifications** to convert sounds into meaningful visual and mobile alerts.

The system continuously listens to the environment, identifies sounds using a trained AI model, and instantly notifies users through:

- 🌐 Streamlit Web Dashboard
- 📱 Mobile Push Notifications
- 🎨 Visual Emojis & Color Indicators

---

## ✨ Features

- 🎯 Real-time environmental sound detection
- 🤖 AI-powered sound classification using AST (Audio Spectrogram Transformer)
- 📱 Instant mobile notifications through PushBullet
- 🎨 Visual alerts with emojis and confidence scores
- 🔒 Privacy-focused local audio processing
- ⚡ Low-latency real-time inference
- 📈 Scalable architecture for adding new sound classes
- ♿ Designed specifically for deaf and hard-of-hearing users

---

## 🏗️ System Architecture

```text
Environment Sounds
         │
         ▼
   Microphone Input
         │
         ▼
  Audio Preprocessing
 (Librosa / Spectrogram)
         │
         ▼
 Audio Spectrogram
 Transformer (AST)
         │
    ┌────┴────┐
    ▼         ▼
Web UI    PushBullet
(Streamlit) Notifications
    │         │
    └────┬────┘
         ▼
         User
```

---

## 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | Python |
| AI Model | Audio Spectrogram Transformer (AST) |
| Audio Processing | Librosa |
| Deep Learning | TensorFlow, PyTorch |
| Numerical Computing | NumPy |
| Deployment | ONNX Runtime |
| Notifications | PushBullet API |
| Dataset | ESC-50 |

---

## 🔄 How It Works

1. The user starts the application.
2. The microphone continuously captures environmental sounds.
3. Audio clips are converted into Log-Mel Spectrograms.
4. The AST model analyzes and classifies the sound.
5. The system generates a confidence score.
6. Results are displayed on the Streamlit dashboard.
7. Push notifications are sent to the user's smartphone.
8. The monitoring process continues in real time.

---

## 📊 Performance

### Real-Time Testing Accuracy

| Sound | Accuracy |
|---------|---------|
| 🐶 Dog | 100% |
| 🐦 Bird | 100% |
| 🚪 Doorbell | 93.75% |
| 🚨 Alarm | 97.5% |
| 🚗 Car Horn | 92.5% |
| 🚓 Siren | 100% |
| 🐱 Cat | 97.5% |

### Overall Accuracy

**95.6%**

### ESC-50 Validation Accuracy

**96.4%**

---

## 🎯 Supported Sounds

- 🚪 Doorbell
- 🚨 Alarm
- 🚓 Siren
- 🚗 Car Horn
- 🐶 Dog Bark
- 🐱 Cat Sound
- 🐦 Bird Sound

Additional sound categories can be added through future model fine-tuning.

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/husn18/AI-SOUND-SYSTEM-.git
cd AI-SOUND-SYSTEM-
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 📱 PushBullet Setup

1. Create a PushBullet account.
2. Generate an API Access Token.
3. Add the token to your environment variables.

```env
PUSHBULLET_API_KEY=your_api_key_here
```

---

## 🌍 Applications

### ♿ Accessibility Assistance
Provides real-time environmental awareness for deaf and hard-of-hearing individuals.

### 🏠 Smart Homes
Detects alarms, doorbells, and emergency sounds.

### 👴 Elderly Care
Improves safety through instant notifications.

### 🏢 Workplace Accessibility
Enhances inclusivity in offices and institutions.

### 🎓 Educational Environments
Assists hearing-impaired students in classrooms and campuses.

---

## 🔮 Future Scope

- ⌚ Smartwatch integration
- 📳 Wearable vibration alerts
- 🏠 Smart home automation support
- ☁️ Optional cloud backup
- 📊 Detection history dashboard
- 🎯 Custom sound training
- 🚀 Raspberry Pi deployment
- 📡 ESP32 integration
- 🚨 Emergency contact notifications
- 🔊 Multi-sound detection capability

---

## 👥 Team

### Group Leader
**Husandeep Singh**  
B.Tech CSE, NIT Kurukshetra

### Team Members
- Abhishek Rai
- Prince Pal

---

## 🎓 Academic Information

**Course:** IoT Programming (CSPC 209)  
**Institute:** National Institute of Technology Kurukshetra (NIT KKR)

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📜 License

This project is developed for educational, research, and accessibility purposes.

---

## ⭐ Support

If you found this project useful, please consider giving it a **Star ⭐** on GitHub.

Together, we can build technology that makes the world more accessible for everyone.
