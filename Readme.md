🔊 AI Alert System for Deaf People

An AI-powered Environmental Sound Recognition and IoT Alert System designed to help deaf and hard-of-hearing individuals stay aware of their surroundings through real-time visual and mobile notifications.










📖 Overview

People with hearing impairments often miss important environmental sounds such as:

🚪 Doorbells
🚨 Fire Alarms
🚓 Sirens
🚗 Vehicle Horns
🐶 Dog Barking
⏰ Alarms

Missing these sounds can affect safety, communication, and independence.

The AI Alert System for Deaf People addresses this challenge by combining Artificial Intelligence, Environmental Sound Recognition, and IoT-based notifications to convert sounds into meaningful visual and mobile alerts.

The system continuously listens to the environment, identifies sounds using a trained AI model, and instantly notifies users through:

🌐 Streamlit Web Dashboard
📱 Mobile Push Notifications
🎨 Visual Emojis & Color Indicators
✨ Key Features
🎯 Real-Time Sound Detection

Continuously monitors environmental sounds through a microphone.

🤖 AI-Powered Classification

Uses an Audio Spectrogram Transformer (AST) model trained on environmental sound datasets.

📱 Mobile Notifications

Sends instant alerts to smartphones using PushBullet.

🎨 Accessible User Interface

Visual alerts with emojis and confidence scores for quick recognition.

🔒 Privacy First

Audio processing occurs locally on the device. Raw audio is not uploaded to external servers.

⚡ Low Latency

Fast local inference enables real-time response.

📈 Scalable Architecture

New sound categories can be added through fine-tuning without retraining the entire model.

🏗️ System Architecture
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
🛠️ Tech Stack
Component	Technology
Frontend	Streamlit
Backend	Python
AI Model	Audio Spectrogram Transformer (AST)
Audio Processing	Librosa
Deep Learning	TensorFlow / PyTorch
Numerical Computing	NumPy
Deployment	ONNX Runtime
Notifications	PushBullet API
Dataset	ESC-50
🔄 Workflow
User starts the application.
Microphone captures environmental sounds.
Audio clips are converted into spectrograms.
AST model classifies the detected sound.
Confidence score is generated.
Results are displayed on the Streamlit dashboard.
Mobile notifications are sent via PushBullet.
System continues monitoring in real time.
📊 Performance
Real-Time Testing Results
Sound Type	Accuracy
Dog Bark	100%
Bird Sound	100%
Doorbell	93.75%
Alarm	97.5%
Car Horn	92.5%
Siren	100%
Cat Sound	97.5%
Overall Accuracy

95.6%

ESC-50 Validation Accuracy

96.4%

🎯 Supported Sound Categories
🚪 Doorbell
🚨 Alarm
🚓 Siren
🚗 Car Horn
🐶 Dog Bark
🐱 Cat Sound
🐦 Bird Sound

More sound classes can be added through future model fine-tuning.

🚀 Installation
Clone Repository
git clone https://github.com/husn18/AI-SOUND-SYSTEM-.git
cd AI-SOUND-SYSTEM-
Create Virtual Environment
python -m venv venv
Activate Environment
Windows
venv\Scripts\activate
Linux / Mac
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Run Application
streamlit run app.py
📱 PushBullet Configuration
Create a PushBullet account.
Generate an Access Token.
Add the token to your environment variables or configuration file.
PUSHBULLET_API_KEY=your_api_key
🌍 Real World Applications
Assistive Technology

Helping deaf and hard-of-hearing individuals stay aware of important sounds.

Smart Homes

Alerting users about alarms, doorbells, and emergencies.

Elderly Care

Providing additional safety monitoring.

Workplace Accessibility

Creating more inclusive environments.

Educational Institutions

Assisting hearing-impaired students in classrooms.

🔮 Future Enhancements
⌚ Smartwatch Integration
📳 Wearable Vibration Alerts
🏠 Smart Home Integration
☁️ Optional Cloud Backup
🧠 Personalized Sound Training
📊 Detection History Dashboard
🔋 Raspberry Pi Deployment
📡 ESP32 Edge Deployment
🚨 Emergency Contact Notifications
🎯 Multi-Sound Detection
👥 Team
Group Leader

Husandeep Singh

B.Tech CSE
National Institute of Technology Kurukshetra
Team Members
Abhishek Rai
Prince Pal
🎓 Academic Information

Course: IoT Programming (CSPC 209)
Institution: National Institute of Technology Kurukshetra (NIT KKR)

🤝 Contributing

Contributions, suggestions, and feature requests are welcome.

Fork the repository
Create a feature branch
git checkout -b feature-name
Commit changes
git commit -m "Add new feature"
Push to GitHub
git push origin feature-name
Open a Pull Request
📜 License

This project is intended for educational, research, and accessibility purposes.

⭐ If you found this project useful, consider giving the repository a star and supporting accessible AI solutions for everyone.
