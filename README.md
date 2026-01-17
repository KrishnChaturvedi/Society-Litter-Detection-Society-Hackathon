🚯 Society Litter Detection System:

AI-Powered Real-Time Litter Monitoring Using YOLOv8 & Streamlit
An intelligent computer vision system that detects litter, humans, and human-with-litter scenarios in real time to promote cleaner public spaces and responsible behavior.
Built using YOLOv8 transfer learning, OpenCV, and Streamlit, this project demonstrates a market-ready AI solution for smart societies, malls, campuses, and residential areas.


🔍 Problem Statement:
Urban areas and societies face persistent littering problems. Manual monitoring is inefficient, costly, and unreliable. Existing surveillance systems lack context-aware intelligence to distinguish between trash, people, and responsible behavior.


💡 Solution Overview
This project introduces an AI-based litter detection system that:
Automatically detects trash in camera frames
Separately identifies humans (never misclassified as trash)
Detects human + trash simultaneously to flag irresponsible disposal
Auto-captures evidence frames
Sends notifications for cleaning or action
Works through an interactive Streamlit web app


🧠 Technical Approach:

1️⃣ Model Architecture
YOLOv8n (Ultralytics) for object detection
Lightweight, fast, real-time inference

2️⃣ Transfer Learning
Pretrained on COCO dataset
Fine-tuned using TACO (Trash Annotations in Context) dataset
Custom training for litter-specific classes

3️⃣ Dual-Model Safety Pipeline
Custom Litter Detection Model → detects trash
COCO Person Model → detects humans
Humans are never classified as trash
Ethical and bias-aware detection

4️⃣ Real-Time Processing
OpenCV for frame handling
Auto-capture logic with cooldown (prevents spam)
Saves incident images locally or sends notifications

5️⃣ Deployment Layer
Streamlit for interactive UI
Sidebar navigation (Home, Camera, Image Upload)
Production-style .gitignore for security

🖥️ Application Features
📷 Live camera litter detection
🖼️ Image upload detection
🚨 Trash alert notifications
⚠️ Human-with-trash warnings
📁 Evidence image storage
🎨 Clean, user-friendly UI
🔐 Secure deployment practices
🏗️ Project Structure


Litter Detection/
│
├── src/
│   ├── app.py              # Streamlit app
│   ├── main.py             
│   ├── main2.py            
│
├── requirements.txt
├── .gitignore
└── README.md


⚠️ Datasets, trained weights, virtual environments, and secrets are intentionally excluded using .gitignore.
🔐 Security & Best Practices
❌ No datasets pushed to GitHub
❌ No trained model weights exposed
❌ No API keys committed
✅ Industry-standard ML deployment workflow
This mirrors real-world production systems where models and data are stored securely outside version control.


🌍 Societal Impact
Encourages cleaner public spaces
Enables faster waste management response
Reduces manual monitoring costs
Promotes civic responsibility using AI
Scalable for smart cities and private spaces

🚀 Market Readiness
This project is designed to scale into:
Smart city surveillance systems
Mall & campus cleanliness monitoring
Residential society automation
Government & municipal waste solutions
Future upgrades:
Mobile app integration
Cloud notifications
FastAPI backend
Secure model hosting
Multi-camera support


📄 License

This project is developed for educational and hackathon purposes.
Commercial usage requires proper authorization.
