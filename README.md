# Beyond-the-Smile-A-Machine-Learning-Breakthrough-in-Dental-Disease-Detection
## 📌 Project Overview
The Integrated Dental AI Diagnostic System is a multi-model deep learning framework designed to assist dentists in automated diagnosis using both dental X-rays and real-world oral images. The system integrates CNN-based disease classification, YOLO-based radiographic detection, and explainable AI to deliver accurate, interpretable, and clinically relevant diagnostic outputs.

---

## 🎯 Objectives
- Automated diagnosis from oral images and dental X-rays  
- Accurate localization of dental anomalies  
- Integration of multiple AI models into a single pipeline  
- Generation of explainable, clinician-friendly outputs  
- Professional report generation and patient record storage  

---

## 🧠 System Architecture
The system consists of three primary modules:

1. Disease Classification Model (EfficientNet-B3)  
   - Input: RGB oral images  
   - Output: Probabilities for six dental diseases  

2. X-ray Detection Model (YOLOv8)  
   - Input: Grayscale dental X-rays  
   - Output: Bounding boxes for dental anomalies  

3. Explainable AI Module  
   - Generates short, clinically meaningful explanations  

An Auto-Detection layer routes images to the appropriate model automatically.

---

## 📂 Datasets
- Combined Dental X-ray Dataset (OPG + Radiographs)  
- Oral Disease Dataset (6 Classes): Calculus, Data Caries, Gingivitis, Mouth Ulcer, Tooth Discoloration, Hypodontia  

---

## ⚙️ Technologies Used
- Python  
- TensorFlow, PyTorch  
- EfficientNet-B3, YOLOv8  
- Streamlit  
- MongoDB Atlas  
- Google Gemini API  

---

## 📊 Performance Summary
- Disease Classification Accuracy: ~94%  
- YOLO Detection mAP@50: ~0.34  
- Stable training with minimal overfitting  

---

## 🧪 Testing
- Unit testing for preprocessing and tensor validation  
- Integration testing for GPU inference and pipeline stability  
- Safe fallback mechanisms for database connectivity  

---

## 📄 Features
- Image upload and live camera capture  
- Auto model selection  
- Real-time predictions  
- Explainable AI summaries  
- PDF report generation  
- Patient records dashboard  

---

## 🚀 Running the Project
1. Install dependencies  
   pip install -r requirements.txt  

2. Set environment variables  
   GEMINI_API_KEY  
   MONGO_URI  

3. Launch the application  
   streamlit run app.py  

---

## 🔐 Ethics & Security
- Environment-based credential handling  
- Explainable predictions to support clinical trust  
- Designed as a clinical decision support tool  

---

## 📈 Future Scope
- CBCT integration  
- Longitudinal patient tracking  
- Mobile deployment  
- Orthodontic analysis  

---

## 👨‍💻 Project Details
Beyond the Smile: A Machine Learning Breakthrough in Dental Disease Detection  
Department of AIML, DSATM  
Academic Year: 2025–2026  

---

## 📜 License
Academic and research use only.
