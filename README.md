# Spam Detection ML

## Overview
Spam filter using **SVM** with text + sender + time features. Preprocess, train, evaluate, save model.

# Tech
- Python 3.9  
- pandas, matplotlib, scikit‑learn, scipy, joblib  

# Dataset
`spam_detection_dataset.csv`  
Features: Sender, Message, Length, Timestamp → hour, day.  
Target: Label (spam/ham).

## Steps
1. Drop `MessageID`, process `Timestamp`.  
2. Clean text, vectorize Message + Sender.  
3. Scale Length, encode Label.  
4. Train SVM (poly kernel).  
5. Evaluate (accuracy, precision, recall, F1, confusion matrix).  
6. Save with joblib (`model.pkl`, `scaler.pkl`, `encoders.pkl`).

## Run
```bash
pip install -r requirements.txt
python spam_detection.py