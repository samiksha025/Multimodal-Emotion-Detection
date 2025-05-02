import gradio as gr
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from PIL import Image
import os
import joblib
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from huggingface_hub import hf_hub_download
import datetime
import json
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# ---------- TEXT MODEL ----------
text_model_path = "bert_emotion_model"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_path).to(device)
label_encoder_text = joblib.load("bert_emotion_model/label_encoder.joblib")
 
def predict_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = label_encoder_text.inverse_transform([pred_idx])[0]
        confidence = round(probs[0][pred_idx].item(), 4)
    return pred_label, confidence
 
# ---------- VOICE MODEL (with safetensors patch) ----------
voice_model_path = "wav2vec2-emotion-model"
voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    voice_model_path,
    from_safetensors=True
).to(device)
voice_processor = Wav2Vec2Processor.from_pretrained(voice_model_path)
voice_model.eval()
id2label_voice = voice_model.config.id2label
 
def predict_voice(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = voice_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = voice_model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = id2label_voice[pred_idx]
        confidence = round(probs[0][pred_idx].item(), 4)
    return pred_label, confidence
 
# ---------- IMAGE MODEL ----------
class_labels = ['angry', 'fear', 'happy', 'sad', 'surprise']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
image_model_path = hf_hub_download(
    repo_id="SweArmy22/emotion-journal-app",
    filename="final_emotion_image_classifier.pth"
)
 
image_model = models.resnet18(pretrained=False)
image_model.fc = torch.nn.Linear(image_model.fc.in_features, len(class_labels))
image_model.load_state_dict(torch.load(image_model_path, map_location=device))
image_model = image_model.to(device)
image_model.eval()
 
def predict_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = image_model(img_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = class_labels[pred_idx]
        confidence = round(probs[0][pred_idx].item(), 4)
    return pred_label, confidence
 
# ---------- FUSION ----------
def fusion_predict(text, audio_path, image):
    text_label, text_conf = predict_text(text)
    voice_label, voice_conf = predict_voice(audio_path)
    image_label, image_conf = predict_image(image)
    labels = [text_label, voice_label, image_label]
    final_label = max(set(labels), key=labels.count)
    final_confidence = round((text_conf + voice_conf + image_conf) / 3, 4)
    return final_label, final_confidence, text_conf, voice_conf, image_conf
 
# ---------- LOGGING ----------
def log_interaction(user_id, text, image_path, audio_path, result):
    log = {
        "user_id": user_id,
        "timestamp": str(datetime.datetime.now()),
        "text": text,
        "image": image_path,
        "audio": audio_path,
        "result": result
    }
    with open("user_logs.json", "a") as f:
        f.write(json.dumps(log) + "\n")
 
# ---------- MINDFULNESS + PROMPT ----------
def get_mindfulness_suggestions(emotion):
    suggestions = {
        "angry": "Take deep breaths and try to release your frustration.",
        "fear": "Acknowledge your fears, but focus on grounding yourself in the present moment.",
        "happy": "Enjoy the moment and spread your happiness to others.",
        "sad": "It’s okay to feel sad, try to engage in activities that lift your mood.",
        "surprise": "Take a moment to process the unexpected and stay mindful of your reaction."
    }
    return suggestions.get(emotion, "Try to reflect on your emotions and breathe deeply.")
 
def get_reflection_prompt(emotion):
    prompts = {
        "angry": "What triggered this anger, and how can you address it calmly?",
        "fear": "What is causing your fear, and how can you overcome it?",
        "happy": "What are you grateful for right now, and how can you spread positivity?",
        "sad": "What steps can you take to move forward and improve your mood?",
        "surprise": "How can you adapt to unexpected changes and stay positive?"
    }
    return prompts.get(emotion, "Reflect on your feelings and how you can improve your emotional well-being.")
 
# ---------- MAIN GRADIO APP ----------
def analyze_emotion(text, image, audio):
    audio_path = "/tmp/temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio)
    final_label, final_confidence, text_conf, voice_conf, image_conf = fusion_predict(text, audio_path, image)
    mindfulness_suggestion = get_mindfulness_suggestions(final_label)
    reflection_prompt = get_reflection_prompt(final_label)
    confidence_scores = {
        "text_confidence": text_conf,
        "voice_confidence": voice_conf,
        "image_confidence": image_conf
    }
    return final_label, confidence_scores, mindfulness_suggestion, reflection_prompt
 
gr.Interface(
    fn=analyze_emotion,
    inputs=[
        gr.Textbox(label="Enter journal text"),
        gr.Image(label="Upload an image"),
        gr.Audio(label="Upload your voice")
    ],
    outputs=[
        gr.Textbox(label="Detected Emotion"),
        gr.JSON(label="Confidence Scores"),
        gr.Textbox(label="Mindfulness Suggestions"),
        gr.Textbox(label="Reflection Prompt")
    ],
    title="AI-powered Mood Journal & Emotion Tracker"
).launch()