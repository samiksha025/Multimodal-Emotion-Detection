***Multimodal Emotion Recognition: AI-Powered Mood Journal & Emotion Tracker***

**Project Description**

This repository contains the implementation of a multimodal emotion recognition system, integrating text, audio, and facial image analysis to accurately identify human emotions. Built using state-of-the-art deep learning models, this tool provides users with real-time emotional insights, mindfulness suggestions, and self-reflection prompts to support emotional well-being and mental health journaling.

**Project Overview**

In the growing field of artificial intelligence, emotion recognition provides an essential human touch to digital interactions. This project leverages three core modalities—text, audio, and image—to deliver comprehensive emotion detection:

Text Analysis: BERT (Bidirectional Encoder Representations from Transformers)

Audio Analysis: Wav2Vec2

Facial Expression Analysis: ResNet18 (CNN)

These modalities are integrated using a late decision fusion strategy for robust predictions.

**Technologies & Techniques**

BERT: Deep transformer model for text analysis, achieving ~99.9% accuracy.

Wav2Vec2: Transformer-based model analyzing audio waveforms directly, achieving ~94% accuracy.

ResNet18: Convolutional neural network for facial emotion recognition, achieving ~70% accuracy.

Gradio: Interactive web interface providing real-time multimodal emotion analysis.

Late Fusion: Decision-level integration ensuring accurate results even with incomplete or ambiguous input.

**Dataset**

Text Data: Emotion-labeled journal entries (balanced across five primary emotions: happy, sad, angry, fearful, and surprised).

Audio Data: RAVDESS emotional speech dataset.

Image Data: Facial images labeled with emotions from publicly available datasets.

**Key Features**

Real-time emotion detection using text, voice, and facial images

Mindfulness suggestions tailored to detected emotions

Reflection prompts to encourage emotional awareness

User-friendly graphical interface


**Results and Observations**

![image](https://github.com/user-attachments/assets/34d1f562-7db5-4708-a77d-fa6253ec68ef)

![image](https://github.com/user-attachments/assets/79b790d9-ee62-4d17-8fe8-a450bbf41248)

![image](https://github.com/user-attachments/assets/4ee098a7-0f29-457a-8fe8-36e07bce280f)

![image](https://github.com/user-attachments/assets/73d93eca-1ba6-47ed-aa90-189104a4a5f7)

![image](https://github.com/user-attachments/assets/3ea8594d-1078-4910-9ee7-065f2d44edf4)

**HuggingFace URL: (UI Interface):**
UI - Emotion Journal App - a Hugging Face Space by SweArmy22
Repo files - SweArmy22/emotion-journal-app at main







