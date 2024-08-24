# DeepFake Detection Using Fine-Tuned Wav2Vec2

This project implements a DeepFake detection system using the Wav2Vec2 model fine-tuned for audio classification tasks. The goal is to accurately distinguish between genuine and DeepFake audio samples by leveraging advanced deep learning techniques, particularly in the domain of speech processing.

## Introduction

With the rising threat of deepfake attacks, accurate solutions are crucial for biometric voice security. This study proposes a method for identifying audio deepfakes using the Wave2Vec2 framework. Custom preprocessing during fine-tuning on the In-the-Wild dataset notably improved accuracy, comparable to peer models. Adding five layers (dropout, dense, tanh, dropout, out_proj) to the Wav2Vec2 architecture further enhanced classification performance. To ensure prediction reliability, a custom algorithm was implemented where the model’s logits were compared to a threshold and the process repeated until a confident result was achieved. The finetuned Wav2Vec2 model demonstrated compelling performance when finetuning only custom classification layers (Grounded Wav2Vec2) and entire model (Adaptive Wav2Vec2) configurations. 

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/Sarkarsubham2002/DeepFake-detection-Using-Wav2Vec2.git
cd DeepFake-detection-Using-Wav2Vec2
pip install -r requirements.txt
```

## Dataset

The dataset used for this study is the ’In-the-Wild’ Dataset consisting of audio deepfakes for a set of politicians and other public figures audio for deepfake detection. The dataset has used cqtspec and logspec features instead of melspec features. The dataset comprises 38 hours of recordings, including 17.2 hours of deepfakes and 20.8 hours of bonafide audio data, featuring public figures. This dataset reveals significant performance drops in models in real-world scenarios, hinting at possible overspecialization to the ASVspoof benchmark.

## Model Architecture

The project utilizes the Wav2Vec2 model, pre-trained on large-scale speech data and fine-tuned for the specific task of DeepFake detection. The model architecture includes:

- **Base Model**: Wav2Vec2 pre-trained model.
- **Custom Classification Head**: A fully connected layer, dropout layer, and softmax layer are added to the base model for binary classification (Real vs. Fake).

## Comparison of Results

The fine-tuned Wav2Vec2 model demonstrates superior performance compared to baseline models. The table below summarizes the results of both of our models:

| **Model**               | **Accuracy (%)** | **EER (%)** |
|-------------------------|------------------|-------------|
| **Adaptive Wave2Vec2**  | **99.59**        | **1.4**     |
| **Grounded Wave2Vec2**  | **99.42**        | **1.53**    |

The fine-tuned Wav2Vec2 model outperforms other models in both accuracy and EER, making it a robust solution for detecting DeepFake audio.
