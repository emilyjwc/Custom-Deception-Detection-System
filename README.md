# Custom Deception Detection System 🎭

A CNN-based deception detection system that identifies facial expressions and stress levels to predict whether a subject is lying. Achieved **88% accuracy** and awarded **3rd Place at NCHU MIS Capstone Project**.

## Demo

📹 [Watch System Demo](https://www.youtube.com/watch?v=37y7yUIQHcY&t=497s) · 📄 [Read Paper Excerpt](https://drive.google.com/file/d/1LTwMkeOzU7VZDmBtiKkzgFVj2eequpfs/view)

## Overview

This system was developed as part of a research project at the Dept. of Management Information Systems, National Chung Hsing University (NCHU). It detects deception by analyzing facial micro-expressions and physiological stress indicators in video footage.

## Key Features

- **CNN-based classification** — detects deception from facial expressions and stress levels
- **Transfer learning** — fine-tuned across multiple real-world datasets to resolve domain mismatch
- **Custom interface** — desktop GUI for real-time video analysis
- **88% accuracy** on test data

## Datasets

- **Training:** "The Real Life Deception" dataset
- **Transfer learning:** "The Airport Security" and "Nothing to Declare" video clips

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras, OpenCV
- **Interface:** HTML/CSS (desktop wrapper)
- **Model format:** `.h5`

## Repository Structure

```
├── Interface Code/     # Desktop GUI interface
├── Model Code/         # CNN model training and fine-tuning
├── build/ & dist/      # Packaged application files
└── mode35new_nohappydata.h5   # Trained model weights
```

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 88% |
| Award | 3rd Place — NCHU MIS Capstone Project (Jun 2022) |
