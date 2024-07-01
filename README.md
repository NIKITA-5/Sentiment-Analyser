# Sentiment Analysis and Text Classification

This project implements basic sentiment analysis and text classification using natural language processing (NLP) techniques. It includes text preprocessing, emotion detection, and classification using a Naive Bayes classifier.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Files](#files)

## Introduction
The goal of this project is to clean and process text data, classify it into predefined categories, and analyze the emotions expressed in the text. This project demonstrates basic text preprocessing, vectorization, and classification using machine learning techniques.

## Features
- Text preprocessing: converting to lowercase and removing punctuation.
- Stop words removal.
- Emotion detection from text.
- Text classification using a Naive Bayes classifier.
- Visualization of emotion analysis.

## Requirements
- Python 3.x
- NumPy
- scikit-learn
- matplotlib

## Usage
1. Prepare your text data in a text file named `read.txt`.
2. Prepare the emotion mappings in a file named `emotion.txt`.
3. Run the main script:
   ```bash
   python main.py
   ```
4. Enter a sentence when prompted to classify and analyze its sentiment.

## Files
- `main.py`: The main script to run the text classification and emotion analysis.
- `read.txt`: The input text file.
- `emotion.txt`: The emotion mappings file.
- `graph.png`: The output graph showing the emotion analysis.
