---
title: Dawai AI Engine
emoji: ⚕️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Dawai Backend System - Artificial Intelligence and API Gateway

## Overview
This repository contains the backend infrastructure and Artificial Intelligence logic for "Dawai", a healthcare application tailored for the elderly. The system provides core functionalities including computer vision for medicine recognition and a natural language processing engine for the symptom-checking assistant.

## System Architecture

### 1. Computer Vision Module (Optical Character Recognition)
The system employs a dual-engine OCR pipeline to ensure high accuracy and performance:
- **Engine:** RapidOCR optimized via ONNXRuntime.
- **Implementation:** The pipeline utilizes parallel processing (ThreadPoolExecutor) to run dedicated Arabic and English recognition models simultaneously. This resolves the bilingual text extraction problem without relying on external or cloud-based APIs, ensuring patient privacy and zero-latency offline performance.

### 2. Natural Language Processing (Chatbot Assistant)
The symptom-checker chatbot is built to understand colloquial Egyptian Arabic using advanced semantic analysis.
- **Engine:** CAMeL-BERT (SentenceTransformers).
- **Contextual Awareness:** The system implements a stateful context mechanism. If a user asks for an alternative form (e.g., "syrup" instead of "tablets") after a recommendation, the engine queries the local database for medicines with matching therapeutic uses and returns the appropriate alternative.
- **Graceful Degradation:** To maintain system stability during unrecognized or highly misspelled inputs, the engine utilizes a Fuzzy String Matching fallback mechanism (RapidFuzz), activating only if the NLP confidence score falls below the 82% threshold.

## API Endpoints

### 1. POST /scan
- **Description:** Receives a captured image of a medicine box, processes the image through the OCR pipeline, normalizes the extracted text, and queries the database for therapeutic uses and side effects.
- **Input:** multipart/form-data (image file)
- **Output:** JSON object containing the matched medicine details (English Name, Arabic Name, Uses, Side Effects) and the confidence score.

### 2. POST /chat
- **Description:** Receives user symptoms in natural language, performs semantic similarity checks against a trained symptom dataset, and recommends appropriate over-the-counter medication.
- **Input:** JSON object containing the user's message and the ID of the previously recommended medicine (if any).
- **Output:** JSON object containing the textual response and the new recommended medicine ID.

## Setup Instructions
1. Initialize a Python virtual environment.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Start the FastAPI server by executing `uvicorn dawai_api:app --host 0.0.0.0 --port 8000`.
4. The system will load the ONNX models and the Pandas database into memory upon startup.
