# Cascaded Sign Language Recognizer

An offline, low-latency computer vision pipeline that translates physical hand gestures into text.

## Architecture

This project utilizes a multi-stage classification pipeline to ensure high accuracy while maintaining sub-150ms inference times on CPU:

1. **Geometric Hand Tracking**: Extracts 21 3D spatial coordinates from live video feeds using MediaPipe.
2. **Initial Classification**: A lightweight Random Forest model trained on hand landmarks provides immediate inference (~5ms latency) for clear, unambiguous signs.
3. **Deep Learning Cascade**: If the geometric model outputs a confidence score of $\le 85\%$, the raw image frame is dynamically routed to a PyTorch Convolutional Neural Network. This ResNet18 model was fine-tuned using transfer learning and aggressive data augmentation to handle varying lighting conditions and angles.
4. **Semantic Routing**: A Retrieval-Augmented Generation (RAG) module utilizing a FAISS vector index performs cosine similarity searches to ensure physical sign predictions map logically to recognized linguistic constructs, preventing hallucinated outputs.

## Project Structure

```text
├── api/                  # FastAPI endpoints
├── core/                 # Core AI logic (pipeline, RAG indexing)
├── services/             # Live webcam service runners
├── scripts/              # Data collection and training pipelines
├── tests/                # Verification and stress tests
├── benchmarks/           # Benchmarking tools
├── models/               # Pre-trained CNN weights and RF models
└── databases/            # SQLite state and routing configurations
```

## Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/rohit24rm5-rgb/Sign-Language-Recognizer.git
cd Sign-Language-Recognizer
```

2. Initialize a virtual environment and install dependencies:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

3. Run the live camera service:
```bash
python services/vision_service.py
```

## Performance Metrics

- **CNN Validation Accuracy**: 100.00%
- **End-to-End Latency**: ~110 milliseconds (CPU inference)
