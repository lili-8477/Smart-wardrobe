# Multi-Modal RAG Wardrobe App

This repository demonstrates how to build a **multi-modal Retrieval-Augmented Generation (RAG) application** that lets you store and search a collection of clothing items using **images and text** descriptions. Think of it as a digital wardrobe: when you snap a photo of a new item, the app can tell you if something similar already exists in your collection.

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Running the Backend (FastAPI)](#running-the-backend-fastapi)
  - [Rebuilding the FAISS Index](#rebuilding-the-faiss-index)
  - [Running the Frontend (Streamlit)](#running-the-frontend-streamlit)
- [File Structure](#file-structure)
- [Customization and Tips](#customization-and-tips)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)
- [License](#license)

---

## Features
1. **Image + Text Embeddings**: Uses [OpenAI’s CLIP ViT Model](https://github.com/openai/CLIP) to generate embeddings for both images and text, unifying them in the same feature space.
2. **Efficient Similarity Search**: Leverages [FAISS](https://github.com/facebookresearch/faiss) for fast similarity lookups.
3. **RESTful API**: Built with [FastAPI](https://fastapi.tiangolo.com/), so you can integrate it with other services or frontends.
4. **User-Friendly Frontend**: A minimal [Streamlit](https://streamlit.io/) app for capturing or uploading images, browsing your wardrobe, and querying for similar items.

---

## How It Works
1. **Data Storage**: Clothing images are placed in an `images` folder, and their metadata (comments, tags, etc.) lives in a JSON file (e.g., `comments.json`).
2. **Embedding**: Each image is processed through the CLIP model to create an embedding vector. If you have text descriptions or tags, those are embedded as well, then averaged (or weighted) with the image embedding.
3. **Indexing**: Embeddings get stored in a FAISS index for quick nearest-neighbor searches.
4. **Search**: When a user uploads a new image (plus optional text), the system embeds the query, compares it in the FAISS index, and retrieves the top-matching items.

---

## Tech Stack
- **Python 3.8+**
- **[PyTorch](https://pytorch.org/)** (for running the CLIP model)
- **[Transformers](https://github.com/huggingface/transformers)** (for CLIP)
- **[FAISS](https://github.com/facebookresearch/faiss)**
- **[FastAPI](https://fastapi.tiangolo.com/)**
- **[Streamlit](https://streamlit.io/)**
- **[Requests](https://docs.python-requests.org/)** (for frontend-backend communication)

---

## Setup and Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/aidenkoh801/multimodal-rag.git
   cd multimodal-rag

2. **Create a Virtual Environment (optional but recommended)**
python -m venv venv
source venv/bin/activate  # Linux/Mac
# On Windows: venv\Scripts\activate