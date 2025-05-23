# Multi-Modal RAG Wardrobe App

This repository demonstrates how to build a **multi-modal Retrieval-Augmented Generation (RAG) application** that lets you store and search a collection of clothing items using **images and text** descriptions. Think of it as a digital wardrobe: when you snap a photo of a new item, the app can tell you if something similar already exists in your collection.

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Running the Backend (FastAPI)](#running-the-backend-fastapi)
  - [Running the Frontend (Streamlit)](#running-the-frontend-streamlit)
- [File Structure](#file-structure)
- [Customization and Tips](#customization-and-tips)
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
- **Python 3.11+**
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
  ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
  ```
## Usage
### Running the Backend (FastAPI)
  Start the FastAPI server:

```
uvicorn main:app --reload
```
main:app should point to the location of your FastAPI instance if it’s defined in a file called main.py with app = FastAPI().
Adjust the --port as needed.

### Running the Frontend (Streamlit)
Start Streamlit:
```
streamlit run app.py
```
## File Structure
Below is an example structure you might have in your project:
```graphql
.
├── main.py                # FastAPI code with endpoints
├── app.py                 # Streamlit frontend
├── models/
│   └── clip_model.py      # CLIP model & helper functions
├── requirements.txt
├── comments.json          # JSON with image metadata
├── images/
│   ├── item1.jpg
│   ├── item2.jpg
│   └── ...
└── ...
```
## Customization and Tips
Embedding Weights
In the code, the image and text embeddings are averaged. To bias the search more towards image features or text features, modify the ratio, for example:
```python
combined_emb = (0.7 * (image_emb / norm_image) + 0.3 * (text_emb / norm_text))
```
Hyperparameters
  max_distance: Determines how “strict” or “loose” the similarity threshold is.
  k: Number of top matches to return.
Deployment
  You can deploy FastAPI on services like AWS, Azure, or Heroku.
  Streamlit can also be hosted on Streamlit Cloud or other platforms (e.g., Docker containers).

## License
This project is licensed under the MIT License. Feel free to modify and distribute it as needed.
