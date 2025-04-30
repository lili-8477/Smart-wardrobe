import io
import os
import json
import uuid
import numpy as np
import faiss
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.staticfiles import StaticFiles
from PIL import Image
from typing import List
import torch
from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, AutoModelForObjectDetection
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yoloprocessor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
yolomodel = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia").to(device)

def segment_clothing(image_path):
    """Detect fashion items using YOLOS Fashionpedia"""
    # Load and process image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    inputs = yoloprocessor(images=image, return_tensors="pt").to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = yolomodel(**inputs)
    
    # Post-process results
    results = yoloprocessor.post_process_object_detection(
        outputs,
        threshold=0.5,
        target_sizes=[(h, w)]  # Original image dimensions
    )[0]
    
    components = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        components.append({
            "category": yolomodel.config.id2label[label.item()],
            "confidence": score.item(),
            "bbox": box.tolist(),  # [xmin, ymin, xmax, ymax]
            "mask": None
        })
    
    return components, image


@app.post("/analyze-outfit")
async def analyze_outfit(image: UploadFile = File(...)):
    # Save temporary file
    temp_path = f"segments/outfits/temp_{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(await image.read())
    
    # Get components using YOLOS
    components, base_image = segment_clothing(temp_path)
    
    output = []
    excluded_categories = {
            'headband, head covering, hair accessory','hood', 'collar', 'lapel', 'epaulette', 'sleeve',
            'pocket', 'neckline', 'buckle', 'zipper', 'applique',
            'bead', 'bow', 'flower', 'fringe', 'ribbon',
            'rivet', 'ruffle', 'sequin', 'tassel'
        }
    category_tracker = {}  # Track highest confidence per category
        
    for comp in components:

        if comp['confidence'] >= 0.75 and comp['category'] not in excluded_categories:
            current_category = comp['category']
            current_conf = comp['confidence']

            # Update if category not seen or higher confidence than existing
            if current_category not in category_tracker or \
                current_conf > category_tracker[current_category]['confidence']:
                
                # Crop and save segment
                x1, y1, x2, y2 = map(int, comp['bbox'])
                cropped = base_image[y1:y2, x1:x2]
                seg_path = f"segments/{uuid.uuid4()}.jpg"
                cv2.imwrite(seg_path, cropped)

                # Update tracker with complete component data
                category_tracker[current_category] = {
                    "category": current_category,
                    "confidence": float(current_conf),
                    "segment_path": seg_path
                }

    # Convert tracker dict values to output list
    output = list(category_tracker.values())
    
    return output

# Configuration
IMAGE_DIR = "images"
SEG_DIR = "segments"
COMMENTS_FILE = "image_comments.json"
EMBEDDING_DIM = 512  # CLIP base model dimension
INDEX_PATH = "faiss_index.index"

# Create directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize FAISS index and metadata store
index = faiss.IndexFlatL2(EMBEDDING_DIM)
metadata = []
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open("metadata.json", "r") as f:
        metadata = json.load(f)

# Serve images statically
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Serve segments statically
app.mount("/segments", StaticFiles(directory=SEG_DIR), name="segments")

def process_image_text_pair(image_path: str, text: str):
    """Process image and text to generate embeddings"""
    image = Image.open(image_path)
    
    # Process image
    image_inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)
    
    # Process text
    text_inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        # Get image embedding
        image_emb = model.get_image_features(**image_inputs)
        image_emb = image_emb.cpu().numpy()[0]
        image_emb = image_emb / np.linalg.norm(image_emb)
        
        # Get text embedding
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb.cpu().numpy()[0]
        text_emb = text_emb / np.linalg.norm(text_emb)
        
        # Combined embedding (for image+text search)
        combined_emb = (image_emb + text_emb) / 2
    
    return {
        "image_emb": image_emb.astype('float32'),
        "text_emb": text_emb.astype('float32'),
        "combined_emb": combined_emb.astype('float32')
    }

@app.post("/add-item")
async def add_item(
    image: UploadFile = File(...),
    comment: str = Form(...),
    tags: str = Form(""),
):
    # Save image file
    filename = image.filename
    image_path = os.path.join(IMAGE_DIR, filename)
    with open(image_path, "wb") as f:
        f.write(await image.read())
    
    # Process tags
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    
    # Update comments data with new structure
    comments_data = {}
    if os.path.exists(COMMENTS_FILE):
        with open(COMMENTS_FILE) as f:
            comments_data = json.load(f)
    
    comments_data[filename] = {
        "comment": comment,
        "tags": tag_list
    }
    
    with open(COMMENTS_FILE, "w") as f:
        json.dump(comments_data, f, indent=2)
    
    # Generate embeddings using combined comment + tags
    combined_text = f"{comment} {' '.join(tag_list)}"
    embeddings = process_image_text_pair(image_path, combined_text)
    
    # Add to FAISS index
    index.add(np.array([embeddings["combined_emb"]]))
    
    # Update metadata
    metadata.append({
        "filename": filename,
        "comment": comment,
        "tags": tag_list,
        "embeddings": {
            "image": embeddings["image_emb"].tolist(),
            "text": embeddings["text_emb"].tolist()
        }
    })
    
    # Persist changes
    faiss.write_index(index, INDEX_PATH)
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return {"message": "Item added successfully"}

def check_index_state():
    """Check the current state of the index and metadata"""
    print("\n=== Index State Check ===")
    print(f"Number of items in metadata: {len(metadata)}")
    if len(metadata) > 0:
        print("\nSample item structure:")
        sample = metadata[0]
        print(f"Keys in metadata: {sample.keys()}")
        if "embeddings" in sample:
            print(f"Embedding keys: {sample['embeddings'].keys()}")
    print("======================\n")

@app.post("/rebuild-index")
async def rebuild_index():
    global index, metadata
    print("\nStarting index rebuild...")
    
    # Reset index and metadata
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = []
    
    if os.path.exists(COMMENTS_FILE):
        with open(COMMENTS_FILE) as f:
            comments_data = json.load(f)
        
        print(f"Found {len(comments_data)} items in comments file")
        
        for filename, data in comments_data.items():
            # Handle legacy string format
            if isinstance(data, str):
                comment = data
                tags = []
            else:
                comment = data.get("comment", "")
                tags = data.get("tags", [])
            
            image_path = os.path.join(IMAGE_DIR, filename)
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {filename}")
                continue
                
            print(f"Processing {filename}...")
            combined_text = f"{comment} {' '.join(tags)}"
            
            try:
                emb_data = process_image_text_pair(image_path, combined_text)
                
                # Add to FAISS index
                index.add(np.array([emb_data["combined_emb"]]))
                
                # Update metadata with new structure
                metadata.append({
                    "filename": filename,
                    "comment": comment,
                    "tags": tags,
                    "embeddings": {
                        "image": emb_data["image_emb"].tolist(),
                        "text": emb_data["text_emb"].tolist()
                    }
                })
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        if metadata:
            print(f"\nSaving {len(metadata)} items to index and metadata...")
            faiss.write_index(index, INDEX_PATH)
            with open("metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            print("Save complete")
        else:
            print("No items were processed successfully")
    
    # Check the state after rebuilding
    check_index_state()
    return {"message": "Index rebuilt successfully"}

# Add check at startup
check_index_state()

@app.post("/search")
async def search_image(
    image: UploadFile = File(...),
    comment: str = Form(""),
    k: int = Query(3, ge=1, le=50),
    max_distance: float = Query(0.4, description="Maximum L2 distance threshold")
) -> List[dict]:
    """Search using image + optional text query"""
    # Process query image
    img = Image.open(io.BytesIO(await image.read()))
    
    if comment:  # Both image and text search
        inputs = processor(
            text=[comment], 
            images=img,
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        with torch.no_grad():
            features = model(**inputs)
            image_emb = features.image_embeds.cpu().numpy()[0]
            text_emb = features.text_embeds.cpu().numpy()[0]
        
        # Same combination as storage
        query_emb = (image_emb / np.linalg.norm(image_emb) + 
                    text_emb / np.linalg.norm(text_emb)) / 2
    else:  # Image-only search
        inputs = processor(
            images=img,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            image_emb = model.get_image_features(**inputs).cpu().numpy()[0]
        
        # Only use normalized image embedding
        query_emb = image_emb / np.linalg.norm(image_emb)
    
    query_emb = query_emb.astype('float32').reshape(1, -1)
    distances, indices = index.search(query_emb, k)
    print(distances)
    
    results = [
        metadata[idx]
        for d, idx in zip(distances[0], indices[0])
        if d <= max_distance and idx < len(metadata)
    ]
    
    return results

@app.post("/search-by-text")
async def search_by_text(
    text: str = Form(...),
    k: int = Query(3, ge=1, le=50),
    max_distance: float = Query(1.0, description="Maximum L2 distance threshold")
) -> List[dict]:
    """Search for items using text description only"""
    print(f"Searching for text: {text}")
    print(f"Number of items in metadata: {len(metadata)}")
    
    # Generate text embedding
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
        text_emb = text_emb.cpu().numpy()[0]
        text_emb = text_emb / np.linalg.norm(text_emb)
    
    print("Generated text embedding shape:", text_emb.shape)
    
    # Search in metadata using text embeddings
    results = []
    for item in metadata:
        if "embeddings" in item and "text" in item["embeddings"]:
            stored_text_emb = np.array(item["embeddings"]["text"])
            distance = np.linalg.norm(text_emb - stored_text_emb)
            print(f"Item {item['filename']} distance: {distance}")
            if distance <= max_distance:
                results.append({
                    "filename": item["filename"],
                    "comment": item["comment"],
                    "tags": item["tags"],
                    "distance": float(distance),
                    "similarity": float(1 - distance)
                })
    
    print(f"Found {len(results)} results")
    
    # Sort by distance (ascending) and return top k
    results.sort(key=lambda x: x["distance"])
    return results[:k]

@app.get("/items")
async def get_all_items():
    """Get all items with image URLs, comments, and tags"""
    if not os.path.exists(COMMENTS_FILE):
        return []
    
    with open(COMMENTS_FILE, "r") as f:
        comments_data = json.load(f)
    
    items = []
    for filename, data in comments_data.items():
        # Verify image actually exists
        image_path = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(image_path):
            continue
        
        if isinstance(data, str):
            comment = data
            tags = []
        else:
            comment = data.get("comment", "")
            tags = data.get("tags", [])
        
        items.append({
            "filename": filename,
            "image_url": f"/images/{filename}",
            "comment": comment,
            "tags": tags
        })
    
    return items

@app.delete("/items/{filename}")
async def delete_item(filename: str):
    try:
        # Remove image file
        image_path = os.path.join(IMAGE_DIR, filename)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Remove from comments data
        with open(COMMENTS_FILE, "r") as f:
            comments_data = json.load(f)
        del comments_data[filename]
        with open(COMMENTS_FILE, "w") as f:
            json.dump(comments_data, f, indent=2)
        
        # Remove from metadata and save
        global metadata
        metadata = [item for item in metadata if item['filename'] != filename]
        with open("metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Rebuild index to remove embeddings
        index.reset()
        if metadata:
            embeddings = [process_image_text_pair(
                os.path.join(IMAGE_DIR, item['filename']),
                f"{item['comment']} {' '.join(item['tags'])}"
            ) for item in metadata]
            index.add(np.array(embeddings))
            faiss.write_index(index, INDEX_PATH)
        
        return {"message": "Item deleted successfully"}
    
    except Exception as e:
        return {"error": str(e)}