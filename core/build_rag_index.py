import os
import glob
import json
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

def build_index():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
    signs = ["Hello", "iloveyou", "yes", "No", "Thankyou"]
    
    print("Loading CLIP model (this may take a minute)...")
    # Using CLIP model to embed images
    model = SentenceTransformer('clip-ViT-B-32')
    
    embeddings = []
    metadata = []
    
    print("Processing Kaggle dataset images...")
    for sign in signs:
        folder = os.path.join(data_dir, sign)
        if not os.path.exists(folder):
            folder = os.path.join(data_dir, sign.upper())
            if not os.path.exists(folder):
                print(f"Warning: Folder for {sign} not found!")
                continue
                
        images = glob.glob(os.path.join(folder, "*.jpg"))
        if not images:
            print(f"Warning: No images found for {sign}")
            continue
            
        # Use up to 30 images per sign to keep the index lightweight
        images = images[:30]
        
        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                # CLIP expects PIL images
                emb = model.encode(img)
                embeddings.append(emb)
                metadata.append({"label": sign, "path": img_path})
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        print(f"Processed {len(images)} images for {sign}")
        
    if not embeddings:
        print("No embeddings generated. Exiting.")
        return
        
    # Convert to numpy array of float32 for FAISS
    embeddings_matrix = np.array(embeddings).astype('float32')
    
    # L2 normalize for cosine similarity search
    faiss.normalize_L2(embeddings_matrix)
    
    d = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(d) # Inner Product on normalized vectors = Cosine Similarity
    
    index.add(embeddings_matrix)
    
    # Save index and metadata
    output_dir = os.path.dirname(__file__)
    faiss.write_index(index, os.path.join(output_dir, "rag_index.faiss"))
    
    with open(os.path.join(output_dir, "rag_metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    print(f"Successfully built FAISS index with {index.ntotal} embeddings.")

if __name__ == "__main__":
    build_index()
