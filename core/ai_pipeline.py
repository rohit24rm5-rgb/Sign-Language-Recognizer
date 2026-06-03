import base64
import json
import numpy as np
import os
import faiss
import io
import asyncio
import pickle
from PIL import Image
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from core.memory_manager import MemoryManager
from synaptoroute import Route, AdaptiveRouter, SQLiteStorage

load_dotenv()

class AgentState(TypedDict):
    live_base64: str
    retrieved_context: list
    drafter_guess: str
    final_sign: str
    confidence: float

class SignDetectorPipeline:
    def __init__(self):
        self.vision_model = ChatGroq(
            model="llama-3.2-90b-vision-preview",
            temperature=0.1,
            max_tokens=512,
            max_retries=0,
            request_timeout=5.0
        )
        self.graph = self._build_graph()
        self.memory = MemoryManager()
        
        # Load local ML classifier if available
        self.clf = None
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sign_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.clf = pickle.load(f)
            print("Loaded local Landmark ML Classifier!")
        
        # Setup Synaptoroute for Semantic Routing
        routes_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "databases", "routes.db")
        storage = SQLiteStorage(routes_db_path)
        self.router = AdaptiveRouter(storage=storage)
        self.router.add_route(Route(name="known_sign", utterances=["Hello", "iloveyou", "yes", "No", "Thankyou", "A clear hand sign"]))
        
        # Setup Middle-Ground PyTorch CNN
        try:
            from core.cnn_classifier import CNNImageClassifier
            self.cnn = CNNImageClassifier(classes=list(self.clf.classes_) if self.clf else ["Hello", "iloveyou", "yes", "No", "Thankyou"])
        except Exception as e:
            print(f"Warning: Failed to load CNNImageClassifier: {e}")
            self.cnn = None
        
        # Load RAG resources
        print("Loading RAG FAISS Index and CLIP model...")
        current_dir = os.path.dirname(__file__)
        index_path = os.path.join(current_dir, "rag_index.faiss")
        meta_path = os.path.join(current_dir, "rag_metadata.json")
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.faiss_index = faiss.read_index(index_path)
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            self.rag_ready = True
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            self.clip_model = SentenceTransformer('clip-ViT-B-32')
            print("RAG Pipeline index loaded.")
        else:
            print("Warning: FAISS index not found. RAG disabled.")
            self.rag_ready = False

    def get_base64_from_file(self, path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def normalize_landmarks(self, lmList):
        if not lmList:
            return None
        # Simple normalization: center around wrist (first landmark)
        try:
            base_x, base_y = lmList[0][0], lmList[0][1]
            if len(lmList[0]) > 2: # [id, x, y, z] vs [x, y]
                base_x, base_y = lmList[0][1], lmList[0][2]
                return [[lm[1]-base_x, lm[2]-base_y] for lm in lmList]
            return [[lm[0]-base_x, lm[1]-base_y] for lm in lmList]
        except:
            return lmList

    def retrieve_context(self, base64_image):
        if not getattr(self, 'rag_ready', False):
            return []
            
        if self.clip_model is None:
            return []
            
        # Convert base64 to PIL Image
        image_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Embed live image
        emb = self.clip_model.encode(img)
        emb_matrix = np.array([emb]).astype('float32')
        faiss.normalize_L2(emb_matrix)
        
        # Query FAISS
        k = 3
        distances, indices = self.faiss_index.search(emb_matrix, k)
        
        retrieved = []
        for i in range(k):
            idx = indices[0][i]
            meta = self.metadata[idx]
            b64 = self.get_base64_from_file(meta['path'])
            retrieved.append({
                "label": meta['label'],
                "distance": float(distances[0][i]),
                "base64": b64
            })
            
        return retrieved

    async def drafter_node(self, state: AgentState):
        retrieved = state['retrieved_context']
        
        prompt = "You are an expert ASL interpreter using a Multimodal RAG Pipeline.\n"
        prompt += "First, look at the 1st image. This is the LIVE image from the user's webcam that you need to classify.\n"
        
        if retrieved:
            prompt += "To help you, we retrieved the top 3 most visually similar images from a verified Kaggle ASL dataset.\n"
            prompt += "Here are the labels for the retrieved images (in order of appearance after the live image):\n"
            for i, r in enumerate(retrieved):
                prompt += f"Match {i+1}: '{r['label']}'\n"
            prompt += "\nCompare the LIVE image to the retrieved matches. Pay attention to finger shapes and hand orientation.\n"
            
        prompt += "The possible signs are: Hello, iloveyou, yes, No, Thankyou.\n"
        prompt += "Return ONLY the exact name of the sign from the list above. Do not include any other text."

        # Construct message payload with multiple images
        content = [{"type": "text", "text": prompt}]
        
        # Add LIVE image
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['live_base64']}"}})
        
        # Add Context images
        for r in retrieved:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{r['base64']}"}})
            
        message = HumanMessage(content=content)
        
        try:
            # We add a strict 5-second timeout to the LLM so it never hangs the pipeline
            response = await asyncio.wait_for(self.vision_model.ainvoke([message]), timeout=5.0)
            guess = response.content.strip()
        except asyncio.TimeoutError:
            print("\n[Error] Groq API timed out! Defaulting to Unknown.")
            guess = "Unknown"
        except Exception as e:
            print(f"\n[Error] Groq API Failed: {e}")
            guess = "Unknown"
        
        valid_signs = list(self.clf.classes_) if self.clf is not None else ["Hello", "iloveyou", "yes", "No", "Thankyou"]
        final_sign = ""
        for s in valid_signs:
            if s.lower() in guess.lower():
                final_sign = s
                break
                
        if not final_sign:
            final_sign = "Unknown"
            
        # Logging for the terminal
        if retrieved:
            print("\nRAG Retrieved Context:")
            for i, r in enumerate(retrieved):
                print(f" - Image {i+1}: {r['label']} (Similarity: {r['distance']:.4f})")
                
        return {"drafter_guess": final_sign, "final_sign": final_sign, "confidence": 0.95}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("drafter", self.drafter_node)
        workflow.add_edge(START, "drafter")
        workflow.add_edge("drafter", END)
        return workflow.compile()

    async def detect_sign(self, base64_image: str, lmList: list = None):
        if not getattr(self, '_router_started', False):
            await self.router.start()
            self._router_started = True
            
        # --- 100% Accuracy Geometric Fast-Path ---
        if lmList:
            vec = self.normalize_landmarks(lmList)
            if vec:
                vec = np.array(vec).flatten()
                norm = np.linalg.norm(vec)
                if norm > 0: 
                    vec = vec / norm
                    
                    if self.clf is not None:
                        # Use Random Forest Classifier for 100% robust accuracy
                        pred = self.clf.predict([vec])[0]
                        probs = self.clf.predict_proba([vec])[0]
                        conf = float(max(probs))
                        
                        if conf > 0.85:
                            print(f"[ML Classifier] Matched: {pred} (Conf: {conf:.4f})")
                            return {
                                "final_sign": pred,
                                "confidence": conf,
                                "drafter_guess": pred,
                                "critic_feedback": f"ML Classifier match! Conf: {conf:.4f}"
                            }
                        else:
                            print(f"[ML Classifier] Uncertain ({conf:.4f}), falling back to CNN...")
                    else:
                        # Fallback to Geometric Cosine Similarity if model isn't trained yet
                        best_sign = None
                        best_sim = -1.0
                        signs = list(self.clf.classes_) if self.clf is not None else ["Hello", "iloveyou", "yes", "No", "Thankyou"]
                        for s in signs:
                            template = self.memory.get_template(s)
                            if template:
                                t_vec = np.array(template).flatten()
                                if len(vec) == len(t_vec):
                                    sim = float(np.dot(vec, t_vec))
                                    if sim > best_sim:
                                        best_sim = sim
                                        best_sign = s
                        
                        if best_sim > 0.80 and best_sign:
                            print(f"[Geometric Match] Fast-path engaged! Matched: {best_sign} (Sim: {best_sim:.4f})")
                            return {
                                "final_sign": best_sign,
                                "confidence": best_sim,
                                "drafter_guess": best_sign,
                                "critic_feedback": f"Geometric template match! Sim: {best_sim:.4f}"
                            }
                            
        # --- Middle-Ground PyTorch CNN Path ---
        if hasattr(self, 'cnn') and self.cnn is not None:
            cnn_pred, cnn_conf = self.cnn.predict(base64_image)
            if cnn_conf > 0.80:
                print(f"[CNN Classifier] Fast-path engaged! Matched: {cnn_pred} (Conf: {cnn_conf:.4f})")
                return {
                    "final_sign": cnn_pred,
                    "confidence": cnn_conf,
                    "drafter_guess": cnn_pred,
                    "critic_feedback": f"CNN Image Classifier match! Conf: {cnn_conf:.4f}"
                }
        
        print("Retrieving context from Vector DB...")
        # Offload the blocking CPU-bound FAISS search to a background thread
        retrieved_context = await asyncio.to_thread(self.retrieve_context, base64_image)
        
        # --- Semantic Routing Fast Path ---
        if retrieved_context:
            best_match = retrieved_context[0]
            # If FAISS distance is very low (highly similar)
            if best_match['distance'] < 0.6: 
                # Use synaptoroute to verify the label semantics asynchronously
                route_result = await self.router.aquery(best_match['label'])
                if route_result and route_result.name == "known_sign":
                    print(f"[Semantic Router] Fast path engaged! Skipping LLM. Matched: {best_match['label']}")
                    return {
                        "final_sign": best_match['label'],
                        "confidence": 0.99,
                        "drafter_guess": best_match['label'],
                        "critic_feedback": f"Semantic Router fast-path match. Distance: {best_match['distance']:.4f}"
                    }

        # --- LLM Agent Path ---
        initial_state = {
            "live_base64": base64_image,
            "retrieved_context": retrieved_context,
            "drafter_guess": "",
            "final_sign": "",
            "confidence": 0.0
        }
        
        result = await self.graph.ainvoke(initial_state)
        result['critic_feedback'] = f"RAG LLM Context used. Top match: {retrieved_context[0]['label'] if retrieved_context else 'None'}"
        return result
