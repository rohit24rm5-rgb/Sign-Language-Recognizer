import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import urllib.request
import urllib.error
import urllib.parse
import concurrent.futures
import base64
import json
import uuid

# We will test the core pipeline directly instead of FastAPI to isolate the system pipeline performance,
# as well as avoiding multipart/form-data complexity with urllib.
# Actually, the user wants system issues, testing the API is better.
# Let's write a simple multipart uploader using urllib to hit the FastAPI endpoint.

import io
import cv2
import numpy as np

API_URL = "http://127.0.0.1:8000/detect"

def create_dummy_image():
    # Create a dummy white image 300x300
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    is_success, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()

def send_request(req_id):
    start_time = time.time()
    try:
        boundary = uuid.uuid4().hex
        headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
        
        body = io.BytesIO()
        body.write(f'--{boundary}\r\n'.encode('utf-8'))
        body.write(b'Content-Disposition: form-data; name="file"; filename="dummy.jpg"\r\n')
        body.write(b'Content-Type: image/jpeg\r\n\r\n')
        body.write(create_dummy_image())
        body.write(f'\r\n--{boundary}--\r\n'.encode('utf-8'))
        
        req = urllib.request.Request(API_URL, data=body.getvalue(), headers=headers, method='POST')
        
        with urllib.request.urlopen(req) as response:
            status = response.status
            text = response.read().decode('utf-8')
            
        latency = time.time() - start_time
        return req_id, status, latency, text
    except urllib.error.HTTPError as e:
        latency = time.time() - start_time
        return req_id, e.code, latency, str(e.read())
    except Exception as e:
        latency = time.time() - start_time
        return req_id, 500, latency, str(e)

def run_stress_test(num_requests=10, max_workers=2):
    print(f"Starting stress test with {num_requests} requests, max {max_workers} concurrent workers...")
    latencies = []
    successes = 0
    failures = 0
    
    start_total_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(send_request, i): i for i in range(num_requests)}
        for future in concurrent.futures.as_completed(futures):
            req_id, status, latency, text = future.result()
            latencies.append(latency)
            if status == 200:
                successes += 1
                print(f"Request {req_id} SUCCESS - Latency: {latency:.2f}s")
            else:
                failures += 1
                print(f"Request {req_id} FAILED ({status}) - Latency: {latency:.2f}s - {text[:100]}")
                
    total_time = time.time() - start_total_time
    
    print("\n--- STRESS TEST RESULTS ---")
    print(f"Total Requests: {num_requests}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Throughput: {num_requests / total_time:.2f} req/s")
    if latencies:
        print(f"Min Latency: {min(latencies):.2f}s")
        print(f"Max Latency: {max(latencies):.2f}s")
        print(f"Avg Latency: {sum(latencies)/len(latencies):.2f}s")
    
    with open("stress_test_results.txt", "w") as f:
        f.write(f"Total Requests: {num_requests}\n")
        f.write(f"Successful: {successes}\n")
        f.write(f"Failed: {failures}\n")
        f.write(f"Total Time: {total_time:.2f}s\n")
        f.write(f"Throughput: {num_requests / total_time:.2f} req/s\n")
        if latencies:
            f.write(f"Min Latency: {min(latencies):.2f}s\n")
            f.write(f"Max Latency: {max(latencies):.2f}s\n")
            f.write(f"Avg Latency: {sum(latencies)/len(latencies):.2f}s\n")

if __name__ == "__main__":
    # Small test because Groq API has rate limits and Faiss model might load multiple times if not careful
    run_stress_test(num_requests=5, max_workers=2)
