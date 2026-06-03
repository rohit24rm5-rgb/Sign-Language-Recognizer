import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import os, requests
load_dotenv()
r = requests.get('https://api.groq.com/openai/v1/models', headers={'Authorization': f"Bearer {os.getenv('GROQ_API_KEY')}"})
data = r.json().get('data', [])
print("All available models:")
for m in data:
    print(m['id'])
