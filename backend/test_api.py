import urllib.request
from urllib.error import HTTPError
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

data = json.dumps({"name": "Dr. Rajveer Rai", "email": "rajveerrai2807@gmail.com", "password": "123", "medical_license": "MD-1234"}).encode('utf-8')
req = urllib.request.Request("http://127.0.0.1:8000/api/auth/register", data=data, headers={'Content-Type': 'application/json'})

try:
    response = urllib.request.urlopen(req, timeout=5)
    print(response.read().decode('utf-8'))
except HTTPError as e:
    print("HTTP Error:", e.code)
    print("Body:", e.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
