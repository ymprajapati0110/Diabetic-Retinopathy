import urllib.request

req = urllib.request.Request("http://127.0.0.1:8000/api/auth/register", method="OPTIONS")
req.add_header("Origin", "http://localhost:3000")
req.add_header("Access-Control-Request-Method", "POST")

try:
    response = urllib.request.urlopen(req)
    print("OPTIONS OK")
    print(response.getcode())
except Exception as e:
    print("OPTIONS failed:", e)
