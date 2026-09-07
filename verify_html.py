
from fastapi.testclient import TestClient
from saas_web import app

client = TestClient(app)
response = client.get("/")
print(response.text.find("aria-invalid"))
