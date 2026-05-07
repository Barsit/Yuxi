"""手动测试自定义嵌入/重排序模型 API"""
import httpx

BASE = "http://localhost:5050"

# 尝试从 .env 读取管理员账户
import os
from pathlib import Path

env_path = Path("/app/.env")
admin_user = "admin"
admin_pass = "admin123456"

if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("YUXI_SUPER_ADMIN_NAME="):
            admin_user = line.split("=", 1)[1].strip()
        elif line.startswith("YUXI_SUPER_ADMIN_PASSWORD="):
            admin_pass = line.split("=", 1)[1].strip()

# 登录
resp = httpx.post(f"{BASE}/api/auth/token", data={
    "username": admin_user,
    "password": admin_pass,
})
print(f"Login: {resp.status_code}")
if resp.status_code != 200:
    print(f"Login failed: {resp.text[:200]}")
    exit(1)

token = resp.json().get("access_token", "")
headers = {"Authorization": f"Bearer {token}"}

# 1. 测试 GET 自定义嵌入模型
r = httpx.get(f"{BASE}/api/system/custom-embed-models", headers=headers)
print(f"\n1. GET custom-embed-models: {r.status_code}")
assert r.status_code == 200
data = r.json()
assert "models" in data
print(f"   -> models count: {len(data['models'])}")

# 2. 测试添加自定义嵌入模型
r = httpx.post(f"{BASE}/api/system/custom-embed-models", headers=headers, json={
    "model_id": "custom/test-embed",
    "model_data": {
        "name": "Test Embed",
        "dimension": 768,
        "base_url": "https://api.example.com/v1/embeddings",
        "api_key": "test-key-123"
    }
})
print(f"2. POST custom-embed-model (add): {r.status_code}")
assert r.status_code == 200

# 3. 验证添加后在列表中
r = httpx.get(f"{BASE}/api/system/custom-embed-models", headers=headers)
print(f"3. GET after add: {r.status_code}")
assert r.status_code == 200
assert "custom/test-embed" in r.json()["models"]
print("   -> custom/test-embed found in list")

# 4. 测试更新自定义嵌入模型
r = httpx.put(f"{BASE}/api/system/custom-embed-models", headers=headers, json={
    "model_id": "custom/test-embed",
    "model_data": {
        "name": "Test Embed Updated",
        "dimension": 1024,
        "base_url": "https://api.example.com/v1/embeddings",
        "api_key": "test-key-456"
    }
})
print(f"4. PUT custom-embed-model (update): {r.status_code}")
assert r.status_code == 200

# 5. 删除自定义嵌入模型
r = httpx.delete(f"{BASE}/api/system/custom-embed-models?model_id=custom%2Ftest-embed", headers=headers)
print(f"5. DELETE custom-embed-model: {r.status_code}")
assert r.status_code == 200

# 6. 验证已删除
r = httpx.get(f"{BASE}/api/system/custom-embed-models", headers=headers)
assert "custom/test-embed" not in r.json()["models"]
print("6. Verified: custom/test-embed deleted")

# 7. 测试添加自定义重排序模型
r = httpx.post(f"{BASE}/api/system/custom-reranker-models", headers=headers, json={
    "model_id": "custom/test-reranker",
    "model_data": {
        "name": "Test Reranker",
        "base_url": "https://api.example.com/v1/rerank",
        "api_key": "test-key-789"
    }
})
print(f"7. POST custom-reranker-model (add): {r.status_code}")
assert r.status_code == 200

# 8. 验证添加后在列表中
r = httpx.get(f"{BASE}/api/system/custom-reranker-models", headers=headers)
assert "custom/test-reranker" in r.json()["models"]
print("8. Verified: custom/test-reranker in list")

# 9. 删除重排序模型
r = httpx.delete(f"{BASE}/api/system/custom-reranker-models?model_id=custom%2Ftest-reranker", headers=headers)
print(f"9. DELETE custom-reranker-model: {r.status_code}")
assert r.status_code == 200

# 10. 验证 config dump 中包含 custom 字段
r = httpx.get(f"{BASE}/api/system/config", headers=headers)
cfg = r.json()
embed_models = cfg.get("embed_model_names", {})
if embed_models:
    first_key = list(embed_models.keys())[0]
    first_model = embed_models[first_key]
    assert "custom" in first_model
    print(f"10. Config embed_model_names has custom field: True")
    print(f"    First model '{first_key}': custom={first_model['custom']}")

# 11. 验证 reranker_names 也有 custom 字段
reranker_models = cfg.get("reranker_names", {})
if reranker_models:
    first_rk = list(reranker_models.keys())[0]
    assert "custom" in reranker_models[first_rk]
    print(f"11. Config reranker_names has custom field: True")

print("\n=== All tests passed! ===")
