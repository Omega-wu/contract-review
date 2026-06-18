# -*- coding: utf-8 -*-
import requests
import json
from PIL import Image
import base64
import io
import numpy as np

# === 1?? ¶ÁÈ¡±¾µØÍ¼Ïñ ===
img_path = "/data/doc_review/test_pdf/动土作业票_1.png"
with open(img_path, "rb") as f:
    img_bytes = f.read()

# === 2?? ½«Í¼Æ¬×ªÎª base64£¨·þÎñÒ»°ãÒªÇóÕâÖÖ¸ñÊ½£© ===
img_base64 = base64.b64encode(img_bytes).decode("utf-8")

# === 3?? ¹¹ÔìÇëÇóÌå ===
payload = {
    "file_id": "test_img_001",
    "image": img_base64
}

# === 4?? ·¢ËÍÇëÇó ===
url = "http://127.0.0.1:8002/layout/task"   # ? Èç¹û½Ó¿ÚÂ·¾¶²»Í¬ÇëÌæ»»
resp = requests.post(url, json=payload)

# === 5?? ´òÓ¡½á¹û ===
print("Status:", resp.status_code)
if resp.status_code == 200:
    try:
        result = resp.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Èç¹û½á¹ûÀï°üº¬ boxes£¬¿ÉÒÔ²é¿´ shape
        if isinstance(result, list) and len(result) > 0 and "boxes" in result[0]:
            boxes = np.array(result[0]["boxes"])
            print(" boxes shape:", boxes.shape)
    except Exception as e:
        print("½âÎöÏìÓ¦Ê±³ö´í:", e)
        print("ÏìÓ¦ÎÄ±¾:", resp.text)
else:
    print("ÇëÇóÊ§°Ü:", resp.text)
