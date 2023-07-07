import requests

img_path = "./surprised_cat.jpg"
url = "http://localhost:5000/predict"

with open(img_path, "rb") as f:
    files = {"image": f}
    res = requests.post(url, files=files)
if res.status_code == 200:
    try:
        prediction = res.json()["prediction"]
        print("Prediction :", prediction)
    except Exception as e:
        print("API error:", str(e))
else:
    print("API error", res.text)
