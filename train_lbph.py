import cv2
import os
import json
import numpy as np

def train_model():
    data_dir = "data"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_dir):
        print("錯誤: 找不到 data/ 目錄，請先執行 collect_faces.py 採集資料！")
        return

    faces = []
    labels = []
    label_map = {}
    current_id = 0

    print("正在載入人臉圖像進行訓練...")

    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            person_dir = os.path.join(root, dir_name)
            if dir_name not in label_map:
                label_map[current_id] = dir_name
                target_id = current_id
                current_id += 1
            else:
                target_id = [k for k, v in label_map.items() if v == dir_name][0]

            for file_name in os.listdir(person_dir):
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, file_name)
                    
                    # 💡 使用 np.fromfile 與 cv2.imdecode 解決 Windows 中文路徑無法讀取的問題
                    try:
                        img_array = np.fromfile(img_path, dtype=np.uint8)
                        gray_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                        if gray_img is not None:
                            faces.append(gray_img)
                            labels.append(target_id)
                    except Exception as e:
                        print(f"無法讀取圖片 {img_path}: {e}")

    if len(faces) == 0:
        print("沒有找到可用於訓練的照片！")
        return

    # 創建 LBPH 辨識器並進行訓練
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # 儲存訓練模型與標籤 JSON
    model_path = os.path.join(models_dir, "lbph_model.yml")
    labels_path = os.path.join(models_dir, "labels.json")

    recognizer.save(model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)

    print(f"訓練成功完成！")
    print(f"- 模型檔已儲存至: {model_path}")
    print(f"- 標籤檔已儲存至: {labels_path}")

if __name__ == "__main__":
    train_model()