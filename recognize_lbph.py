import cv2
import os
import json

def recognize_faces():
    cascade_path = "haarcascade_frontalface_alt.xml"
    model_path = os.path.join("models", "lbph_model.yml")
    labels_path = os.path.join("models", "labels.json")

    # 檢查相關模型檔案是否存在
    if not (os.path.exists(cascade_path) and os.path.exists(model_path) and os.path.exists(labels_path)):
        print("錯誤: 缺少模型檔案或標籤檔案，請先執行訓練程序！")
        return

    # 載入 Haar Cascade 與 LBPH 模型
    face_cascade = cv2.CascadeClassifier(cascade_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    with open(labels_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
        # JSON 鍵預設讀取為字串，轉換回整數
        label_map = {int(k): v for k, v in label_map.items()}

    THRESHOLD = 70.0  # LBPH 距離閾值 (數值越小代表越相似，高於此值顯示 Unknown)
    cap = cv2.VideoCapture(0)

    print("開始即時人臉辨識 (按 'q' 鍵退出)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法獲取攝影機畫面")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))

            # 特徵分析與距離比對
            label_id, distance = recognizer.predict(face_roi)

            # 判斷距離是否小於等於閾值
            if distance <= THRESHOLD and label_id in label_map:
                name = label_map[label_id]
                text = f"{name} ({round(distance, 1)})"
                color = (0, 255, 0)
            else:
                text = f"Unknown ({round(distance, 1)})"
                color = (0, 0, 255)

            # 繪製外框與姓名文字
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("LBPH Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()