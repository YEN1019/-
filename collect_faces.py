


import cv2
import os

def collect_faces():
    person_name = input("請輸入人員標籤/姓名 (例如: student01): ").strip()
    if not person_name:
        print("姓名不能為空！")
        return

    save_dir = os.path.join("data", person_name)
    os.makedirs(save_dir, exist_ok=True)

    # 載入 Haar Cascade 模型
    cascade_path = "haarcascade_frontalface_alt.xml"
    if not os.path.exists(cascade_path):
        print(f"錯誤: 找不到 {cascade_path}")
        return
        
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)

    count = 0
    max_samples = 50  # 預設擷取 50 張樣本

    print("開始採集人臉，請面向鏡頭 (按 'q' 鍵可提前結束)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法獲取攝影機畫面")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            
            file_path = os.path.join(save_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)

            # 畫框與計數提示
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Collected: {count}/{max_samples}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Collect Faces", frame)

        if cv2.waitKey(100) & 0xFF == ord('q') or count >= max_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"完成採集！共儲存 {count} 張照片至 {save_dir}")

if __name__ == "__main__":
    collect_faces()
   