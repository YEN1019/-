import time
import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe 手部偵測模組
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils

# 開啟攝影機 (0 代表預設網路攝影機)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 建立獨立畫布 (np.zeros) 用來記錄軌跡
canvas = None

# 畫筆顏色設定 (BGR 格式)
colors = {
    "BLUE": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
}
current_color = colors["BLUE"]
brush_thickness = 5

# 記錄上一個點的位置 (用於畫線)
prev_x, prev_y = 0, 0

# 用於計算 FPS
prev_time = 0

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("無法讀取攝影機畫面。")
    break

  # 左右鏡像翻轉，讓操作更直覺
  frame = cv2.flip(frame, 1)
  h, w, c = frame.shape

  # 初始化畫布（如果尚未建立）
  if canvas is None:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

  # 轉換色彩空間為 RGB 給 MediaPipe 使用
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  result = hands.process(rgb_frame)

  gesture_text = "None"

  if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
      # 繪製手部骨架
      mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

      # 取得關鍵點座標
      landmarks = hand_landmarks.landmark

      # 任務 2：取得食指指尖 Landmark 8 與中指指尖 Landmark 12
      index_finger_tip = landmarks[8]
      middle_finger_tip = landmarks[12]

      ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
      mx, my = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

      # 任務 3：判斷食指與中指的伸直狀態（以指尖與指根 Y 座標簡單判定）
      index_is_up = index_finger_tip.y < landmarks[6].y  # 食指伸直
      middle_is_up = middle_finger_tip.y < landmarks[10].y  # 中指伸直

      # 判定手勢與動作
      if index_is_up and not middle_is_up:
        gesture_text = "Drawing (Index Up)"
        # 任務 4：使用 OpenCV cv2.line() 畫出移動軌跡
        if prev_x == 0 and prev_y == 0:
          prev_x, prev_y = ix, iy

        cv2.line(
            canvas,
            (prev_x, prev_y),
            (ix, iy),
            current_color,
            brush_thickness,
        )
        prev_x, prev_y = ix, iy

      elif index_is_up and middle_is_up:
        gesture_text = "Selection Mode (Two Fingers)"
        prev_x, prev_y = 0, 0  # 斷開軌跡

        # 頂部選單按鈕點擊判定
        if iy < 100:
          if 50 < ix < 200:  # CLEAR 按鈕
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
          elif 250 < ix < 400:  # BLUE
            current_color = colors["BLUE"]
          elif 450 < ix < 600:  # GREEN
            current_color = colors["GREEN"]
          elif 650 < ix < 800:  # RED
            current_color = colors["RED"]
          elif 850 < ix < 1000:  # YELLOW
            current_color = colors["YELLOW"]
      else:
        gesture_text = "Idle"
        prev_x, prev_y = 0, 0

      # 顯示食指指尖座標提示
      cv2.circle(frame, (ix, iy), 8, current_color, -1)

  else:
    prev_x, prev_y = 0, 0

  # 繪製頂部 UI 按鈕介面
  cv2.rectangle(frame, (50, 20), (200, 80), (255, 255, 255), -1)
  cv2.putText(
      frame,
      "CLEAR",
      (75, 60),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 0, 0),
      2,
      cv2.LINE_AA,
  )

  cv2.rectangle(frame, (250, 20), (400, 80), (255, 0, 0), -1)
  cv2.putText(
      frame,
      "BLUE",
      (280, 60),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
  )

  cv2.rectangle(frame, (450, 20), (600, 80), (0, 255, 0), -1)
  cv2.putText(
      frame,
      "GREEN",
      (470, 60),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
  )

  cv2.rectangle(frame, (650, 20), (800, 80), (0, 0, 255), -1)
  cv2.putText(
      frame,
      "RED",
      (690, 60),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
  )

  cv2.rectangle(frame, (850, 20), (1000, 80), (0, 255, 255), -1)
  cv2.putText(
      frame,
      "YELLOW",
      (875, 60),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 0, 0),
      2,
      cv2.LINE_AA,
  )

  # 任務 5：整合攝影機影像與獨立畫布
  # 將畫布上的軌跡疊加回攝影機畫面
  gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
  _, inv_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
  inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)

  frame = cv2.bitwise_and(frame, inv_canvas)
  frame = cv2.bitwise_or(frame, canvas)

  # 計算並顯示 FPS
  current_time = time.time()
  fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
  prev_time = current_time

  cv2.putText(
      frame,
      f"FPS: {int(fps)}",
      (30, 120),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 255, 0),
      2,
  )
  cv2.putText(
      frame,
      f"Gesture: {gesture_text}",
      (30, 160),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 255, 255),
      2,
  )

  # 顯示畫面
  cv2.imshow("Virtual Air Canvas", frame)

  # 鍵盤互動控制
  key = cv2.waitKey(1) & 0xFF
  if key == ord("c") or key == ord("C"):  # 按 C 清除畫布
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
  elif key == ord("s") or key == ord("S"):  # 按 S 儲存繪圖
    cv2.imwrite("saved_drawing.png", canvas)
    print("已儲存繪圖至 saved_drawing.png")
  elif key == 27:  # 按 Esc 鍵離開
    break

cap.release()
cv2.destroyAllWindows()