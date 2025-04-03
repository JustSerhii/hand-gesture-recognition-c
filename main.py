import cv2
import math
import numpy as np
import mediapipe as mp
from collections import deque

# Ініціалізація MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Відстеження останніх N кадрів
history = deque(maxlen=5)

# Функція для обчислення евклідової відстані між двома точками
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Функція для визначення кута між трьома точками
def calculate_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Функція для визначення кута нахилу кисті щодо горизонту
def calculate_wrist_tilt(wrist, mid_palm):
    return np.degrees(math.atan2(mid_palm.y - wrist.y, mid_palm.x - wrist.x))

# Функція для перевірки форми напівкола
def check_semicircle_shape(landmarks):
    # Отримуємо координати кінчиків пальців (без великого)
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Центр кривизни: середня точка між кінчиками вказівного та мізинця
    center_x = (index_tip.x + pinky_tip.x) / 2
    center_y = (index_tip.y + pinky_tip.y) / 2

    # Відстані до центру
    d_index = calculate_distance(index_tip, type('', (), {"x": center_x, "y": center_y})())
    d_middle = calculate_distance(middle_tip, type('', (), {"x": center_x, "y": center_y})())
    d_ring = calculate_distance(ring_tip, type('', (), {"x": center_x, "y": center_y})())
    d_pinky = calculate_distance(pinky_tip, type('', (), {"x": center_x, "y": center_y})())

    # Середня відстань
    avg_distance = (d_index + d_middle + d_ring + d_pinky) / 4

    # Перевіряємо, що всі пальці знаходяться на приблизно однаковій відстані від центру (±10%)
    return (abs(d_index - avg_distance) < 0.02 and
            abs(d_middle - avg_distance) < 0.02 and
            abs(d_ring - avg_distance) < 0.02 and
            abs(d_pinky - avg_distance) < 0.02)

# Функція для визначення жесту "C"
def recognize_c_gesture(landmarks):
    if not landmarks:
        return False

    # Основні точки руки
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    mid_palm = landmarks[9]  # Точка біля середини кисті

    # Відстані між пальцями
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    middle_ring_dist = calculate_distance(middle_tip, ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)

    # Співвідношення ширини до висоти кисті
    hand_width = calculate_distance(thumb_tip, pinky_tip)
    hand_height = calculate_distance(wrist, middle_tip)
    aspect_ratio = hand_width / hand_height if hand_height != 0 else 0

    # Кути згину пальців
    index_angle = calculate_angle(landmarks[5], landmarks[6], index_tip)
    middle_angle = calculate_angle(landmarks[9], landmarks[10], middle_tip)
    ring_angle = calculate_angle(landmarks[13], landmarks[14], ring_tip)
    pinky_angle = calculate_angle(landmarks[17], landmarks[18], pinky_tip)

    # Кут нахилу кисті щодо горизонту
    wrist_tilt = calculate_wrist_tilt(wrist, mid_palm)

    # Вивід для налагодження
    print(f"Wrist Tilt: {wrist_tilt}")

    # **Фільтрація вертикального положення кисті**
    if not (-140 <= wrist_tilt <= -90):  # Кисть має бути нахилена в цьому діапазоні
        return False

    # **Перевірка форми напівкола**
    if not check_semicircle_shape(landmarks):
        return False

    # **Перевірка критеріїв для літери "C"**
    if (0.07 < thumb_index_dist < 0.155 and
            index_middle_dist < 0.06 and
            middle_ring_dist < 0.08 and
            ring_pinky_dist < 0.1 and
            0.2 < aspect_ratio < 1.0 and
            55 < index_angle < 165 and
            50 < middle_angle < 150 and
            45 < ring_angle < 140 and
            60 < pinky_angle < 155):
        return True

    return False

# Ініціалізація відеопотоку
cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            is_c_gesture = recognize_c_gesture(landmarks)

            history.append(is_c_gesture)

            # Відображення тексту, якщо жест розпізнано
            if sum(history) > len(history) // 2:
                cv2.putText(frame, "Gesture: C", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 32:
        break

cap.release()
cv2.destroyAllWindows()
