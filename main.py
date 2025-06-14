import cv2
import mediapipe as mp
import numpy as np
import time
import random
from PIL import ImageFont, ImageDraw, Image

mp_hands = mp.solutions.hands
mp_selfie = mp.solutions.selfie_segmentation
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
selfie = mp_selfie.SelfieSegmentation(model_selection=1)
cap = cv2.VideoCapture(0)

font_path = "C:/Windows/Fonts/malgun.ttf"  # 한글 폰트 경로

bubbles = []
particles = []

slow_words = ["머뭇거림", "망설임", "흐릿한 기억", "사라지는 의지", "잔잔함", "고요함"]
medium_words = ["생각은 흐른다", "번뜩임", "기억의 조각", "흔적은 남는다", "파도치는 생각", "질문"]
fast_words = ["열정", "불안", "갈망", "폭발", "소용돌이", "순간의 선택"]
very_fast_words = ["격렬함", "폭풍", "광란", "순간의 폭발", "극한의 선택"]

def get_bubble_color(speed):
    if speed < 20:
        return (255, 200, 100)      # 연한 노랑/하늘색 (BGR)
    elif speed < 50:
        return (200, 255, 200)      # 연두색 (BGR)
    elif speed < 100:
        return (255, 182, 193)      # 연한 분홍 (BGR)
    else:
        return (180, 120, 255)      # 보라/핑크 (BGR)

def get_word_by_speed(speed):
    if speed < 20:
        return random.choice(slow_words)
    elif speed < 50:
        return random.choice(medium_words)
    elif speed < 100:
        return random.choice(fast_words)
    else:
        return random.choice(very_fast_words)

class Bubble:
    def __init__(self, center, text, speed):
        self.center = center
        self.text = text
        self.alpha = 1.0
        self.speed = speed
        if 50 <= speed < 100:
            self.size = 18 + int(speed // 6)  # 분홍색 구간만 더 작게
        else:
            self.size = 24 + int(speed // 5)
        self.life = 0
        self.color = get_bubble_color(speed)
        self.particles_created = False

    def update(self):
        self.life += 1
        self.alpha = max(0, 1 - self.life / 90)
        self.size *= 0.99
        if self.alpha <= 0 and not self.particles_created:
            self._create_particles()
            self.particles_created = True

    def _create_particles(self):
        for _ in range(18):
            angle = random.uniform(0, 2*np.pi)
            speed = random.uniform(2, 4)
            particles.append({
                'pos': [self.center[0], self.center[1]],
                'vel': [np.cos(angle)*speed, np.sin(angle)*speed],
                'life': random.randint(15, 30),
                'color': (255, 255, 255),  # 흰색
                'size': random.randint(2, 4)
            })

    def draw(self, frame):
        overlay = frame.copy()
        x, y = self.center
        radius = int(self.size)
        cv2.circle(overlay, (x, y), radius, self.color, -1)
        cv2.addWeighted(overlay, self.alpha * 0.7, frame, 1 - self.alpha * 0.7, 0, frame)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, 18)
        bbox = draw.textbbox((0, 0), self.text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x - text_width // 2
        text_y = y - text_height // 2
        draw.text((text_x, text_y), self.text, font=font, fill=(255,255,255,int(255*self.alpha)))
        frame[:,:] = np.array(img_pil)

class Particle:
    @staticmethod
    def update_all():
        for p in particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.07
            p['life'] -= 1
            if p['life'] <= 0:
                particles.remove(p)

    @staticmethod
    def draw_all(frame):
        for p in particles:
            color = p['color']
            size = p['size']
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            for i in range(5):
                angle = i * (2*np.pi/5)
                dx = int(np.cos(angle) * size)
                dy = int(np.sin(angle) * size)
                cv2.line(frame, pos, (pos[0]+dx, pos[1]+dy), color, 1)
            cv2.circle(frame, pos, size, color, -1)

# --- VideoWriter 설정 (녹화 기능) ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

prev_pos = None
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # 배경 제거: 얼굴만 남기고 나머지는 검정색
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_selfie = selfie.process(rgb)
    mask = results_selfie.segmentation_mask
    condition = (mask > 0.5).astype(np.uint8)
    bg = np.zeros(frame.shape, dtype=np.uint8)
    frame = frame * condition[:,:,None] + bg * (1 - condition[:,:,None])
    frame = frame.astype(np.uint8)

    # 손 추적
    results = hands.process(rgb)
    curr_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])
            curr_pos = (x, y)
            if prev_pos is not None:
                dist = ((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2) ** 0.5
                dt = curr_time - prev_time
                speed = dist / dt if dt > 0 else 0
                if speed > 10:
                    text = get_word_by_speed(speed)
                    bubbles.append(Bubble(curr_pos, text, speed))
            prev_pos = curr_pos
            prev_time = curr_time

    for bubble in bubbles[:]:
        bubble.update()
        if bubble.alpha <= 0 and bubble.particles_created:
            bubbles.remove(bubble)
        else:
            bubble.draw(frame)

    Particle.update_all()
    Particle.draw_all(frame)

    cv2.imshow('Thought Bubbles', frame)
    out.write(frame)  # --- 영상 녹화 ---

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
