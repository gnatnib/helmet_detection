import cv2
import os
import re

# =========================
# LIST VIDEO INPUT
# =========================
video_paths = [
    r"D:\Projects\comvis\PANTAUSEMAR - PANTAU SEMARANG CCTV ONLINE KOTA SEMARANG6.mp4"
]

output_folder = "dataset"
os.makedirs(output_folder, exist_ok=True)

existing_files = os.listdir(output_folder)

frame_numbers = []
for f in existing_files:
    match = re.match(r"frame_(\d+)\.jpg", f)
    if match:
        frame_numbers.append(int(match.group(1)))

saved = max(frame_numbers) + 1 if frame_numbers else 1

print(f"Mulai menyimpan dari frame_{saved:04d}.jpg")

interval = 1  # 1 frame per detik

for video_path in video_paths:
    print(f"\nMemproses video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        print("Gagal membaca FPS, skip video ini")
        continue

    frame_interval = int(fps * interval)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_name = os.path.join(
                output_folder,
                f"frame_{saved:04d}.jpg"
            )
            cv2.imwrite(frame_name, frame)
            print(f"Menyimpan {frame_name}")
            saved += 1

        count += 1

    cap.release()

print(f"\nSelesai. Total frame tersimpan sekarang: {saved - 1}")
