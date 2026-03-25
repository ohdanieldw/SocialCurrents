import cv2
from pathlib import Path

video_path = Path("data") / "sample.MP4"

print("Video path:", video_path.resolve())
print("Video path exists?:", video_path.exists())


cap = cv2.VideoCapture(str(video_path))

print("Opened:", cap.isOpened())
print("Frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

cap.release()
