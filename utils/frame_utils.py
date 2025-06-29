import os
import cv2

def extract_frames_from_video(video_path, output_dir, max_frames=20):
    """
    Uniformly extract max_frames images from the video and save them as jpg.
   
        video_path: 
        output_dir: the folder path where the image frame is saved
        max_frames: max
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"video can't open:{video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"video is empty:{video_path}")
        return

    frame_interval = max(1, total_frames // max_frames)

    count, saved = 0, 0
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1

    cap.release()
