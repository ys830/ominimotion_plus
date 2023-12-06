import cv2
import numpy as np

def frame_difference(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    return np.sum(diff)

def collect_keyframes(video_path, base_frequency=3, max_frequency=10, diff_threshold=21505562):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    keyframes = [prev_frame]
    frame_count = 1
    current_frequency = base_frequency

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        diff = frame_difference(prev_frame, current_frame)

        if diff > diff_threshold:
            # 在差异较大的地方增加采样频率
            current_frequency = min(current_frequency + 1, max_frequency)
        else:
            # 在差异较小的地方降低采样频率
            current_frequency = max(current_frequency - 1, base_frequency)

        if frame_count % current_frequency == 0:
            keyframes.append(current_frame)

        frame_count += 1
        prev_frame = current_frame

    cap.release()
    return keyframes

# 用法示例
video_path = '/data/yisi/mywork/ominimotion_plus/video-keyframe-detector-master/videos/bear.mp4'
keyframes = collect_keyframes(video_path)

# 这里可以对采集到的关键帧进行进一步处理，比如保存到文件或显示
for idx, keyframe in enumerate(keyframes):
    cv2.imwrite(f'keyframe_{idx}.jpg', keyframe)
