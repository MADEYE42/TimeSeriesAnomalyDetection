import cv2
import os

def extract_frames_from_videos(input_dir, output_dir, interval_sec=1):
    # Supported classes
    categories = ['Fall', 'Normal']

    for category in categories:
        input_path = os.path.join(input_dir, category)
        output_path = os.path.join(output_dir, category)
        os.makedirs(output_path, exist_ok=True)

        video_files = [f for f in os.listdir(input_path) if f.endswith('.mp4')]

        for video_idx, video_file in enumerate(video_files, 1):
            video_path = os.path.join(input_path, video_file)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"❌ Failed to open video: {video_file}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                print(f"⚠️ Invalid FPS in video: {video_file}")
                cap.release()
                continue

            interval_frames = int(fps * interval_sec)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            extracted_count = 0

            print(f"🔄 Processing {video_file} (FPS={fps}, Total Frames={total_frames})")

            while frame_count < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()

                if not ret:
                    break

                frame_filename = f"{category}_{video_idx}_frame_{extracted_count+1:02d}.jpg"
                frame_output_path = os.path.join(output_path, frame_filename)
                cv2.imwrite(frame_output_path, frame)

                extracted_count += 1
                frame_count += interval_frames

            cap.release()
            print(f"✅ Extracted {extracted_count} frames from {video_file} to {output_path}")

    print("🎉 Frame extraction completed for all videos.")


# 📌 Run the extraction
if __name__ == "__main__":
    input_video_directory = "VideoDatasetFallVsNormal"
    output_frames_directory = "ExtractedFramesFallVsNormal"
    extract_frames_from_videos(input_video_directory, output_frames_directory, interval_sec=1)
