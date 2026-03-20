"""Process all video sources into .pt samples. Handles subdirectories."""
import sys
from pathlib import Path
from generate_dataset import generate_dataset

sources = [
    ("I:/prism-dataset/raw/medieval/pexels", "medieval"),
    ("I:/prism-dataset/raw/fps-pov/pexels", "fps-pov"),
    ("I:/prism-dataset/raw/caves/stock", "caves-stock"),
    ("I:/prism-dataset/raw/caves/mines", "caves-mines"),
    ("I:/prism-dataset/raw/effects/pixabay", "effects"),
    ("I:/prism-dataset/raw/xd-violence", "xd-violence"),
    ("I:/prism-dataset/raw/dl3dv-drone", "drone"),
]

output = Path("I:/prism-dataset/dataset-v2")
output.mkdir(parents=True, exist_ok=True)

for video_dir, name in sources:
    vdir = Path(video_dir)
    videos = list(vdir.glob("*.mp4")) + list(vdir.glob("*.mkv")) + list(vdir.glob("*.webm"))
    if not videos:
        print(f"[{name}] No videos in {vdir}, skipping")
        continue
    print(f"\n{'='*60}")
    print(f"[{name}] {len(videos)} videos from {vdir}")
    print(f"{'='*60}")
    
    generate_dataset(
        video_paths=videos[:200],  # cap at 200 per source for diversity
        output_dir=output,
        display_size=(1080, 1920),
        resolution_presets=["performance", "ultra_performance"],  # 2x and 3x
        max_frames_per_video=500,
        skip_frames=3,
        game_style="random",
        device="cuda:0",
    )

print("\nALL SOURCES PROCESSED!")
