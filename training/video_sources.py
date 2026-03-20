"""
Video source manager — downloads and manages cinematic training videos.

Good training data = diverse, high-quality, cinematic footage:
- Outdoor landscapes, forests, cities (like game environments)
- Dynamic camera movement (tests motion vector quality)
- Varied lighting (day/night/golden hour)
- High detail textures (foliage, stone, water, skin)

Sources used:
- Blender Open Movies (CC licensed, 4K, cinematic quality)
- Free stock footage (Pexels, Pixabay — CC0)
"""

import subprocess
import sys
from pathlib import Path

# Blender Open Movies — CC licensed, high production quality, free
# These are short films with game-like environments (fantasy, sci-fi, nature)
BLENDER_OPEN_MOVIES = {
    "sintel": {
        "url": "https://download.blender.org/demo/movies/Sintel.2010.1080p.mkv",
        "description": "Fantasy film — dragon fights, snow, caves, villages. Perfect game-like environments.",
        "duration": "14min",
        "resolution": "1080p",
    },
    "tears_of_steel": {
        "url": "https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov",
        "description": "Sci-fi — urban environments, robots, explosions. Good for modern game scenarios.",
        "duration": "12min",
        "resolution": "1080p",
    },
    "spring": {
        "url": "https://download.blender.org/demo/movies/Spring%20-%20Blender%20Open%20Movie.webm",
        "description": "Nature/fantasy — lush forests, meadows, dynamic lighting. Enshrouded-like.",
        "duration": "8min",
        "resolution": "1080p",
    },
    "cosmos_laundromat": {
        "url": "https://download.blender.org/demo/movies/CosmosLaundromat-First_Cycle.mp4",
        "description": "Surreal landscapes — grass fields, dramatic skies, varied materials.",
        "duration": "12min",
        "resolution": "1080p",
    },
}

# YouTube — cinematic footage compilations (use yt-dlp, 1080p max)
# These are royalty-free / CC footage compilations
YOUTUBE_SOURCES = {
    "nature_4k_1": {
        "url": "https://www.youtube.com/watch?v=LXb3EKWsInQ",
        "description": "4K nature — forests, mountains, rivers. Game environment material.",
        "max_resolution": "1080",
    },
    "cinematic_nature": {
        "url": "https://www.youtube.com/watch?v=2dD0M_gsmFk",
        "description": "Cinematic nature compilation — varied environments.",
        "max_resolution": "1080",
    },
    "medieval_landscapes": {
        "url": "https://www.youtube.com/watch?v=9MP8E5Ht6pw",
        "description": "Medieval/fantasy landscapes — castles, villages, forests.",
        "max_resolution": "1080",
    },
}


def download_blender_movie(name: str, output_dir: Path) -> Path:
    """Download a Blender open movie."""
    if name not in BLENDER_OPEN_MOVIES:
        raise ValueError(f"Unknown movie: {name}. Available: {list(BLENDER_OPEN_MOVIES.keys())}")

    info = BLENDER_OPEN_MOVIES[name]
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(info["url"]).suffix
    output_path = output_dir / f"{name}{ext}"

    if output_path.exists():
        print(f"[OK] {name} already downloaded: {output_path}")
        return output_path

    print(f"[DL] Downloading {name} ({info['description']})...")
    print(f"     {info['url']}")

    subprocess.run(["curl", "-L", "-o", str(output_path), info["url"]], check=True)
    print(f"[OK] Saved to {output_path}")
    return output_path


def download_youtube(name: str, output_dir: Path, max_res: str = "1080") -> Path:
    """Download a YouTube video using yt-dlp."""
    if name not in YOUTUBE_SOURCES:
        raise ValueError(f"Unknown source: {name}. Available: {list(YOUTUBE_SOURCES.keys())}")

    info = YOUTUBE_SOURCES[name]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.mp4"

    if output_path.exists():
        print(f"[OK] {name} already downloaded: {output_path}")
        return output_path

    print(f"[DL] Downloading {name} ({info['description']})...")
    max_res = info.get("max_resolution", max_res)

    subprocess.run([
        sys.executable, "-m", "yt_dlp",
        "-f", f"bestvideo[height<={max_res}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_res}]",
        "-o", str(output_path),
        "--merge-output-format", "mp4",
        info["url"],
    ], check=True)

    print(f"[OK] Saved to {output_path}")
    return output_path


def download_all(output_dir: Path, include_youtube: bool = False):
    """Download all available training videos."""
    videos = []

    print("=== Downloading Blender Open Movies ===")
    for name in BLENDER_OPEN_MOVIES:
        try:
            path = download_blender_movie(name, output_dir)
            videos.append(path)
        except Exception as e:
            print(f"[ERR] Failed to download {name}: {e}")

    if include_youtube:
        print("\n=== Downloading YouTube Sources ===")
        for name in YOUTUBE_SOURCES:
            try:
                path = download_youtube(name, output_dir)
                videos.append(path)
            except Exception as e:
                print(f"[ERR] Failed to download {name}: {e}")

    print(f"\n=== Downloaded {len(videos)} videos ===")
    return videos


def list_local_videos(video_dir: Path) -> list[Path]:
    """Find all video files in a directory."""
    exts = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in exts]
    print(f"Found {len(videos)} videos in {video_dir}")
    for v in videos:
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"  {v.name} ({size_mb:.0f} MB)")
    return videos


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download training videos")
    parser.add_argument("--output", type=Path, default=Path("data/videos"))
    parser.add_argument("--youtube", action="store_true", help="Also download YouTube sources")
    parser.add_argument("--list", action="store_true", help="Just list local videos")
    args = parser.parse_args()

    if args.list:
        list_local_videos(args.output)
    else:
        download_all(args.output, include_youtube=args.youtube)
