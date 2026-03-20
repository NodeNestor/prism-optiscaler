"""
Prism Dataset Downloader — downloads all 23 sources to I:/prism-dataset/

Usage:
  python download_all.py                    # download everything
  python download_all.py --only pexels      # download one source
  python download_all.py --list             # show all sources
  python download_all.py --skip-youtube     # skip yt-dlp sources (slow)

Estimated total: ~404 GB across 23 sources → 535K frames
"""

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# Config
# ============================================================================

BASE_DIR = Path("I:/prism-dataset/raw")
HF_TOKEN = None  # loaded at runtime

def get_hf_token():
    global HF_TOKEN
    if HF_TOKEN:
        return HF_TOKEN
    for p in [Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"]:
        if p.exists():
            HF_TOKEN = p.read_text().strip()
            return HF_TOKEN
    return None


def run(cmd, cwd=None, check=True):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check)


def pip_ensure(pkg):
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        run(f"{sys.executable} -m pip install {pkg} -q")


# ============================================================================
# Individual downloaders
# ============================================================================

def dl_pexels(out: Path, queries: list[str], per_query: int = 200):
    """Download stock videos from Pexels API."""
    pip_ensure("requests")
    import requests

    out.mkdir(parents=True, exist_ok=True)

    # Try to get API key from env or prompt
    api_key = os.environ.get("PEXELS_API_KEY", "")
    if not api_key:
        key_file = Path.home() / ".pexels_api_key"
        if key_file.exists():
            api_key = key_file.read_text().strip()
        else:
            print("  [!] No Pexels API key. Get one free at https://www.pexels.com/api/")
            print(f"      Save it to {key_file} or set PEXELS_API_KEY env var")
            return

    headers = {"Authorization": api_key}
    total = 0

    for query in queries:
        print(f"  Pexels: searching '{query}'...")
        page = 1
        downloaded = 0
        while downloaded < per_query:
            resp = requests.get(
                "https://api.pexels.com/videos/search",
                headers=headers,
                params={"query": query, "per_page": 80, "page": page, "size": "medium"},
            )
            if resp.status_code != 200:
                print(f"  [!] Pexels API error: {resp.status_code}")
                break

            data = resp.json()
            videos = data.get("videos", [])
            if not videos:
                break

            for v in videos:
                if downloaded >= per_query:
                    break
                # Get HD video file
                files = sorted(v.get("video_files", []), key=lambda f: f.get("height", 0), reverse=True)
                hd = next((f for f in files if f.get("height", 0) >= 720), files[0] if files else None)
                if not hd:
                    continue

                vid_path = out / f"pexels_{query.replace(' ', '_')}_{v['id']}.mp4"
                if vid_path.exists():
                    downloaded += 1
                    continue

                try:
                    r = requests.get(hd["link"], stream=True, timeout=30)
                    with open(vid_path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                    downloaded += 1
                    total += 1
                except Exception as e:
                    print(f"    [!] Failed: {e}")

            page += 1
            time.sleep(0.5)  # rate limit

        print(f"    Got {downloaded} videos for '{query}'")

    print(f"  Pexels total: {total} videos")


def dl_pixabay(out: Path, queries: list[str], per_query: int = 150):
    """Download stock videos from Pixabay API."""
    pip_ensure("requests")
    import requests

    out.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("PIXABAY_API_KEY", "")
    if not api_key:
        key_file = Path.home() / ".pixabay_api_key"
        if key_file.exists():
            api_key = key_file.read_text().strip()
        else:
            print("  [!] No Pixabay API key. Get one free at https://pixabay.com/api/docs/")
            print(f"      Save it to {key_file} or set PIXABAY_API_KEY env var")
            return

    total = 0
    for query in queries:
        print(f"  Pixabay: searching '{query}'...")
        page = 1
        downloaded = 0
        while downloaded < per_query:
            resp = requests.get(
                "https://pixabay.com/api/videos/",
                params={"key": api_key, "q": query, "per_page": 200, "page": page},
            )
            if resp.status_code != 200:
                print(f"  [!] Pixabay API error: {resp.status_code}")
                break

            data = resp.json()
            hits = data.get("hits", [])
            if not hits:
                break

            for v in hits:
                if downloaded >= per_query:
                    break
                vids = v.get("videos", {})
                # Prefer "large" then "medium"
                vid_info = vids.get("large", vids.get("medium", {}))
                url = vid_info.get("url", "")
                if not url:
                    continue

                vid_path = out / f"pixabay_{query.replace(' ', '_')}_{v['id']}.mp4"
                if vid_path.exists():
                    downloaded += 1
                    continue

                try:
                    r = requests.get(url, stream=True, timeout=30)
                    with open(vid_path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                    downloaded += 1
                    total += 1
                except Exception as e:
                    print(f"    [!] Failed: {e}")

            page += 1
            time.sleep(0.3)

        print(f"    Got {downloaded} videos for '{query}'")

    print(f"  Pixabay total: {total} videos")


def dl_inter4k(out: Path):
    """Download Inter4K dataset (15 GB) — falls back to YouTube 4K search."""
    out.mkdir(parents=True, exist_ok=True)
    if (out / "60fps").exists() or len(list(out.glob("*.mp4"))) > 10:
        print("  Inter4K already downloaded")
        return
    # Inter4K hosted on Google Drive — try gdown, fallback to YouTube 4K search
    try:
        pip_ensure("gdown")
        import gdown
        # Inter4K Google Drive folder ID
        print("  Trying gdown for Inter4K...")
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1Oa-7Ij_sGmlVJrLsKbEn0EjuwwMaAw3V",
            output=str(out), quiet=False, resume=True,
        )
    except Exception as e:
        print(f"  [!] gdown failed: {e}")
        print("  Falling back to YouTube 4K diverse footage...")
        dl_youtube_search(out, "4K 60fps nature cinematic diverse", max_videos=100)
        dl_youtube_search(out, "4K 60fps urban city street", max_videos=50)
        dl_youtube_search(out, "4K 60fps aerial drone landscape", max_videos=50)


def dl_hf_dataset(out: Path, repo_id: str, name: str = "", subset: str = ""):
    """Download a HuggingFace dataset in a SUBPROCESS to isolate memory."""
    out.mkdir(parents=True, exist_ok=True)
    token = get_hf_token()
    # Run in subprocess so if HF indexing eats RAM, it dies cleanly
    script = f'''
import sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id="{repo_id}",
        repo_type="dataset",
        local_dir=r"{out}",
        token="{token or ''}",
    )
    print("HF download complete: {repo_id}")
except Exception as e:
    print(f"HF download failed: {{e}}", file=sys.stderr)
    sys.exit(1)
'''
    run(f'{sys.executable} -c "{script}"', check=False)


def dl_youtube_channel(out: Path, url: str, max_videos: int = 100):
    """Download videos from a YouTube channel/playlist."""
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "downloaded.txt"
    run(
        f'yt-dlp '
        f'--format "bestvideo[height<=1080]+bestaudio/best[height<=1080]" '
        f'--merge-output-format mp4 '
        f'--download-archive "{archive}" '
        f'--max-downloads {max_videos} '
        f'--sleep-interval 2 --max-sleep-interval 5 '
        f'--output "{out}/%(title)s [%(id)s].%(ext)s" '
        f'"{url}"',
        check=False,
    )


def dl_youtube_search(out: Path, query: str, max_videos: int = 50):
    """Download videos from YouTube search."""
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "downloaded.txt"
    run(
        f'yt-dlp '
        f'--format "bestvideo[height<=1080]+bestaudio/best[height<=1080]" '
        f'--merge-output-format mp4 '
        f'--download-archive "{archive}" '
        f'--max-downloads {max_videos} '
        f'--sleep-interval 2 --max-sleep-interval 5 '
        f'--output "{out}/%(title)s [%(id)s].%(ext)s" '
        f'"ytsearch{max_videos}:{query}"',
        check=False,
    )


def dl_archive_org(out: Path, identifier: str):
    """Download from archive.org using ia CLI."""
    pip_ensure("internetarchive")
    out.mkdir(parents=True, exist_ok=True)
    run(f'ia download {identifier} --destdir="{out}" --glob="*.mp4"', check=False)
    # Fallback to other formats
    run(f'ia download {identifier} --destdir="{out}" --glob="*.avi"', check=False)


def dl_opendv_mini(out: Path):
    """Download OpenDV-mini (25hrs driving, ~44GB)."""
    out.mkdir(parents=True, exist_ok=True)
    # OpenDV-mini is available via the DriveAGI repo
    print("  OpenDV-mini: check https://github.com/OpenDriveLab/DriveAGI for download links")
    print("  Or use yt-dlp with the video list from the repo")
    dl_youtube_search(out, "dashcam driving 4K city highway", max_videos=200)
    dl_youtube_search(out, "dashcam night rain driving", max_videos=100)
    dl_youtube_search(out, "dashcam snow fog driving", max_videos=100)


def dl_bdd100k(out: Path):
    """BDD100K requires manual registration."""
    out.mkdir(parents=True, exist_ok=True)
    print("  BDD100K: Register at https://bdd-data.berkeley.edu/ to download")
    print("  Then place files in:", out)
    # Supplement with dashcam footage
    dl_youtube_search(out, "dashcam compilation rain night fog snow 1080p", max_videos=150)


def dl_nasa(out: Path):
    """Download NASA ISS footage from archive.org."""
    out.mkdir(parents=True, exist_ok=True)
    dl_archive_org(out, "NASA-Ultra-High-Definition")
    dl_youtube_search(out, "ISS space station interior tour 4K", max_videos=20)


def dl_tartanair(out: Path):
    """Download TartanAir V2 — uses subprocess per environment to prevent RAM explosion."""
    out.mkdir(parents=True, exist_ok=True)
    # Use the safe per-environment downloader
    run(f'{sys.executable} download_tartanair.py --output "{out}" --max-ram 4', check=False)


def dl_xd_violence(out: Path):
    """Download XD-Violence dataset from HuggingFace."""
    dl_hf_dataset(out, "jherng/xd-violence")


def dl_hdvila_filtered(out: Path, queries: list[str], per_query: int = 50):
    """Download caption-filtered videos from HD-VILA via YouTube."""
    out.mkdir(parents=True, exist_ok=True)
    for query in queries:
        subdir = out / query.replace(" ", "_")
        dl_youtube_search(subdir, f"{query} 4K cinematic", max_videos=per_query)


# ============================================================================
# Master source registry
# ============================================================================

SOURCES = {
    # ---- Nature / Landscapes ----
    "pexels-nature": {
        "fn": lambda o: dl_pexels(o, [
            "forest", "mountain landscape", "river waterfall", "ocean waves",
            "sunset golden hour", "storm lightning", "rain", "snow landscape",
            "meadow field", "lake reflection", "autumn forest", "tropical beach",
        ], per_query=200),
        "desc": "Nature/weather stock footage",
        "est_gb": 20,
        "frames": 40000,
    },
    "pixabay-nature": {
        "fn": lambda o: dl_pixabay(o, [
            "nature landscape", "ocean underwater", "rain heavy", "snow mountain",
            "lightning storm", "volcano", "aurora borealis", "foggy forest",
        ], per_query=150),
        "desc": "Nature/atmospheric stock footage",
        "est_gb": 12,
        "frames": 25000,
    },
    "inter4k": {
        "fn": dl_inter4k,
        "desc": "4K 60fps diverse outdoors (1000 videos)",
        "est_gb": 15,
        "frames": 30000,
    },

    # ---- Driving / Urban ----
    "driving": {
        "fn": dl_opendv_mini,
        "desc": "Dashcam driving — cities, highways, rain, night, fog",
        "est_gb": 44,
        "frames": 35000,
    },
    "bdd100k": {
        "fn": dl_bdd100k,
        "desc": "Weather-tagged dashcam (manual registration + YT supplement)",
        "est_gb": 20,
        "frames": 30000,
    },

    # ---- Interiors ----
    "realestate10k": {
        "fn": lambda o: dl_hf_dataset(o, "RE10K/RealEstate10K"),
        "desc": "Interior walkthroughs — rooms, hallways",
        "est_gb": 15,
        "frames": 25000,
    },

    # ---- Aerial / Drone ----
    "dl3dv-drone": {
        "fn": lambda o: dl_hf_dataset(o, "DL3DV/DL3DV-Drone"),
        "desc": "Aerial drone footage of landscapes",
        "est_gb": 30,
        "frames": 20000,
    },

    # ---- Cinematic / Diverse ----
    "finevideo": {
        "fn": lambda o: dl_hf_dataset(o, "HuggingFaceFV/finevideo"),
        "desc": "43K diverse cinematic YouTube videos",
        "est_gb": 25,
        "frames": 35000,
    },
    "hdvila-filtered": {
        "fn": lambda o: dl_hdvila_filtered(o, [
            "medieval castle walkthrough", "underwater coral reef",
            "campfire night forest", "ancient ruins exploration",
            "foggy mountain trail", "neon city night rain",
            "cherry blossom japan", "northern lights landscape",
        ], per_query=50),
        "desc": "Caption-filtered cinematic (castles, underwater, etc)",
        "est_gb": 20,
        "frames": 40000,
    },

    # ---- War / Combat / Action ----
    "xd-violence": {
        "fn": dl_xd_violence,
        "desc": "Explosions, shooting, fighting, riots (217 hrs)",
        "est_gb": 15,
        "frames": 15000,
    },
    "archive-war": {
        "fn": lambda o: [
            dl_archive_org(o / "wwii", "HCT6juyt"),
            dl_archive_org(o / "ukraine", "ukraine-war-combat-footage"),
            dl_archive_org(o / "wwii-color", "82164-udey-saipan-action"),
        ],
        "desc": "WWII + Ukraine combat footage (public domain)",
        "est_gb": 25,
        "frames": 20000,
    },
    "kinetics-action": {
        "fn": lambda o: [
            dl_youtube_search(o / "combat", q, max_videos=30)
            for q in [
                "sword fighting real", "boxing match knockout",
                "MMA fight highlight", "fencing competition",
                "archery competition", "martial arts sparring",
                "car chase movie scene", "parkour POV rooftop",
                "skiing POV GoPro 4K", "snowboarding POV",
                "skateboarding POV", "motorcycle racing onboard",
            ]
        ],
        "desc": "Action: combat, sports, racing, parkour",
        "est_gb": 15,
        "frames": 30000,
    },

    # ---- First-Person / Egocentric ----
    "xperience": {
        "fn": lambda o: dl_hf_dataset(o, "ropedia-ai/xperience-10m"),
        "desc": "First-person POV with depth (selective)",
        "est_gb": 15,
        "frames": 20000,
    },

    # ---- Caves / Underground (GAP FIX) ----
    "caves": {
        "fn": lambda o: [
            dl_youtube_channel(o / "caveman", "https://www.youtube.com/@CavemanHikes/videos", max_videos=50),
            dl_youtube_channel(o / "mines", "https://www.youtube.com/@ExploringAbandonedMines/videos", max_videos=50),
            dl_pexels(o / "stock", ["cave exploration", "tunnel underground", "subway station"], per_query=100),
        ],
        "desc": "Cave exploration, mines, tunnels, flashlight lighting",
        "est_gb": 25,
        "frames": 25000,
    },

    # ---- Medieval / Historical (GAP FIX) ----
    "medieval": {
        "fn": lambda o: [
            dl_youtube_channel(o / "prowalk", "https://www.youtube.com/@ProWalkTours/videos", max_videos=80),
            dl_hf_dataset(o / "walking_tours", "shawshankvkt/Walking_Tours"),
            dl_pexels(o / "stock", ["medieval castle", "cathedral interior", "ancient ruins", "stone village"], per_query=150),
            dl_pixabay(o / "stock_px", ["medieval", "castle", "cathedral", "roman ruins"], per_query=100),
        ],
        "desc": "Medieval towns, castles, cathedrals, ruins, walkthroughs",
        "est_gb": 45,
        "frames": 30000,
    },

    # ---- Desert / Arid (GAP FIX) ----
    "desert": {
        "fn": lambda o: [
            dl_pexels(o, ["desert landscape", "sand dunes", "canyon", "Sahara", "arid wasteland"], per_query=150),
            dl_pixabay(o, ["desert", "canyon", "sand dunes", "dry landscape"], per_query=100),
            dl_youtube_search(o / "yt", "4K drone desert canyon landscape", max_videos=50),
        ],
        "desc": "Desert, canyons, dunes, arid landscapes",
        "est_gb": 13,
        "frames": 15000,
    },

    # ---- Sci-Fi / Industrial / Night (GAP FIX) ----
    "scifi-industrial": {
        "fn": lambda o: [
            dl_pexels(o, ["cyberpunk city night", "neon city rain", "server room", "factory industrial", "futuristic"], per_query=100),
            dl_pixabay(o, ["cyberpunk", "neon city", "industrial factory", "data center"], per_query=80),
            dl_nasa(o / "nasa"),
        ],
        "desc": "Neon cities, industrial, space station, server rooms",
        "est_gb": 10,
        "frames": 15000,
    },

    # ---- Sports / Fast Action (GAP FIX) ----
    "sports-action": {
        "fn": lambda o: [
            dl_youtube_search(o, q, max_videos=30)
            for q in [
                "GoPro skiing POV 4K", "mountain biking POV trail",
                "surfing POV GoPro", "skydiving POV 4K",
                "rally onboard camera", "F1 onboard lap",
            ]
        ],
        "desc": "Sports POV: skiing, biking, surfing, racing",
        "est_gb": 10,
        "frames": 15000,
    },

    # ---- Explosions / Effects (extra coverage) ----
    "effects": {
        "fn": lambda o: [
            dl_pixabay(o, [
                "explosion fire", "smoke explosion", "demolition building",
                "fireworks", "sparks welding", "dust storm",
            ], per_query=100),
        ],
        "desc": "Explosions, fire, demolition, particles, effects",
        "est_gb": 5,
        "frames": 10000,
    },

    # ---- Synthetic (pre-training + validation) ----
    "tartanair": {
        "fn": dl_tartanair,
        "desc": "Synthetic with perfect depth/flow/normals (15 environments)",
        "est_gb": 20,
        "frames": 30000,
    },
}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download all Prism training data")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--only", nargs="+", help="Only download these sources")
    parser.add_argument("--skip", nargs="+", default=[], help="Skip these sources")
    parser.add_argument("--skip-youtube", action="store_true", help="Skip all yt-dlp sources (slow)")
    parser.add_argument("--list", action="store_true", help="Just list all sources")
    args = parser.parse_args()

    if args.list:
        total_gb = 0
        total_frames = 0
        print(f"\n{'#':<3} {'Source':<20} {'Est GB':>7} {'Frames':>8}  Description")
        print("-" * 85)
        for i, (name, info) in enumerate(SOURCES.items(), 1):
            print(f"{i:<3} {name:<20} {info['est_gb']:>5} GB {info['frames']:>7}  {info['desc']}")
            total_gb += info["est_gb"]
            total_frames += info["frames"]
        print("-" * 85)
        print(f"    {'TOTAL':<20} {total_gb:>5} GB {total_frames:>7}")
        return

    sources = SOURCES
    if args.only:
        sources = {k: v for k, v in SOURCES.items() if k in args.only}
    if args.skip:
        sources = {k: v for k, v in sources.items() if k not in args.skip}

    yt_sources = {"caves", "medieval", "hdvila-filtered", "driving", "bdd100k",
                  "kinetics-action", "sports-action", "desert", "scifi-industrial"}
    if args.skip_youtube:
        sources = {k: v for k, v in sources.items() if k not in yt_sources}
        print("Skipping YouTube-based sources (use without --skip-youtube to include)")

    total_gb = sum(v["est_gb"] for v in sources.values())
    total_frames = sum(v["frames"] for v in sources.values())

    # Check disk space
    disk = shutil.disk_usage(str(args.base_dir.parent if args.base_dir.parent.exists() else Path("I:/")))
    free_gb = disk.free / (1024**3)
    print(f"\nDisk I: — {free_gb:.0f} GB free, need ~{total_gb} GB")
    if free_gb < total_gb:
        print(f"  [WARN] Might not have enough space! ({free_gb:.0f} < {total_gb})")

    print(f"Downloading {len(sources)} sources (~{total_gb} GB, ~{total_frames} frames)")
    print(f"Base directory: {args.base_dir}\n")

    # Progress log file
    log_path = args.base_dir / "download_progress.log"
    args.base_dir.mkdir(parents=True, exist_ok=True)

    completed = []
    failed = []
    start_time = time.time()

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def get_dir_size_gb(path: Path) -> float:
        if not path.exists():
            return 0
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024**3)

    def print_status():
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        done = len(completed)
        total = len(sources)
        downloaded_gb = sum(get_dir_size_gb(args.base_dir / n) for n in completed)
        disk_now = shutil.disk_usage(str(args.base_dir)).free / (1024**3)

        log(f"--- PROGRESS: {done}/{total} sources | {downloaded_gb:.1f} GB downloaded | "
            f"{disk_now:.0f} GB free | elapsed {elapsed_str} ---")

    log(f"Starting download of {len(sources)} sources (~{total_gb} GB)")
    log(f"Progress log: {log_path}")

    for i, (name, info) in enumerate(sources.items(), 1):
        out_dir = args.base_dir / name

        log(f"")
        log(f"{'='*60}")
        log(f"[{i}/{len(sources)}] {name} — {info['desc']}")
        log(f"  Est: {info['est_gb']} GB, {info['frames']} frames")
        log(f"  Dir: {out_dir}")
        log(f"{'='*60}")

        t0 = time.time()
        try:
            result = info["fn"](out_dir)
            dt = time.time() - t0
            size = get_dir_size_gb(out_dir)
            log(f"  [DONE] {name} — {size:.1f} GB in {timedelta(seconds=int(dt))}")
            completed.append(name)
        except Exception as e:
            log(f"  [ERROR] {name}: {e}")
            failed.append((name, str(e)))

        print_status()

    # Final summary
    elapsed = str(timedelta(seconds=int(time.time() - start_time)))
    total_downloaded = sum(get_dir_size_gb(args.base_dir / n) for n in completed)

    log(f"\n{'='*60}")
    log(f"DOWNLOAD COMPLETE")
    log(f"  Completed: {len(completed)}/{len(sources)} sources")
    log(f"  Downloaded: {total_downloaded:.1f} GB")
    log(f"  Time: {elapsed}")
    log(f"  Data: {args.base_dir}")
    if failed:
        log(f"  Failed ({len(failed)}):")
        for name, err in failed:
            log(f"    - {name}: {err}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
