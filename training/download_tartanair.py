"""
TartanAir V2 downloader — downloads one environment at a time to avoid RAM explosion.

The HuggingFace snapshot_download indexes the ENTIRE repo in memory before filtering.
TartanAir V2 has millions of files across 74 environments — indexing all at once
uses 5+ GB RAM and crashes. This script downloads one env at a time, each in a
subprocess so memory is released between environments.

Usage:
  python download_tartanair.py                          # download all envs
  python download_tartanair.py --envs CastleFortress Ocean CoalMine  # specific envs
  python download_tartanair.py --list                   # list all environments
  python download_tartanair.py --max-ram 4              # kill if >4 GB RAM used
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ALL_ENVS = [
    # Medieval / Fantasy
    'AncientTowns', 'Antiquity3D', 'CastleFortress', 'Fantasy', 'GothicIsland',
    'HQWesternSaloon', 'OldScandinavia', 'Rome', 'Ruins',
    # Nature / Forest
    'ForestEnv', 'GreatMarsh', 'SeasonalForestAutumn', 'SeasonalForestSpring',
    'SeasonalForestSummerNight', 'SeasonalForestWinter', 'SeasonalForestWinterNight',
    'TerrainBlending',
    # Underground / Caves
    'CoalMine', 'Sewerage', 'ShoreCaves',
    # Desert / Arid
    'DesertGasStation', 'EndofTheWorld', 'WesternDesertTown',
    # Sci-fi / Cyberpunk
    'BrushifyMoon', 'Cyberpunk', 'CyberPunkDowntown', 'PolarSciFi',
    # Industrial / Post-apocalyptic
    'AbandonedCable', 'AbandonedFactory', 'AbandonedFactory2', 'AbandonedSchool',
    'Apocalyptic', 'CarWelding', 'ConstructionSite', 'FactoryWeather',
    'IndustrialHangar', 'Slaughter', 'UrbanConstruction',
    # Cities / Urban
    'Downtown', 'HongKong', 'JapaneseAlley', 'JapaneseCity', 'MiddleEast',
    'ModUrbanCity', 'ModernCityDowntown', 'OldIndustrialCity', 'SoulCity',
    'VictorianStreet',
    # Interiors
    'AmericanDiner', 'ArchVizTinyHouseDay', 'ArchVizTinyHouseNight',
    'CountryHouse', 'Hospital', 'House', 'Office', 'Prison',
    'Restaurant', 'RetroOffice', 'Supermarket',
    # Water / Coast
    'NordicHarbor', 'Ocean', 'SeasideTown', 'WaterMillDay', 'WaterMillNight',
    # Night variants
    'OldBrickHouseDay', 'OldBrickHouseNight', 'OldTownFall', 'OldTownNight',
    'OldTownSummer', 'OldTownWinter',
    # Neighborhoods
    'AmusementPark', 'Gascola', 'ModularNeighborhood', 'ModularNeighborhoodIntExt',
]

# Files we need per environment (~10 GB each)
FILES_PER_ENV = [
    'image_lcam_front.zip',
    'depth_lcam_front.zip',
    'flow_lcam_front.zip',
]

def get_hf_token() -> str:
    for p in [Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"]:
        if p.exists():
            return p.read_text().strip()
    return ""


def download_env(env: str, out_dir: Path, token: str, max_ram_gb: float = 6.0) -> bool:
    """Download one environment in a subprocess. Returns True if successful."""
    # Write a temp script file to avoid shell escaping nightmares
    script_path = out_dir / "_download_env.py"
    script_path.write_text(f'''
import sys
from huggingface_hub import hf_hub_download

env = "{env}"
out_dir = r"{out_dir}"
token = "{token}"
files = {FILES_PER_ENV!r}

for fname in files:
    path = env + "/Data_easy/" + fname
    print(f"  Downloading {{path}}...")
    try:
        hf_hub_download(
            repo_id="theairlabcmu/tartanair2",
            repo_type="dataset",
            filename=path,
            local_dir=out_dir,
            token=token,
        )
        print(f"  OK: {{path}}")
    except Exception as e:
        print(f"  SKIP: {{path}} ({{e}})")

print(f"  Done: {env}")
''')

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Monitor output and memory
    try:
        import psutil
        ps_proc = psutil.Process(proc.pid)
    except:
        ps_proc = None

    while proc.poll() is None:
        line = proc.stdout.readline()
        if line:
            print(line.rstrip())

        # Safety: kill if using too much RAM
        if ps_proc:
            try:
                mem_gb = ps_proc.memory_info().rss / (1024 ** 3)
                if mem_gb > max_ram_gb:
                    print(f"  [!] RAM limit hit ({mem_gb:.1f} GB > {max_ram_gb} GB), killing...")
                    proc.kill()
                    return False
            except:
                pass

    return proc.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Download TartanAir V2 environments safely")
    parser.add_argument("--output", type=Path, default=Path("I:/prism-dataset/raw/tartanair"))
    parser.add_argument("--envs", nargs="+", default=None, help="Specific environments to download")
    parser.add_argument("--list", action="store_true", help="List all environments")
    parser.add_argument("--max-ram", type=float, default=6.0, help="Kill subprocess if it exceeds this GB")
    args = parser.parse_args()

    if args.list:
        print(f"{len(ALL_ENVS)} environments:")
        for e in ALL_ENVS:
            print(f"  {e}")
        return

    envs = args.envs or ALL_ENVS
    token = get_hf_token()
    if not token:
        print("[!] No HuggingFace token found")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(envs)} environments to {args.output}")
    print(f"RAM limit per subprocess: {args.max_ram} GB")
    print(f"Files per env: {FILES_PER_ENV}")
    print(f"Estimated per env: ~10 GB, total: ~{len(envs) * 10} GB\n")

    completed = []
    failed = []

    for i, env in enumerate(envs, 1):
        # Skip if already downloaded
        env_dir = args.output / env / "Data_easy"
        existing = [f for f in FILES_PER_ENV if (env_dir / f).exists()]
        if len(existing) == len(FILES_PER_ENV):
            print(f"[{i}/{len(envs)}] {env} — already complete, skipping")
            completed.append(env)
            continue

        print(f"\n[{i}/{len(envs)}] {env}")
        t0 = time.time()
        ok = download_env(env, args.output, token, max_ram_gb=args.max_ram)
        dt = time.time() - t0

        if ok:
            print(f"  Completed in {dt:.0f}s")
            completed.append(env)
        else:
            print(f"  FAILED after {dt:.0f}s")
            failed.append(env)

    print(f"\n{'='*60}")
    print(f"TartanAir download complete")
    print(f"  OK: {len(completed)}/{len(envs)}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
