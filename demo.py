"""One-click root runner for GenEvac-Sentinel presentation demo."""
import sys
from pathlib import Path

# Add scripts directory
scripts_dir = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from demo_presentation import run_demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GenEvac-Sentinel Presentation Showcase")
    parser.add_argument("--show", action="store_true", help="Display interactive matplotlib plot windows")
    parser.add_argument("--out", type=str, default="presentation_assets", help="Output directory for slide PNG assets")
    args = parser.parse_args()
    run_demo(interactive=args.show, output_dir=args.out)

