import os
import sys

def ensure_dependencies():
    for pkg, mod in [("pytorch-msssim", "pytorch_msssim"), ("transformers", "transformers"), ("einops", "einops"), ("tifffile", "tifffile")]:
        try:
            __import__(mod)
        except ImportError:
            import subprocess
            print(f"Installing missing package: {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

ensure_dependencies()

def get_patches_dir():
    candidates = [
        "/kaggle/input/datasets/suvrajitdeysuv/isro-thermal-patches/output/output/patches",
        "/kaggle/input/isro-thermal-patches/output/output/patches",
        "output/output/patches",
        "D:/Projects/ISRO/output/output/patches"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Stage 1 Ablation Experiments (Table 1 Rows 1-4)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--variant", type=str, default="all", choices=["all", "full", "wo_phy", "wo_ssim", "cnn_sr"], help="Variant name or 'all'")
    parser.add_argument("--patches_dir", type=str, default=None, help="Custom patches directory")
    parser.add_argument("--metadata_json", type=str, default="data/metadata_mapping.json")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--results_file", type=str, default=None)
    args = parser.parse_args()

    patches_dir = args.patches_dir or get_patches_dir()
    checkpoint_dir = args.checkpoint_dir or ("/kaggle/working/checkpoints/stage1" if os.path.exists("/kaggle/working") else "checkpoints/stage1")
    results_file = args.results_file or ("/kaggle/working/ablation_results.json" if os.path.exists("/kaggle/working") else "ablation_results.json")

    print("=" * 60)
    print("🚀 RUNNING STAGE 1 ABLATION SUITE (TABLE 1 ROWS 1-4)")
    print(f"Patches Dir:    {patches_dir}")
    print(f"Metadata JSON:  {args.metadata_json}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Variant:        {args.variant}")
    print(f"Checkpoints:    {checkpoint_dir}")
    print(f"Results File:   {results_file}")
    print("=" * 60)

    import run_ablation
    sys.argv = [
        "run_ablation.py",
        "--stage", "1",
        "--variant", args.variant,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--patches_dir", patches_dir,
        "--metadata_json", args.metadata_json,
        "--checkpoint_dir", checkpoint_dir,
        "--results_file", results_file
    ]
    run_ablation.main()

if __name__ == "__main__":
    main()
