import os
import json
from run_ablation import print_ablation_table

def main():
    candidates = [
        "/kaggle/working/ablation_results.json",
        "ablation_results.json"
    ]
    file_path = None
    for c in candidates:
        if os.path.exists(c):
            file_path = c
            break

    if not file_path:
        print("❌ No ablation_results.json found yet. Run run_stage1.py or run_stage2.py first.")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    print(f"Loaded results from: {file_path}")
    print_ablation_table(data)

if __name__ == "__main__":
    main()
