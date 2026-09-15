import json
import os
from src.data import build_sample_packages, generate_package_documents


def main():
    base_dir = "data/synthetic/documents"
    os.makedirs(base_dir, exist_ok=True)
    packages = build_sample_packages()

    print("Generating synthetic procurement documents...")
    total_files = 0
    for pkg in packages:
        files = generate_package_documents(pkg, base_dir=base_dir)
        total_files += len(files)
        print(f"  [{pkg.supplier.id}] {pkg.supplier.name}: {len(files)} documents")

    # Save ground truth metadata
    gt_file = "data/synthetic/ground_truth.json"
    with open(gt_file, "w") as f:
        json.dump([pkg.model_dump(mode="json") for pkg in packages], f, indent=2)

    print(f"\nDone: {total_files} PDF files generated in '{base_dir}'.")
    print(f"Ground truth metadata saved in '{gt_file}'.")


if __name__ == "__main__":
    main()
