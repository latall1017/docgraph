import os
from src.data import build_sample_packages, generate_package_documents


def test_generate_sample_pdfs(tmp_path):
    packages = build_sample_packages()
    all_generated = []

    for pkg in packages:
        files = generate_package_documents(pkg, base_dir=str(tmp_path))
        all_generated.extend(files)

    # 3 from Alpha + 3 from Beta + 4 from Gamma + 1 from Delta = 11 PDFs
    assert len(all_generated) == 11

    # Verify that each file was created and is a non-empty PDF
    for file_path in all_generated:
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) > 1000  # valid PDF is > 1KB
        assert file_path.endswith(".pdf")
