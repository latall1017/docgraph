import json
import os
from src.data import build_sample_packages


def test_build_sample_packages():
    packages = build_sample_packages()
    assert len(packages) == 4

    # 1. Normal package
    pkg1 = packages[0]
    assert pkg1.supplier.name == "Alpha Services SAS"
    assert len(pkg1.expected_anomalies) == 0

    # 2. Invoice exceeding PO
    pkg2 = packages[1]
    assert "invoice_exceeds_po" in pkg2.expected_anomalies
    assert pkg2.invoices[0].total_amount_ht > pkg2.purchase_orders[0].total_amount_ht

    # 3. Amended contract
    pkg3 = packages[2]
    assert len(pkg3.amendments) == 1
    assert pkg3.invoices[0].total_amount_ht > pkg3.contract.maximum_amount
    assert pkg3.invoices[0].total_amount_ht <= pkg3.amendments[0].new_maximum_amount
    assert len(pkg3.expected_anomalies) == 0

    # 4. Missing PO
    pkg4 = packages[3]
    assert "missing_po" in pkg4.expected_anomalies
    assert pkg4.invoices[0].po_ref is None


def test_export_sample_packages_to_json(tmp_path):
    packages = build_sample_packages()
    out_file = tmp_path / "ground_truth_sample.json"

    data = [pkg.model_dump(mode="json") for pkg in packages]
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)

    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        loaded = json.load(f)
    assert len(loaded) == 4
