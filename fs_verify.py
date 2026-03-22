import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from data.feature_store import (
    FeatureWriter, 
    FeatureReader, 
    FeatureBuilder, 
    SnapshotTester, 
    ModelProvenance,
    init_store,
    STORE_PATH
)

def test_feature_store_flow():
    print("Testing Feature Store Flow...")
    
    # Clean setup
    if STORE_PATH.exists():
        STORE_PATH.unlink()
    init_store()
    
    writer = FeatureWriter(version="v1.0.0")
    builder = FeatureBuilder(writer=writer)
    
    # 1. Build features
    raw_data = {
        "ticker": "RELIANCE",
        "rsi_14": 55.5,
        "macd_signal": 0.1,
        "pe_ratio": 25.0,
        "total_score": 80.0,
    }
    print("Building features...")
    builder.build(raw_data, "2024-03-22", label=1.0)
    
    # Add enough rows for SnapshotTester (MIN_ROWS = 500)
    print("Adding batch data for snapshot testing...")
    batch = []
    for i in range(550):
        batch.append({
            "ticker": f"TICKER_{i}",
            "date": "2024-03-22",
            "features": {"rsi_14": 50.0 + i % 10, "pe_ratio": 20.0},
            "raw_source": {"source": "test"},
            "label": 1.0 if i % 2 == 0 else 0.0
        })
    writer.write_batch(batch)
    
    # 2. Read features
    print("Reading features via FeatureReader...")
    reader = FeatureReader(version="v1.0.0")
    X, y, meta = reader.load_dataset("2024-03-22", "2024-03-22")
    print(f"Loaded {len(X)} rows.")
    assert len(X) >= 501
    
    # 3. Snapshot
    print("Creating dataset snapshot...")
    snapshot_id = reader.snapshot(X, y, meta)
    print(f"Snapshot ID: {snapshot_id}")
    
    # 4. Test Snapshot
    print("Running SnapshotTester...")
    tester = SnapshotTester()
    test_results = tester.run_all(X, y, meta, snapshot_id, reader)
    for test, passed in test_results.items():
        print(f"  - {test}: {'PASS' if passed else 'FAIL'}")
    
    # 5. Register Provenance
    print("Registering Model Provenance...")
    prov = ModelProvenance()
    model_id = prov.register(
        snapshot_id=snapshot_id,
        dataset_hash=reader.get_snapshot(snapshot_id)["dataset_hash"],
        algorithm="test_algo",
        version="v1.0.0",
        walk_forward_auc=0.75,
        brier_score=0.2,
        artifact_path="runtime/models/test.pkl"
    )
    print(f"Model ID: {model_id}")
    
    # 6. Verify lineage
    print("Verifying Audit Trail...")
    trail = prov.audit_trail()
    assert len(trail) >= 1
    assert trail[0]["model_id"] == model_id
    print("Lineage [OK]")

if __name__ == "__main__":
    try:
        test_feature_store_flow()
        print("\nFEATURE STORE VERIFIED SUCCESSFULLY")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
