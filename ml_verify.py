import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from engines.ml.ensemble import SovereignEnsemble
from engines.ml.calibration import calibrate_model
from engines.ml.validation import walk_forward_validate
from engines.ml.trainer import SovereignTrainer
from ml.predict import ModelPredictor

def test_ensemble():
    print("Testing SovereignEnsemble...")
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    ensemble = SovereignEnsemble()
    ensemble.fit(X, y)
    probs = ensemble.predict_proba(X)
    print(f"Ensemble Probs shape: {probs.shape}")
    assert probs.shape == (100, 2)
    print("Ensemble Fit/Predict [OK]")

def test_calibration():
    print("Testing Calibration...")
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.rand(100)
    
    calibrated, report = calibrate_model(y_true, y_prob)
    print(f"Brier Score: {report['brier_score']:.4f}")
    assert "mean_predicted" in report
    print("Calibration [OK]")

def test_trainer_init():
    print("Testing SovereignTrainer Init...")
    trainer = SovereignTrainer()
    print("Trainer Init [OK]")

if __name__ == "__main__":
    try:
        test_ensemble()
        test_calibration()
        test_trainer_init()
        print("\nALL CORE ML COMPONENTS VERIFIED")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {str(e)}")
        sys.exit(1)
