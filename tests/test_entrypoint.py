import os
from pathlib import Path
import subprocess
import sys


def test_importing_entrypoint_does_not_start_the_experiment(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", "import main"], cwd=tmp_path,
        env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "results").exists()


def test_main_completes_with_bounded_training_and_a_small_dataset(tmp_path, monkeypatch):
    import main

    original_data = main.DataHandler.DataHandler
    original_network = main.NeuralNetwork.NeuralNetwork

    class SmallDataset(original_data):
        def create_dataset(self, n_samples=400, noise=0.2, random_state=42):
            return super().create_dataset(n_samples=40, noise=noise, random_state=random_state)

    class BoundedNetwork(original_network):
        def train_model(self, *args, **kwargs):
            kwargs["epochs"] = 1
            return super().train_model(*args, **kwargs)

        def train_with_mini_batch(self, *args, **kwargs):
            kwargs["epochs"] = 1
            return super().train_with_mini_batch(*args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main.DataHandler, "DataHandler", SmallDataset)
    monkeypatch.setattr(main.NeuralNetwork, "NeuralNetwork", BoundedNetwork)
    main.main()
    assert len(list(Path("results/nn/loss/10000_epochs").glob("*.png"))) == 6
    assert len(list(Path("results/nn/decision-boundary/10000_epochs").glob("*.png"))) == 6
    assert len(list(Path("results/nn/confusion-matrix/10000_epochs").glob("*.png"))) == 6
    assert len(list(Path("results/svm").glob("svm_decision_boundary_*.png"))) == 12
    assert Path("results/svm/svm_best_decision_boundary.png").exists()
    assert Path("results/svm/confusion-matrix/best_model_test_confusion_matrix.png").exists()
