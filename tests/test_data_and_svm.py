from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC

from DataHandler import DataHandler
from NeuralNetworkHandler import NeuralNetworkHandler
from SupportVectorMachine import SupportVectorMachine


def test_cold_start_creates_data_and_plot_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = DataHandler()
    assert [len(data.X_train), len(data.X_val), len(data.X_test)] == [240, 80, 80]
    data.show_dataset()
    assert Path("results/data/dataset_plot.png").exists()


def test_missing_split_cache_reuses_loaded_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = DataHandler()
    first.create_dataset(n_samples=40, random_state=9)
    Path("data/train_test_validate.pkl").unlink()
    recovered = DataHandler()
    np.testing.assert_array_equal(recovered.X, first.X)
    np.testing.assert_array_equal(recovered.y, first.y)
    assert [len(recovered.X_train), len(recovered.X_val), len(recovered.X_test)] == [24, 8, 8]


def test_complete_cache_round_trip_keeps_dataset_and_split_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = DataHandler()
    restored = DataHandler()
    for original, loaded in zip(first.get_dataset(), restored.get_dataset()):
        np.testing.assert_array_equal(original, loaded)
    for original, loaded in zip(first.get_train_test_validate(), restored.get_train_test_validate()):
        np.testing.assert_array_equal(original, loaded)


def svm_data():
    features, labels = make_moons(n_samples=40, noise=0.1, random_state=42)
    return SimpleNamespace(X=features, y=labels, X_train=features[:24], y_train=labels[:24], X_val=features[24:32], y_val=labels[24:32])


def test_svm_search_trains_each_candidate_once_and_saves_plot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svm = SupportVectorMachine(svm_data())
    svm.train_and_evaluate()
    assert len(svm.models) == 12
    assert sum(kernel == "poly" for _, kernel, _, _ in svm.models) == 6
    assert svm.best_model is not None
    svm.svm_confusion_matrix(np.array([[1, 0], [0, 1]]))
    assert Path("results/svm/confusion-matrix/confusion_matrix.png").exists()


def test_repeated_svm_search_replaces_previous_candidates():
    svm = SupportVectorMachine(svm_data())
    svm.train_and_evaluate()
    first_best = svm.best_model
    svm.train_and_evaluate()
    assert len(svm.models) == 12
    assert svm.best_model is not first_best
    assert svm.best_val_accuracy == max(score for _, _, _, score in svm.models)


def test_svm_retains_a_model_when_every_validation_score_is_zero(monkeypatch):
    monkeypatch.setattr(SVC, "score", lambda self, features, labels: 0)
    svm = SupportVectorMachine(svm_data())
    svm.train_and_evaluate()
    assert svm.best_model is not None
    assert svm.best_val_accuracy == 0


def test_ann_handler_ranks_restored_best_epochs():
    class NetworkFixture:
        def __init__(self):
            self.models = []
            self.validation_losses = [0.1, 0.9]

        def create_ann_model(self, input_dim, hidden_layers):
            model = object()
            self.models.append(model)
            return model

        def compile_model(self, model, learning_rate):
            pass

        def train_model(self, model, *args, **kwargs):
            return SimpleNamespace(history={"val_loss": self.validation_losses})

        def train_with_mini_batch(self, model, *args, **kwargs):
            return SimpleNamespace(history={"val_loss": [0.2, 0.3]})

        def visualize_training(self, *args):
            pass

        def plot_decision_boundary(self, *args):
            pass

        def evaluate_overfitting(self, *args, **kwargs):
            pass

    network = NetworkFixture()
    handler = NeuralNetworkHandler(svm_data(), network, 2)
    handler.train_and_evaluate()
    assert handler.best_model is network.models[0]
    assert handler.best_val_loss == 0.1
    network.validation_losses = [0.4, 0.8]
    handler.train_and_evaluate()
    assert handler.best_model is network.models[7]
    assert handler.best_val_loss == 0.2
