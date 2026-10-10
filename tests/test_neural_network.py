from pathlib import Path

import numpy as np
import pytest
from keras import ops

from NeuralNetwork import NeuralNetwork


def test_binary_f1_does_not_broadcast_flat_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    network = NeuralNetwork(1)
    score = network.f1_score(ops.array([1, 0, 1, 0]), ops.array([[0.9], [0.8], [0.2], [0.1]]))
    assert float(ops.convert_to_numpy(score)) == pytest.approx(0.5)
    assert Path("results/nn/confusion-matrix/1_epochs").is_dir()


def test_public_keras_model_compiles_and_trains_one_epoch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    network = NeuralNetwork(1)
    model = network.create_ann_model(2)
    network.compile_model(model)
    features = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.float32)
    history = network.train_with_mini_batch(model, features, labels, features, labels, batch_size=2, epochs=1)
    assert len(history.history["val_loss"]) == 1
    assert np.isfinite(history.history["val_loss"][0])
    results = model.evaluate(features, labels, verbose=0)
    assert np.isfinite(results).all()


def test_training_methods_use_distinct_batch_sizes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    network = NeuralNetwork(1)

    class FitFixture:
        def fit(self, *args, **kwargs):
            return kwargs

    inputs = np.zeros((2, 2), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.float32)
    stochastic = network.train_model(FitFixture(), inputs, labels, inputs, labels)
    batched = network.train_with_mini_batch(FitFixture(), inputs, labels, inputs, labels)
    assert stochastic["batch_size"] == 1
    assert batched["batch_size"] == 32
    assert stochastic["callbacks"][0].restore_best_weights


def test_early_stopping_restores_weights_from_best_validation_epoch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    network = NeuralNetwork(40)
    model = network.create_ann_model(2)
    model.set_weights([np.zeros_like(weight) for weight in model.get_weights()])
    network.compile_model(model, learning_rate=1.0)
    features = np.zeros((2, 2), dtype=np.float32)
    history = network.train_model(
        model, features, np.ones(2, dtype=np.float32),
        features, np.zeros(2, dtype=np.float32), epochs=40,
    )
    losses = history.history["val_loss"]
    assert 1 < len(losses) < 40
    assert losses[-1] > min(losses) + 0.1
    restored_loss = model.evaluate(features, np.zeros(2, dtype=np.float32), verbose=0)[0]
    assert restored_loss == pytest.approx(min(losses), rel=1e-5)


@pytest.mark.parametrize("labels", [[0, 0], [[0], [0]]])
def test_binary_f1_with_no_positive_labels_is_finite(tmp_path, monkeypatch, labels):
    monkeypatch.chdir(tmp_path)
    network = NeuralNetwork(1)
    score = network.f1_score(ops.array(labels), ops.array([[0.1], [0.5]]))
    assert float(ops.convert_to_numpy(score)) == 0
