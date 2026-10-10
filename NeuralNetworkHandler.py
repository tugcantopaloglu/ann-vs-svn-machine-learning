import DataHandler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetworkHandler:
    # veri seti ve yapay sinir ağı nesnelerinin oluşturulması
    def __init__(self, data, nn, epochs):
        self.data = data
        self.nn = nn
        self.best_model = None
        self.best_val_loss = float('inf')
        self.best_method = None
        self.best_hidden_layers = None
        self.epochs = epochs

    # eğitim ve değerlendirme işlemlerinin yapılması ve sonuçların yazdırılması işlemi (her bir gizli katman için)
    def train_and_evaluate(self):
        self.best_model = None
        self.best_val_loss = float('inf')
        self.best_method = None
        self.best_hidden_layers = None
        # her bir gizli katman için yapay sinir ağı modelinin oluşturulması, eğitilmesi ve değerlendirilmesi işlemleri
        for i in range(1, 4):
            print(f"Stochastic Gradient Descent ile egitim basliyor (Gizli Katmanlar: {i})...")
            ann_model = self.nn.create_ann_model(input_dim=self.data.X_train.shape[1], hidden_layers=i)
            self.nn.compile_model(ann_model, learning_rate=0.01)
            ann_model_history = self.nn.train_model(ann_model, self.data.X_train, self.data.y_train, self.data.X_val, self.data.y_val, epochs=self.epochs)
            val_loss = min(ann_model_history.history['val_loss'])

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = ann_model
                self.best_method = "Stochastic Gradient Descent"
                self.best_hidden_layers = i

            self.nn.visualize_training(ann_model_history, f'training_plot_sgd_{i}_katman')
            self.nn.plot_decision_boundary(ann_model, self.data.X, self.data.y, f'decision_boundary_plot_sgd_{i}_katman')
            self.nn.evaluate_overfitting(ann_model_history, ann_model, self.data.X_train, self.data.y_train, self.data.X_val, self.data.y_val, model_name=f"model_sgd_{i}_katman")

            print(f"Mini Batch Gradient Descent ile egitim basliyor (Gizli Katmanlar: {i})...")
            ann_model_mini_batch = self.nn.create_ann_model(input_dim=self.data.X_train.shape[1], hidden_layers=i)
            self.nn.compile_model(ann_model_mini_batch, learning_rate=0.01)
            ann_model_mini_batch_history = self.nn.train_with_mini_batch(ann_model_mini_batch, self.data.X_train, self.data.y_train, self.data.X_val, self.data.y_val, batch_size=32, epochs=self.epochs)
            val_loss_mini_batch = min(ann_model_mini_batch_history.history['val_loss'])

            if val_loss_mini_batch < self.best_val_loss:
                self.best_val_loss = val_loss_mini_batch
                self.best_model = ann_model_mini_batch
                self.best_method = "Mini Batch Gradient Descent"
                self.best_hidden_layers = i

            self.nn.visualize_training(ann_model_mini_batch_history, f'training_plot_mini_batch_{i}_katman')
            self.nn.plot_decision_boundary(ann_model_mini_batch, self.data.X, self.data.y, f'decision_boundary_plot_mini_batch_{i}_katman')
            self.nn.evaluate_overfitting(ann_model_mini_batch_history, ann_model_mini_batch, self.data.X_train, self.data.y_train, self.data.X_val, self.data.y_val, model_name=f"model_mini_batch_{i}_katman")

        print(f"En iyi model: {self.best_method} ile {self.best_hidden_layers} gizli katman, Dogrulama Kaybi: {self.best_val_loss}")

    # en iyi modelin test verisi üzerinde değerlendirilmesi ve sonuçların yazdırılması işlemi
    def evaluate_best_model(self):
        evaluation_results = self.best_model.evaluate(self.data.X_test, self.data.y_test)
        test_loss = evaluation_results[0]
        test_accuracy = evaluation_results[1]
        print(f"Test Sonuclari: Kayip = {test_loss}, Dogruluk = {test_accuracy}")

        print("Egitim Performansi:")
        train_predictions = (self.best_model.predict(self.data.X_train) > 0.5).astype(int)
        print(confusion_matrix(self.data.y_train, train_predictions))
        print(classification_report(self.data.y_train, train_predictions))

        print("Dogrulama Performansi:")
        val_predictions = (self.best_model.predict(self.data.X_val) > 0.5).astype(int)
        print(confusion_matrix(self.data.y_val, val_predictions))
        print(classification_report(self.data.y_val, val_predictions))

        print("Test Performansi:")
        test_predictions = (self.best_model.predict(self.data.X_test) > 0.5).astype(int)
        print(confusion_matrix(self.data.y_test, test_predictions))
        print(classification_report(self.data.y_test, test_predictions))
