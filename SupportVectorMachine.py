import DataHandler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# SVM için farklı kernel tipleri ve parametrelerini kullanarak modelleri eğitir ve doğrular
class SupportVectorMachine:
    def __init__(self, data):
        self.data = data
        self.best_model = None
        self.best_kernel = None
        self.best_params = None
        self.best_val_accuracy = -float('inf')
        self.models = []

    # SVM için farklı kernel tipleri ve parametrelerini kullanarak modelleri eğitir ve doğrular
    def train_and_evaluate(self):
        self.models = []
        self.best_val_accuracy = -float('inf')
        kernels = ['linear', 'poly', 'rbf']
        
        for kernel in kernels:
            print(f"{kernel} kernel ile SVM egitimi basliyor...")
            
            if kernel == 'poly':
                degrees = [1, 2, 3, 4, 5, 6] # derece parametresi için farklı değerler
                for degree in degrees:
                    model = SVC(kernel=kernel, degree=degree, C=1, gamma='scale')
                    model.fit(self.data.X_train, self.data.y_train)

                    val_accuracy = model.score(self.data.X_val, self.data.y_val)
                    print(f"Kernel: {kernel}, Derece: {degree}, Dogrulama Dogrulugu: {val_accuracy}")

                    self.models.append((model, kernel, {'degree': degree}, val_accuracy))
                    
                    if val_accuracy > self.best_val_accuracy:
                        self.best_val_accuracy = val_accuracy
                        self.best_model = model
                        self.best_kernel = kernel
                        self.best_params = {'degree': degree}
            elif kernel == 'rbf':
                gammas = [0.1, 0.5, 1, 2, 5] # gamma parametresi için farklı değerler
                for gamma in gammas:
                    model = SVC(kernel=kernel, C=1, gamma=gamma)
                    model.fit(self.data.X_train, self.data.y_train)

                    val_accuracy = model.score(self.data.X_val, self.data.y_val)
                    print(f"Kernel: {kernel}, Gamma: {gamma}, Dogrulama Dogrulugu: {val_accuracy}")

                    self.models.append((model, kernel, {'gamma': gamma}, val_accuracy))

                    if val_accuracy > self.best_val_accuracy:
                        self.best_val_accuracy = val_accuracy
                        self.best_model = model
                        self.best_kernel = kernel
                        self.best_params = {'gamma': gamma}
            else:
                model = SVC(kernel=kernel, C=1, gamma='scale')
                model.fit(self.data.X_train, self.data.y_train)

                val_accuracy = model.score(self.data.X_val, self.data.y_val)
                print(f"Kernel: {kernel}, Dogrulama Dogrulugu: {val_accuracy}")

                self.models.append((model, kernel, {}, val_accuracy))
                
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_model = model
                    self.best_kernel = kernel
                    self.best_params = {}

        print(f"En iyi SVM modeli: Kernel={self.best_kernel}, Parametreler={self.best_params}, Dogrulama Dogrulugu={self.best_val_accuracy}")

    # eğitilen modellerin performanslarını değerlendirir ve confusion matrix görsellerini kaydeder
    def evaluate_model(self):
        print("Egitim Performansi:")
        train_predictions = self.best_model.predict(self.data.X_train)
        cm = confusion_matrix(self.data.y_train, train_predictions)
        self.svm_confusion_matrix(cm, file_name='train_confusion_matrix')
        print(cm)
        print(classification_report(self.data.y_train, train_predictions))

        print("Dogrulama Performansi:")
        val_predictions = self.best_model.predict(self.data.X_val)
        cm = confusion_matrix(self.data.y_val, val_predictions)
        self.svm_confusion_matrix(cm, file_name='val_confusion_matrix')
        print(cm)
        print(classification_report(self.data.y_val, val_predictions))

        print("Test Performansi:")
        test_predictions = self.best_model.predict(self.data.X_test)
        cm = confusion_matrix(self.data.y_test, test_predictions)
        self.svm_confusion_matrix(cm, file_name='test_confusion_matrix')
        print(cm)
        print(classification_report(self.data.y_test, test_predictions))

    # karar sınırlarını görselleştirir ve kaydeder
    def svm_plot(self):
        Path('./results/svm').mkdir(parents=True, exist_ok=True)
        for idx, (model, kernel, params, val_accuracy) in enumerate(self.models):
            x_min, x_max = self.data.X[:, 0].min() - 1, self.data.X[:, 0].max() + 1
            y_min, y_max = self.data.X[:, 1].min() - 1, self.data.X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                 np.arange(y_min, y_max, 0.01))

            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
            plt.scatter(self.data.X[:, 0], self.data.X[:, 1], c=self.data.y, edgecolor='k', cmap=plt.cm.Paired)
            plt.title(f"SVM Karar Siniri (Kernel: {kernel}, Params: {params}, Accuracy: {val_accuracy:.2f})")
            plt.xlabel('Özellik 1')
            plt.ylabel('Özellik 2')
            plt.savefig(f'./results/svm/svm_decision_boundary_{kernel}_{idx}.png')
            plt.close()
            #plt.show()
    
    # karar sınırlarını en iyi model için görselleştirir ve kaydeder
    def svm_best_model_plot(self):
        Path('./results/svm').mkdir(parents=True, exist_ok=True)
        x_min, x_max = self.data.X[:, 0].min() - 1, self.data.X[:, 0].max() + 1
        y_min, y_max = self.data.X[:, 1].min() - 1, self.data.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        Z = self.best_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
        plt.scatter(self.data.X[:, 0], self.data.X[:, 1], c=self.data.y, edgecolor='k', cmap=plt.cm.Paired)
        plt.title(f"En İyi SVM Karar Siniri (Kernel: {self.best_kernel}, Params: {self.best_params})")
        plt.xlabel('Özellik 1')
        plt.ylabel('Özellik 2')
        plt.savefig('./results/svm/svm_best_decision_boundary.png')
        plt.close()
        #plt.show()
    
    # confusion matrix görseli oluşturur ve kaydeder
    def svm_confusion_matrix(self, cm, file_name='confusion_matrix'):
        Path('./results/svm/confusion-matrix').mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(f"./results/svm/confusion-matrix/{file_name}.png", dpi=300)
        print(f"Confusion Matrix gorseli kaydedildi: ./results/confusion-matrix/{file_name}.png")
        plt.close()
    
    # en iyi modelin test setindeki performansını değerlendirir ve confusion matrix görselini kaydeder
    def evaluate_best_model_on_test_set(self):
        print("En iyi modelin test setindeki performansi:")
        test_predictions = self.best_model.predict(self.data.X_test)
        cm = confusion_matrix(self.data.y_test, test_predictions)
        print("Test Confusion Matrix:")
        print(cm)
        print(classification_report(self.data.y_test, test_predictions))
        self.svm_confusion_matrix(cm, file_name='best_model_test_confusion_matrix')
