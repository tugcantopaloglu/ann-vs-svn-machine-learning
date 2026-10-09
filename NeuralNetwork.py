from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.metrics import Precision, Recall
from keras import ops
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
# eğitimi bitince durdurma işlemi için earlystopping kullanıldı
from keras.callbacks import EarlyStopping
# yapay sinir ağı modelinin oluşturulması, derlenmesi, eğitilmesi ve sonuçlarının değerlendirilmesi işlemleri
class NeuralNetwork:
    # epoch dosya adı oluşturulması
    def __init__(self, epochs):
        self.epochs = str(epochs)+'_epochs'+"/"
        for name in ('loss', 'decision-boundary', 'confusion-matrix'):
            (Path('./results/nn') / name / self.epochs).mkdir(parents=True, exist_ok=True)
    
    # yapay sinir ağı modelinin oluşturulması işlemi (giriş katmanı, gizli katmanlar ve çıkış katmanı)
    def create_ann_model(self, input_dim, hidden_layers=1):
        model = Sequential()
        model.add(Dense(10, input_dim=input_dim, activation='sigmoid'))

        for _ in range(hidden_layers - 1):
            model.add(Dense(10, activation='sigmoid'))

        model.add(Dense(1, activation='sigmoid'))
        return model

    # modelin derlenmesi işlemi (optimizasyon algoritması, loss fonksiyonu ve metriklerin belirlenmesi)
    def compile_model(self, model, learning_rate=0.01):
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall(), self.f1_score])
        
    # f1 score hesaplama işlemi
    def f1_score(self, y_true, y_pred):
        y_true = ops.reshape(ops.cast(y_true, 'float32'), (-1, 1))
        y_pred = ops.cast(y_pred > 0.5, 'float32')
        tp = ops.sum(y_true * y_pred)
        fp = ops.sum((1 - y_true) * y_pred)
        fn = ops.sum(y_true * (1 - y_pred))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        return f1
    
    # modelin eğitilmesi işlemi (eğitim ve doğrulama verileri, epoch ve batch size parametreleri)
    def train_model(self, model, X_train, y_train, X_val, y_val, epochs=100, batch_size=1):
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=0,restore_best_weights=True)
        return model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stopping])
    
    # modelin mini batch ile eğitilmesi işlemi (eğitim ve doğrulama verileri, batch size ve epoch parametreleri)
    def train_with_mini_batch(self, model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100):
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=0,restore_best_weights=True)
        return model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[early_stopping])
    
    # eğitim sırasında elde edilen sonuçların görselleştirilmesi işlemi (eğitim ve doğrulama kayıpları)
    def visualize_training(self, history, file_name='training_plot'):
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='Egitim Kaybi')
        plt.plot(history.history['val_loss'], label='Dogrulama Kaybi')
        plt.title('Eğitim ve Doğrulama Kayıpları')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./results/nn/loss/'+self.epochs+file_name+'.png')
        plt.close()
        #plt.show()
    
    # karar sınırının çizdirilmesi işlemi (model, veri seti ve etiketler)
    def plot_decision_boundary(self, model, X, y, file_name='decision_boundary_plot'):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = (Z > 0.5).astype(int).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
        plt.title('Karar Siniri')
        plt.xlabel('Ozellik 1')
        plt.ylabel('Ozellik 2')
        plt.savefig('./results/nn/decision-boundary/'+self.epochs+file_name+'.png')
        plt.close()
        #plt.show()
    
    # aşırı öğrenme durumunun değerlendirilmesi işlemi (eğitim ve doğrulama kayıpları, model, eğitim ve doğrulama verileri)
    def evaluate_overfitting(self, history, model, X_train, y_train, X_val, y_val, model_name="model"):
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        print("Egitim Kaybi:", train_loss)
        print("Dogrulama Kaybi:", val_loss)

        if val_loss > train_loss:
            print("Asiri ogrenme tespit edildi! Dogrulama kaybi egitim kaybindan buyuk.")
        else:
            print("Asiri ogrenme tespit edilmedi.")

        print("Dogrulama Performansi:")
        val_predictions = (model.predict(X_val) > 0.5).astype(int)
        cm = confusion_matrix(y_val, val_predictions)
        print(cm)
        print(classification_report(y_val, val_predictions))
        
        self.plot_confusion_matrix(cm, file_name=f"{model_name}_confusion_matrix")
    
    # karışıklık matrisinin çizdirilmesi işlemi (karışıklık matrisi)
    def plot_confusion_matrix(self, cm, file_name='confusion_matrix'):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(f"./results/nn/confusion-matrix/"+self.epochs+file_name+".png", dpi=300)
        plt.close()
    
    # en iyi modelin seçilmesi işlemi (Stochastic Gradient Descent ve Mini Batch Gradient Descent modelleri, doğrulama verileri)
    def select_best_model(self, history_sgd, history_mini_batch, sgd_model, mini_batch_model, X_val, y_val):
        sgd_val_loss = min(history_sgd.history['val_loss'])
        mini_batch_val_loss = min(history_mini_batch.history['val_loss'])

        print("SGD Dogrulama Kaybi:", sgd_val_loss)
        print("Mini Batch Dogrulama Kaybi:", mini_batch_val_loss)

        if sgd_val_loss < mini_batch_val_loss:
            print("En iyi model: Stochastic Gradient Descent")
            self.evaluate_overfitting(history_sgd, sgd_model, None, None, X_val, y_val)
            return sgd_model
        else:
            print("En iyi model: Mini Batch Gradient Descent")
            self.evaluate_overfitting(history_mini_batch, mini_batch_model, None, None, X_val, y_val)
            return mini_batch_model
        
