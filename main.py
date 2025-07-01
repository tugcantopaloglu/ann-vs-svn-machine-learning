#Modülerlik sağlanması amacıyla tüm işlemler farklı class'larda yapılıp burada bulunan main classında çağrılmıştır
import DataHandler
import NeuralNetwork
import NeuralNetworkHandler
import SupportVectorMachine
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # veri seti ve Neural Network nesnelerinin oluşturulması
    data = DataHandler.DataHandler()
    data.show_dataset()
    ''' 
    # Neural Network işlemleri 100 epoch için
    print('100 Epoch Islemleri Basliyor...')
    nn_100_epoch = NeuralNetwork.NeuralNetwork(epochs=100)
    nn_handler_100_epoch = NeuralNetworkHandler.NeuralNetworkHandler(data, nn_100_epoch, 100)
    nn_handler_100_epoch.train_and_evaluate()
    nn_handler_100_epoch.evaluate_best_model()
    
    # Neural Network işlemleri 1000 epoch için
    print('1000 Epoch Islemleri Basliyor...')
    nn_1000_epoch = NeuralNetwork.NeuralNetwork(epochs=1000)
    nn_handler_1000_epoch = NeuralNetworkHandler.NeuralNetworkHandler(data, nn_1000_epoch, 1000)
    nn_handler_1000_epoch.train_and_evaluate()
    nn_handler_1000_epoch.evaluate_best_model()
    ''' 
    # Neural Network işlemleri 10000 epoch için (en iyi training sınırını bulmaya çalıştım early stopping uyguluyoruz zaten tüm aşamalarda) (diğerlerine gerek kalmadığı için yorum satırına aldım ancak denemek isterseniz yorum satırını kaldırmanız yeterli tüm dosya yolları hazır) (kod hiç 10000 epocha ulaşmıyor bile o yüzden çok bir performans sorunu oluşturmuyor)
    print('10000 Epoch Islemleri Basliyor...')
    nn_10000_epoch = NeuralNetwork.NeuralNetwork(epochs=10000)
    nn_handler_10000_epoch = NeuralNetworkHandler.NeuralNetworkHandler(data, nn_10000_epoch, 10000)
    nn_handler_10000_epoch.train_and_evaluate()
    nn_handler_10000_epoch.evaluate_best_model()
    
    # Support Vector Machine işlemleri
    svm = SupportVectorMachine.SupportVectorMachine(data)
    svm.train_and_evaluate()
    svm.evaluate_model()
    svm.svm_plot()
    svm.svm_best_model_plot()
    svm.evaluate_best_model_on_test_set()
    
    print('Program bitti...')
    
if __name__ == "__main__":
    main()
