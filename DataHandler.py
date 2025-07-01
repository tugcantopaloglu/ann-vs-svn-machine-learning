import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pickle

class DataHandler:
    # dataset yoksa oluşturulur, varsa yüklenir
    def __init__(self):
        self.create_or_load_dataset()

    # dataset oluşturulur veya yüklenir
    def create_or_load_dataset(self):
        if self.check_dataset_exists():
            self.load_dataset()
            self.load_train_test_validate()
        else:
            self.create_dataset()
    
    # dataset var mı yok mu kontrol et
    def check_dataset_exists(self):
        try:
            with open('./data/dataset.pkl', 'rb') as f:
                return True
        except FileNotFoundError:
            return False
    
    # dataset oluştur ve train, test, validate olarak ayır (60-20-20)  
    def create_dataset(self, n_samples=400, noise=0.2, random_state=42):
        print('Creating dataset...')
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        self.X = X 
        self.y = y
        print('Splitting dataset into train, test and validate sets...')
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.save_dataset(X=X, y=y)
        self.save_train_test_validate(X_train, X_test, X_val, y_train, y_test, y_val)
        
    # dataset kaydet
    def save_dataset(self, X, y, file_path='./data/dataset.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump((X, y), f)
            
    # train, test ve validate kaydet
    def save_train_test_validate(self, X_train, X_test, X_val, y_train, y_test, y_val, file_path='./data/train_test_validate.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)
            
    # dataset al
    def load_dataset(self, file_path='./data/dataset.pkl'):
        print('Loading dataset from file...')
        with open(file_path, 'rb') as f:
            self.X, self.y = pickle.load(f)
            
    # test ve validate al
    def load_train_test_validate(self, file_path='./data/train_test_validate.pkl'):
        print('Loading train, test and validate sets from file...')
        with open(file_path, 'rb') as f:
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = pickle.load(f)
    
    # dataseti görselleştir ve kaydet
    def show_dataset(self,file_name='dataset_plot'):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[self.y == 0, 0], self.X[self.y == 0, 1], c='green', label='Sinif 0', alpha=0.6)
        plt.scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], c='red', label='Sinif 1', alpha=0.6)
        plt.title('Iki Sinifli Veri Kumesi')
        plt.xlabel('Ozellik 1')
        plt.ylabel('Ozellik 2')
        plt.legend()
        plt.savefig('./results/data/'+file_name+'.png')
        #plt.show()

    def get_train_test_validate(self):
        return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val
        
    def get_dataset(self):
        return self.X, self.y