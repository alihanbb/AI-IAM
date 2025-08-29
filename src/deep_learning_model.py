import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# TensorFlow logging seviyesini ayarla
tf.get_logger().setLevel('ERROR')

class DeepLearningAnomalyDetector:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.history = None
        
    def load_and_prepare_data(self):
        """Veriyi yükle ve derin öğrenme için hazırla"""
        print("🧠 DEEP LEARNING MODELİ VERİ HAZIRLAMA")
    
        
        # Veriyi yükle - Absolute path kullan
        import os
        if not os.path.exists(self.data_path):
            # Relative path dene
            alt_path = os.path.join("..", self.data_path)
            if os.path.exists(alt_path):
                self.data_path = alt_path
            else:
                raise FileNotFoundError(f"Veri dosyası bulunamadı: {self.data_path}")
                
        self.df = pd.read_csv(self.data_path, sep=',')
        self.df['CreatedAt'] = pd.to_datetime(self.df['CreatedAt'])
        
        print(f"✅ Veri yüklendi: {len(self.df)} kayıt")
        print(f"👥 Kullanıcı sayısı: {self.df['UserId'].nunique()}")
        
        # Önce risk skorlarını hesapla (eğer yoksa)
        if 'RiskScore' not in self.df.columns:
            print("⚠️ RiskScore kolonu bulunamadı. Risk skorları hesaplanıyor...")
            from risk_calculator import RiskScoreCalculator
            
            risk_calc = RiskScoreCalculator(self.df)
            self.df = risk_calc.run_full_analysis()
            print("✅ Risk skorları hesaplandı.")
        
        # Hibrit etiketleme uygula (eğer yoksa)
        if   'HybridLabel' not in self.df.columns:
            print("🏷️ Hibrit etiketleme uygulanıyor...")
            from labeling_methods import LabelingMethods
            
            labeler = LabelingMethods(self.df)
            hybrid_labels = labeler.hybrid_labeling()
            # DataFrame'i güncelle
            self.df = labeler.df
            print("✅ Hibrit etiketleme tamamlandı.")
        
        # Sınıf dağılımını kontrol et
        class_dist = self.df['HybridLabel'].value_counts(normalize=True) * 100
        print(f"\n📊 Sınıf Dağılımı:")
        for label, pct in class_dist.items():
            print(f"   {label}: %{pct:.1f}")
        
        # Class imbalance kontrolü
        majority_class = class_dist.max()
        if majority_class > 70:
            print(f"⚠️  Class imbalance tespit edildi! En büyük sınıf: %{majority_class:.1f}")
            print("💡 Class weights kullanılacak...")
        
        return True
    
  
    def prepare_features(self):
        
        print("\n🔧 FEATURE ENGİNEERİNG")
        print("-"*30)
        
        # Zaman özellikleri çıkar
        self.df['hour'] = self.df['CreatedAt'].dt.hour
        self.df['weekday'] = self.df['CreatedAt'].dt.weekday
        self.df['is_weekend'] = (self.df['weekday'] >= 5).astype(int)
        self.df['is_work_hours'] = ((self.df['hour'] >= 9) & (self.df['hour'] <= 18)).astype(int)
        
        # Kategorik değişkenleri encode et
        categorical_columns = ['Browser', 'OS', 'Application', 'MFAMethod', 'Unit', 'Title']
        
        for col in categorical_columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
        
        # Numerical features seç
        numerical_features = [
            'hour', 'weekday', 'is_weekend', 'is_work_hours',
            'Browser_encoded', 'OS_encoded', 'Application_encoded', 
            'MFAMethod_encoded', 'Unit_encoded', 'Title_encoded'
        ]
        
        # Risk skorları da ekle (eğer varsa)
        risk_columns = [col for col in self.df.columns if col.startswith('risk_')]
        numerical_features.extend(risk_columns)
        
        # RiskScore'u da ekle
        if 'RiskScore' in self.df.columns:
            numerical_features.append('RiskScore')
        
        # Mevcut olan feature'ları filtrele
        available_features = [col for col in numerical_features if col in self.df.columns]
        
        print(f"✅ {len(available_features)} feature hazırlandı:")
        for i, feature in enumerate(available_features, 1):
            print(f"   {i:2d}. {feature}")
        
        return available_features
    
    def build_model(self, input_dim, num_classes):
        
        print(f"\n🏗️ DEEP LEARNING MODEL OLUŞTURULUYOR")
        print(f"   Input dimension: {input_dim}")
        print(f"   Number of classes: {num_classes}")
        
        # Optimize edilmiş derin mimari
        model = keras.Sequential([
            # Input layer - Daha güçlü başlangıç
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Hidden layers - Daha derin mimari
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Ek katman - Pattern recognition için
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile - optimize edilmiş parametreler
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Daha düşük LR
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model mimarisi oluşturuldu")
        model.summary()
        
        return model
    
    def train_model(self):
        
        print(f"\n🚀 MODEL EĞİTİMİ BAŞLIYOR")
        print("-"*30)
        
        # Feature'ları hazırla
        features = self.prepare_features()
        
        # X ve y hazırla
        X = self.df[features].values
        
        # Label encoding - HybridLabel kullan
        le_target = LabelEncoder()
        y = le_target.fit_transform(self.df['HybridLabel'])
        self.label_encoders['target'] = le_target
        
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Target vector shape: {y.shape}")
        print(f"   Classes: {le_target.classes_}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   Train shape: {X_train_scaled.shape}")
        print(f"   Test shape: {X_test_scaled.shape}")
        
        # Class weights hesapla (class imbalance için ÇOK ÖNEMLİ!)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"   Class weights: {class_weight_dict}")
        
        # Model oluştur
        self.model = self.build_model(X_train_scaled.shape[1], len(np.unique(y)))
        
        # Callbacks - Improved settings
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Increased patience
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive reduction
            patience=8,  # Earlier intervention
            min_lr=1e-7  # Lower minimum
        )
        
        # Model eğitimi
        print(f"\n🔥 Eğitim başlıyor...")
        
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=150,  # Increased epochs
            batch_size=16,  # Smaller batch size for better gradients
            class_weight=class_weight_dict,  # ÇOK ÖNEMLİ!
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Test performance
        test_predictions = self.model.predict(X_test_scaled)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        
        test_accuracy = accuracy_score(y_test, test_pred_classes)
        
        print(f"\n🎯 MODEL PERFORMANSI")
        print(f"   Test Accuracy: %{test_accuracy*100:.1f}")
        
        # Classification report
        print(f"\n📊 DETAYLI PERFORMANS RAPORU:")
        print(classification_report(
            y_test, test_pred_classes, 
            target_names=le_target.classes_,
            zero_division=0
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, test_pred_classes)
        print(f"\n🔍 CONFUSION MATRIX:")
        print(cm)
        
        return test_accuracy
    
    def predict(self, new_data):
        """Yeni veriler için tahmin yap"""
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        # Feature engineering uygula
        processed_data = self.preprocess_new_data(new_data)
        
        # Scale et
        scaled_data = self.scaler.transform(processed_data)
        
        # Tahmin
        predictions = self.model.predict(scaled_data)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Label'lara dönüştür
        predicted_labels = self.label_encoders['target'].inverse_transform(predicted_classes)
        
        return predicted_labels, predictions
    
    def save_model(self, filepath=None):
        """Modeli kaydet"""
        if filepath is None:
            filepath = f"deep_learning_anomaly_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Keras model kaydet
        self.model.save(f"{filepath}.h5")
        
        # Scaler ve encoders kaydet
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }, f"{filepath}_preprocessors.pkl")
        
        print(f"✅ Model kaydedildi: {filepath}")
        return filepath

def main():
    """Ana test fonksiyonu"""
    print("🧠 DEEP LEARNING ANOMALİ TESPİT SİSTEMİ")
    print("="*60)
    
    # Model oluştur ve eğit
    try:
        dl_model = DeepLearningAnomalyDetector("Data/login_logs_3_months_25000_rows.csv")
        
        # Veriyi hazırla
        dl_model.load_and_prepare_data()
        
        # Modeli eğit
        accuracy = dl_model.train_model()
        
        # Modeli kaydet
        model_path = dl_model.save_model()
        
        print(f"\n🎉 DEEP LEARNING MODELİ BAŞARIYLA EĞİTİLDİ!")
        print(f"   Final Accuracy: %{accuracy*100:.1f}")
        print(f"   Model Path: {model_path}")
        
        return dl_model
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return None

if __name__ == "__main__":
    model = main()
