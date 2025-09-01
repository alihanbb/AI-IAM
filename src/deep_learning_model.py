import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# TensorFlow logging seviyesini ayarla
tf.get_logger().setLevel('ERROR')

class DeepLearningAnomalyDetector:
    
    def __init__(self, data_path, n_splits=5):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.history = None
        self.n_splits = n_splits
        
    def load_and_prepare_data(self):
        """Veriyi yükle ve derin öğrenme için hazırla"""
        print("🧠 DEEP LEARNING MODELİ VERİ HAZIRLAMA")
        print("-" * 30)
    
        if not os.path.exists(self.data_path):
            alt_path = os.path.join("..", self.data_path)
            if os.path.exists(alt_path):
                self.data_path = alt_path
            else:
                raise FileNotFoundError(f"Veri dosyası bulunamadı: {self.data_path}")
            
        self.df = pd.read_csv(self.data_path, sep=',')
        self.df['CreatedAt'] = pd.to_datetime(self.df['CreatedAt'])
        
        print(f"✅ Veri yüklendi: {len(self.df)} kayıt")
        print(f"👥 Kullanıcı sayısı: {self.df['UserId'].nunique()}")
        
        # Risk skorlarını ve etiketlemeyi uygula
        if 'RiskScore' not in self.df.columns:
            print("⚠️ RiskScore kolonu bulunamadı. Risk skorları hesaplanıyor...")
            from risk_calculator import RiskScoreCalculator
            risk_calc = RiskScoreCalculator(self.df)
            self.df = risk_calc.run_full_analysis()
            print("✅ Risk skorları hesaplandı.")
        
        if 'HybridLabel' not in self.df.columns:
            print("🏷️ Hibrit etiketleme uygulanıyor...")
            from labeling_methods import LabelingMethods
            labeler = LabelingMethods(self.df)
            labeler.hybrid_labeling()
            self.df = labeler.df
            print("✅ Hibrit etiketleme tamamlandı.")
        
        # Sınıf dağılımını kontrol et
        class_dist = self.df['HybridLabel'].value_counts(normalize=True) * 100
        print(f"\n📊 Sınıf Dağılımı:")
        for label, pct in class_dist.items():
            print(f"   {label}: %{pct:.1f}")
        
        majority_class = class_dist.max()
        if majority_class > 70:
            print(f"⚠️ Class imbalance tespit edildi! En büyük sınıf: %{majority_class:.1f}")
            print("💡 Class weights kullanılacak...")
        
        return True
    
    def _prepare_features(self, df):
        """DataFrame için özellik mühendisliği uygular"""
        temp_df = df.copy()
        
        # Zaman özellikleri çıkar
        temp_df['hour'] = temp_df['CreatedAt'].dt.hour
        temp_df['weekday'] = temp_df['CreatedAt'].dt.weekday
        temp_df['is_weekend'] = (temp_df['weekday'] >= 5).astype(int)
        temp_df['is_work_hours'] = ((temp_df['hour'] >= 9) & (temp_df['hour'] <= 18)).astype(int)
        
        # Kategorik değişkenleri encode et
        categorical_columns = ['Browser', 'OS', 'Application', 'MFAMethod', 'Unit', 'Title']
        
        for col in categorical_columns:
            if col in temp_df.columns:
                if col not in self.label_encoders:
                    # Eğitim sırasında fit-transform yap
                    le = LabelEncoder()
                    temp_df[f'{col}_encoded'] = le.fit_transform(temp_df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # Test veya tahmin sırasında transform yap
                    le = self.label_encoders[col]
                    temp_df[f'{col}_encoded'] = le.transform(temp_df[col].astype(str))
        
        # Özellikleri seç
        numerical_features = [
            'hour', 'weekday', 'is_weekend', 'is_work_hours',
            'Browser_encoded', 'OS_encoded', 'Application_encoded', 
            'MFAMethod_encoded', 'Unit_encoded', 'Title_encoded'
        ]
        
        # Risk skorlarını da ekle
        risk_columns = [col for col in temp_df.columns if col.startswith('risk_')]
        numerical_features.extend(risk_columns)
        if 'RiskScore' in temp_df.columns:
            numerical_features.append('RiskScore')
            
        available_features = [col for col in numerical_features if col in temp_df.columns]
        
        return temp_df[available_features], available_features
    
    def build_model(self, input_dim, num_classes):
        """Derin öğrenme model mimarisini oluşturur"""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_model(self):
        """
        Stratified K-Fold Cross-Validation ile modeli eğitir ve değerlendirir.
        """
        print(f"\n🚀 MODEL EĞİTİMİ BAŞLIYOR - {self.n_splits} katlı ÇAPRAZ DOĞRULAMA")
        print("-" * 60)
        
        # Feature ve target'ı hazırla
        X_df, features = self._prepare_features(self.df)
        y = LabelEncoder().fit_transform(self.df['HybridLabel'])
        self.label_encoders['target'] = LabelEncoder().fit(self.df['HybridLabel'])
        
        X = X_df.values
        
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Target vector shape: {y.shape}")
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_accuracies = []
        fold_reports = []
        
        for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
            print(f"\n🔬 Katlama {fold}/{self.n_splits}...")
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Feature scaling (Her katlamada yeniden fit et)
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Class weights hesapla
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))
            
            # Model oluştur ve eğit
            self.model = self.build_model(X_train_scaled.shape[1], len(np.unique(y)))
            
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=8, min_lr=1e-7)
            
            self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=150,
                batch_size=16,
                class_weight=class_weight_dict,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Değerlendirme
            test_predictions = self.model.predict(X_test_scaled, verbose=0)
            test_pred_classes = np.argmax(test_predictions, axis=1)
            
            fold_accuracy = accuracy_score(y_test, test_pred_classes)
            fold_accuracies.append(fold_accuracy)
            
            print(f"   Katlama {fold} Doğruluk: %{fold_accuracy*100:.1f}")
            fold_reports.append(classification_report(y_test, test_pred_classes, zero_division=0))
            
        avg_accuracy = np.mean(fold_accuracies)
        print("\n" + "="*60)
        print(f"🎯 Ortalama Doğruluk (Tüm Katlamalar): %{avg_accuracy*100:.2f}")
        print("="*60)
        
        return avg_accuracy

    def _preprocess_new_data(self, new_data):
        """
        Yeni veriler için özellik mühendisliği ve ölçeklendirme uygular.
        Eğitimde kullanılan scaler ve encoder'ları kullanır.
        """
        # Gerekli sütunların varlığını kontrol et
        required_cols = ['UserId', 'CreatedAt', 'MFAMethod', 'ClientIP', 'Application', 'Browser', 'OS', 'Unit', 'Title']
        for col in required_cols:
            if col not in new_data.columns:
                raise ValueError(f"Yeni veri DataFrame'inde '{col}' sütunu eksik.")

        # Risk skorlarını hesapla (Yeni veriler için)
        from risk_calculator import RiskScoreCalculator
        risk_calc_new = RiskScoreCalculator(new_data)
        new_data_with_risk = risk_calc_new.run_full_analysis()
        
        # Özellik mühendisliği uygula
        processed_data, _ = self._prepare_features(new_data_with_risk)
        
        # Veriyi ölçeklendir
        scaled_data = self.scaler.transform(processed_data)
        
        return scaled_data

    def predict(self, new_data):
        """Yeni veriler için tahmin yap"""
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi. Lütfen önce `train_model()` metodunu çalıştırın.")
        if not self.label_encoders or not self.scaler:
            raise ValueError("Pre-processing objeleri (scaler/encoders) bulunamadı. Lütfen önce modelin eğitildiğinden emin olun.")
        
        # Feature engineering ve scaling uygula
        scaled_data = self._preprocess_new_data(new_data)
        
        # Tahmin
        predictions = self.model.predict(scaled_data, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Label'lara dönüştür
        predicted_labels = self.label_encoders['target'].inverse_transform(predicted_classes)
        
        return predicted_labels, predictions
    
    def save_model(self, filepath=None):
        """Modeli kaydet"""
        if self.model is None:
            raise ValueError("Kaydedilecek bir model bulunamadı.")
            
        if filepath is None:
            filepath = f"deep_learning_anomaly_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.model.save(f"{filepath}.keras") # Yeni format
        
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }, f"{filepath}_preprocessors.pkl")
        
        print(f"✅ Model ve ön işleme nesneleri kaydedildi: {filepath}")
        return filepath

    def load_model(self, filepath):
        """Kayıtlı modeli yükler"""
        self.model = keras.models.load_model(f"{filepath}.keras")
        preprocessors = joblib.load(f"{filepath}_preprocessors.pkl")
        self.scaler = preprocessors['scaler']
        self.label_encoders = preprocessors['label_encoders']
        print(f"✅ Model ve ön işleme nesneleri yüklendi: {filepath}")

def main():
    """Ana test fonksiyonu"""
    print("🧠 DEEP LEARNING ANOMALİ TESPİT SİSTEMİ")
    print("="*60)
    
    try:
        data_file = "Data/login_logs_3_months_25000_rows.csv"
        dl_model = DeepLearningAnomalyDetector(data_file)
        
        dl_model.load_and_prepare_data()
        
        avg_accuracy = dl_model.train_model()
        
        model_path = dl_model.save_model()
        
        print(f"\n🎉 MODEL BAŞARIYLA EĞİTİLDİ VE KAYDEDİLDİ!")
        print(f"   Ortalama Doğruluk: %{avg_accuracy*100:.2f}")
        print(f"   Kaydedilen Model Yolu: {model_path}")
             
            
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    main()
