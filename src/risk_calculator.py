import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RiskScoreCalculator:
    """
    Çok bileşenli risk skoru hesaplama sistemi.
    
    Bu sınıf, kullanıcı giriş logları verisini analiz ederek her bir giriş için
    çeşitli faktörlere dayalı olarak bir risk skoru hesaplar.
    """

    def __init__(self, df, config_file=None):
        self.df = df.copy()
        self.risk_weights = self._load_risk_weights(config_file)
        self.user_profiles = {}
        self._prepare_data()

    def _load_risk_weights(self, config_file):
        """
        Risk bileşenleri için varsayılan ağırlıkları yükler.
        Daha sonra bir yapılandırma dosyasından (örneğin JSON) yüklenecek şekilde
        geliştirilebilir.
        """
        # Ağırlıkların toplamı 1'e yakın olmalıdır.
        return {
            'time_anomaly': {'column_name': 'risk_time_anomaly', 'weight': 0.19},
            'device_change': {'column_name': 'risk_device_change', 'weight': 0.10},
            'mfa_change': {'column_name': 'risk_mfa_change', 'weight': 0.16},
            'app_change': {'column_name': 'risk_app_change', 'weight': 0.08},
            'ip_change': {'column_name': 'risk_ip_change', 'weight': 0.13},
            'location': {'column_name': 'risk_location', 'weight': 0.09},
            'session_duration': {'column_name': 'risk_session_duration', 'weight': 0.04},
            'failed_attempts': {'column_name': 'risk_failed_attempts', 'weight': 0.09},
            'combined_temporal': {'column_name': 'risk_combined_temporal', 'weight': 0.04},
            'unit_change': {'column_name': 'risk_unit_change', 'weight': 0.02}, # Ağırlık düşürüldü, location ile çakışıyor
            'title_change': {'column_name': 'risk_title_change', 'weight': 0.02}
        }

    def _prepare_data(self):
        """
        Veriyi analiz için hazırlar. Gerekli sütunları kontrol eder ve veri tipini dönüştürür.
        """
        if 'CreatedAt' not in self.df.columns:
            raise ValueError("DataFrame must contain 'CreatedAt' column.")
        self.df['CreatedAt'] = pd.to_datetime(self.df['CreatedAt'])

    def run_full_analysis(self):
        """
        Tüm risk analiz sürecini çalıştırır ve risk skorları eklenmiş DataFrame'i döndürür.
        """
        print("Kullanıcı profilleri hesaplanıyor...")
        self.calculate_user_profiles()
        print("Risk bileşenleri hesaplanıyor...")
        self.calculate_risk_components()
        print("Nihai risk skoru hesaplanıyor...")
        self.calculate_final_risk_score()
        return self.df

    def calculate_user_profiles(self):
        """
        Her bir kullanıcı için ortalama davranış profili çıkarır.
        """
        self.user_profiles = {}
        for user_id, group in self.df.groupby('UserId'):
            profile = {
                'mean_hour': group['CreatedAt'].dt.hour.mean(),
                'std_hour': group['CreatedAt'].dt.hour.std(),
                'login_count': len(group)
            }
            # Değişim takibi için kategorik setler
            for col in ['Browser', 'OS', 'MFAMethod', 'Application', 'ClientIP', 'Unit', 'Title']:
                profile[f'{col.lower()}_set'] = set(group[col].unique())
            
            self.user_profiles[user_id] = profile

    def calculate_risk_components(self):
        """
        Her bir risk bileşenini hesaplar ve DataFrame'e yeni sütunlar olarak ekler.
        """
        self.df['risk_time_anomaly'] = self.df.apply(self._calc_time_anomaly_zscore, axis=1).clip(0, 1)
        self.df['risk_device_change'] = self.df.apply(self._calc_device_change, axis=1).clip(0, 1)
        self.df['risk_mfa_change'] = self.df.apply(self._calc_mfa_change, axis=1).clip(0, 1)
        self.df['risk_app_change'] = self.df.apply(self._calc_categorical_change, axis=1, column_name='Application').clip(0, 1)
        self.df['risk_ip_change'] = self.df.apply(self._calc_categorical_change, axis=1, column_name='ClientIP').clip(0, 1)
        
        # 'location_change' ve 'unit_change' aynı olduğu için sadece birini kullanıyoruz.
        self.df['risk_unit_change'] = self.df.apply(self._calc_categorical_change, axis=1, column_name='Unit').clip(0, 1)
        self.df['risk_title_change'] = self.df.apply(self._calc_categorical_change, axis=1, column_name='Title').clip(0, 1)

        # Veri setinde olmayan sütunlar için varsayım yapmayın, riski 0 olarak atayın.
        self.df['risk_session_duration'] = 0
        self.df['risk_failed_attempts'] = 0

        # Zaman ve seans süresi risklerini birleştirme (örnek)
        self.df['risk_combined_temporal'] = (self.df['risk_time_anomaly'] + self.df['risk_session_duration']) / 2
        
        # 'risk_location' için özel bir hesaplama olmadığından 'risk_unit_change' ile aynı değeri alabilir
        self.df['risk_location'] = self.df['risk_unit_change']

    def _calc_time_anomaly_zscore(self, row):
        """
        Giriş saatini, kullanıcının ortalama giriş saatine göre Z-Skoru ile değerlendirir.
        """
        user_id = row['UserId']
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            mean_hour = profile['mean_hour']
            std_hour = profile['std_hour']
            hour = row['CreatedAt'].hour
            if profile['login_count'] <= 1 or std_hour == 0:
                # Tek bir giriş veya varyasyon yoksa
                return 0 if hour == mean_hour else 1
            z_score = abs(hour - mean_hour) / std_hour
            # Z-Skorunu 0-1 aralığına normalize etme (örnek olarak tanh veya min-max kullanılabilir)
            return min(z_score / 3, 1)
        return 0.5 # Profil bulunamazsa varsayılan risk

    def _calc_device_change(self, row):
        """
        Kullanıcı için tarayıcı ve işletim sistemi değişimini değerlendirir.
        """
        user_id = row['UserId']
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            
            browser_risk = 0 if row['Browser'] in profile['browser_set'] else 1
            os_risk = 0 if row['OS'] in profile['os_set'] else 1
            
            # Daha güvenli/az güvenli OS geçişlerini puanlama
            secure_os = {'Ubuntu Linux', 'macOS', 'Windows 11'}
            less_secure_os = {'Windows 7', 'Windows XP'}
            if row['OS'] not in profile['os_set']:
                if row['OS'] in secure_os:
                    os_risk = 0.2
                elif row['OS'] in less_secure_os:
                    os_risk = 0.8
            
            return (browser_risk + os_risk) / 2
        return 0.5

    def _calc_mfa_change(self, row):
        """
        Kullanıcının MFA yöntemi değişimini değerlendirir.
        """
        user_id = row['UserId']
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            mfa = row['MFAMethod']
            mfa_set = profile['mfamethod_set']
            
            secure_methods = {'Biometric', 'SecurityKey', 'TOTP'}
            less_secure_methods = {'SMS', 'EmailLink'}

            if mfa in mfa_set:
                return 0
            elif mfa in secure_methods:
                return 0.2
            elif mfa in less_secure_methods:
                return 0.8
            else:
                return 0.5
        return 0.5

    def _calc_categorical_change(self, row, column_name):
        """
        Tekrarlayan kategorik değişim kontrolü için genel fonksiyon.
        """
        user_id = row['UserId']
        if user_id in self.user_profiles:
            profile_set_key = f'{column_name.lower()}_set'
            if profile_set_key in self.user_profiles[user_id]:
                known_values = self.user_profiles[user_id][profile_set_key]
                current_value = row[column_name]
                return 0 if current_value in known_values else 1
        return 0.5

    def calculate_final_risk_score(self):
        """
        Tüm risk bileşenlerini ağırlıklandırarak nihai risk skorunu hesaplar.
        """
        score = np.zeros(len(self.df))
        valid_weights = [info['weight'] for comp, info in self.risk_weights.items() if info['column_name'] in self.df.columns]
        total_weight = sum(valid_weights) or 1.0

        for comp, info in self.risk_weights.items():
            col = info['column_name']
            weight = info['weight']
            if col in self.df.columns:
                vals = pd.to_numeric(self.df[col], errors='coerce').fillna(0).clip(0, 1)
                score += vals * weight
                
        self.df['RiskScore'] = (score / total_weight) * 100
        return self.df['RiskScore']

    def run(self):
        """
        Analiz sürecini başlatan ana fonksiyon.
        """
        return self.run_full_analysis()