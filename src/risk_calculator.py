import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class RiskScoreCalculator:
    """
    Çok bileşenli risk skoru hesaplama sistemi
    """
    
    def __init__(self, df, config_file=None):
        """
        Parametreler:
        df: Analiz edilecek DataFrame
        config_file: Risk ağırlıklarını içeren JSON dosyası (opsiyonel)
        """
        self.df = df.copy()
        self.risk_weights = self.load_risk_weights(config_file)
        self.user_profiles = {}
        
        # Veriyi hazırla
        self.prepare_data()
        
    def load_risk_weights(self, config_file):
        """Risk bileşenlerinin ağırlıklarını yükle"""
        # Varsayılan risk komponenti tanımları
        default_risk_components = {
            'time_anomaly': {
                'column_name': 'risk_time_anomaly',
                'weight': 0.19,  # Artırıldı (0.17 → 0.19)
                'description': 'Kullanıcının normal giriş saatlerinden sapma riski'
            },
            'device_change': {
                'column_name': 'risk_device_change', 
                'weight': 0.10,  # Azaltıldı (0.12 → 0.10)
                'description': 'Browser değişim riski'
            },
            'mfa_change': {
                'column_name': 'risk_mfa_change',
                'weight': 0.18,  # Artırıldı (0.16 → 0.18) - En kritik faktör
                'description': 'MFA yöntemi değişim ve güvenlik seviyesi riski'
            },
            'app_change': {
                'column_name': 'risk_app_change',
                'weight': 0.07,  # Azaltıldı (0.08 → 0.07)
                'description': 'Uygulama değişim riski'
            },
            'ip_change': {
                'column_name': 'risk_ip_change',
                'weight': 0.12,  # Artırıldı (0.10 → 0.12)
                'description': 'IP adresi değişim riski'
            },
            'location': {
                'column_name': 'risk_location',
                'weight': 0.09,  # Artırıldı (0.08 → 0.09)
                'description': 'Coğrafi lokasyon riski'
            },
            'session_duration': {
                'column_name': 'risk_session_duration',
                'weight': 0.05,  # Artırıldı (0.04 → 0.05)
                'description': 'Anormal session süresi riski'
            },
            'failed_attempts': {
                'column_name': 'risk_failed_attempts',
                'weight': 0.11,  # Artırıldı (0.09 → 0.11)
                'description': 'Başarısız giriş denemesi riski'
            },
            'combined_temporal': {
                'column_name': 'risk_combined_temporal',
                'weight': 0.04,  # Aynı
                'description': 'Birleşik temporal anomali riski'
            },
            'unit_change': {
                'column_name': 'risk_unit_change',
                'weight': 0.03,  # Azaltıldı (0.07 → 0.03) - Düşük impact
                'description': 'Departman değişim riski'
            },
            'title_change': {
                'column_name': 'risk_title_change',
                'weight': 0.02,  # Azaltıldı (0.05 → 0.02) - Düşük impact
                'description': 'Pozisyon değişim riski'
            }
        }
        
        if config_file:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # Config yapısına göre yükleme
                if 'risk_weights' in loaded_config:
                    loaded_weights = loaded_config['risk_weights']
                    
                    # Yeni yapıyı destekle (sadece weight değerleri)
                    if isinstance(loaded_weights, dict):
                        for component_name, component_data in default_risk_components.items():
                            if component_name in loaded_weights:
                                if isinstance(loaded_weights[component_name], dict):
                                    # Yeni yapı: {'weight': 0.15, 'column_name': '...'}
                                    default_risk_components[component_name].update(loaded_weights[component_name])
                                else:
                                    # Eski yapı: sadece weight değeri
                                    default_risk_components[component_name]['weight'] = loaded_weights[component_name]
                    
                    print(f"✅ Risk ağırlıkları {config_file} dosyasından yüklendi.")
                    
                    # Yüklenen ağırlıkları göster
                    print("📊 Yüklenen Risk Ağırlıkları:")
                    for name, data in default_risk_components.items():
                        print(f"   {name}: {data['weight']:.3f} ({data['description']})")
                        
                else:
                    print(f"⚠️ {config_file} dosyasında 'risk_weights' anahtarı bulunamadı.")
                    
            except FileNotFoundError:
                print(f"⚠️ {config_file} bulunamadı, varsayılan ağırlıklar kullanılıyor.")
            except Exception as e:
                print(f"⚠️ Config yükleme hatası: {e}, varsayılan ağırlıklar kullanılıyor.")
        
        # Ağırlıkların toplamını kontrol et ve normalize et
        total_weight = sum(comp['weight'] for comp in default_risk_components.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"⚠️ Ağırlık toplamı {total_weight:.3f}, normalize ediliyor...")
            for component_data in default_risk_components.values():
                component_data['weight'] /= total_weight
        
        # Geriye dönük uyumluluk için basit weight dictionary'si oluştur
        simple_weights = {name: data['weight'] for name, data in default_risk_components.items()}
        
        # Component mapping'i saklayalım
        self.risk_components = default_risk_components
        
        return simple_weights
    
    def prepare_data(self):
        """Veriyi risk analizi için hazırla"""
        # Datetime dönüşümü
        if 'CreatedAt' in self.df.columns:
            self.df['CreatedAt'] = pd.to_datetime(self.df['CreatedAt'])
            self.df['hour'] = self.df['CreatedAt'].dt.hour
            self.df['day_of_week'] = self.df['CreatedAt'].dt.dayofweek
            self.df['is_weekend'] = self.df['day_of_week'] >= 5
        
        # Eksik değerleri doldur
        # Kolon isimlerini standartlaştır
        if 'ClientIP' in self.df.columns and 'IPAddress' not in self.df.columns:
            self.df['IPAddress'] = self.df['ClientIP']
        
        self.df = self.df.fillna({
            'IPAddress': 'unknown',
            'MFAMethod': 'none',
            'CreatedAt':  '00:00:00',
            'Application': 'unknown',
            'Browser': 'unknown',
            'OS': 'unknown',
            'Unit': 'unknown',
            'Title': 'unknown'
        })
        
        print(f"📊 Veri hazirlandi: {len(self.df)} kayit")
    
    def create_user_profiles(self):
        """Her kullanici için davraniş profili oluştur"""
        print("👤 Kullanici profilleri oluşturuluyor...")
        for user_id in self.df['UserId'].unique():
            user_data = self.df[self.df['UserId'] == user_id].copy()
            
            if len(user_data) > 1:
                # Temporal patterns
                hour_mean = user_data['hour'].mean()
                hour_std = max(user_data['hour'].std(), 1)  # Min 1 saat sapma
                
                # Working patterns
                weekday_logins = user_data[~user_data['is_weekend']]
                weekend_rate = user_data['is_weekend'].mean()
                
                # Device patterns - Browser ve OS ayrı ayrı analiz
                unique_ips = user_data['IPAddress'].nunique()
                unique_browsers = user_data['Browser'].nunique()
                unique_os = user_data['OS'].nunique()
                
                # Browser patterns
                primary_browser = user_data['Browser'].mode().iloc[0] if len(user_data['Browser'].mode()) > 0 else 'unknown'
                browser_diversity = unique_browsers / len(user_data)
                
                # OS patterns  
                primary_os = user_data['OS'].mode().iloc[0] if len(user_data['OS'].mode()) > 0 else 'unknown'
                os_diversity = unique_os / len(user_data)
                
                # Application patterns
                app_diversity = user_data['Application'].nunique()
                primary_app = user_data['Application'].mode().iloc[0] if len(user_data['Application'].mode()) > 0 else 'unknown'
                
                # Unit/Department patterns
                unit_diversity = user_data['Unit'].nunique()
                primary_unit = user_data['Unit'].mode().iloc[0] if len(user_data['Unit'].mode()) > 0 else 'unknown'

                # Title/Position patterns
                title_diversity = user_data['Title'].nunique()
                primary_title = user_data['Title'].mode().iloc[0] if len(user_data['Title'].mode()) > 0 else 'unknown'

                #MFA patterns
                mfa_diversity = user_data['MFAMethod'].nunique()
                primary_mfa = user_data['MFAMethod'].mode().iloc[0] if len(user_data['MFAMethod'].mode()) > 0 else 'unknown'

                self.user_profiles[user_id] = {
                    'hour_mean': hour_mean,
                    'hour_std': hour_std,
                    'weekend_rate': weekend_rate,
                    'ip_diversity': unique_ips / len(user_data),
                    'browser_diversity': browser_diversity,
                    'primary_browser': primary_browser,
                    'os_diversity': os_diversity,
                    'primary_os': primary_os,
                    'app_diversity': app_diversity,
                    'primary_app': primary_app,
                    'unit_diversity': unit_diversity,
                    'primary_unit': primary_unit,
                    'title_diversity': title_diversity,
                    'primary_title': primary_title,
                    'total_logins': len(user_data),
                    'mfa_diversity': mfa_diversity,
                    'primary_mfa': primary_mfa
                }
        
        print(f"✅ {len(self.user_profiles)} kullanıcı profili oluşturuldu.")
    
    def calculate_time_anomaly_risk(self, row):
        user_id = row['UserId']
        current_hour = row['hour']
        
        if user_id not in self.user_profiles:
            return 0.3  # Yeni kullanıcı için orta risk
        
        profile = self.user_profiles[user_id]
        
        # Z-score hesaplama
        z_score = abs(current_hour - profile['hour_mean']) / profile['hour_std']
        
        # 0-1 arası normalize et (3 sigma = max risk)
        risk = min(z_score / 3.0, 1.0)
        
        return risk
    
    def calculate_device_change_risk(self, row, prev_row):
        user_id = row['UserId']
        current_browser = row['Browser']
        
        if user_id not in self.user_profiles:
            return 0.2  # Yeni kullanıcı için düşük-orta risk
        
        profile = self.user_profiles[user_id]
        
        # Primary browser'dan farklıysa risk hesapla
        if current_browser != profile['primary_browser']:
            # Base risk - browser değişimi
            base_risk = 0.7
            
            # Diversity factor - kullanıcı çok browser kullanıyorsa normal
            diversity_factor = min(profile['browser_diversity'], 0.5)  # Max 0.5 azaltma
            
            # Final risk calculation
            risk = max(base_risk - (diversity_factor * 2), 0.1)  # Min 0.1 risk
            return risk
        
        return 0.0  # Aynı browser, risk yok

        
    def calculate_os_change_risk(self, row):
        user_id = row['UserId']
        current_os = row['OS']
        
        if user_id not in self.user_profiles:
            return 0.2  # Yeni kullanıcı için düşük-orta risk
        
        profile = self.user_profiles[user_id]
        
        # Primary OS'ten farklıysa risk hesapla
        if current_os != profile['primary_os']:
            # Base risk - OS değişimi browser'dan daha kritik
            base_risk = 0.8
            
            # Diversity factor - kullanıcı çok OS kullanıyorsa normal
            diversity_factor = min(profile['os_diversity'], 0.4)  # Max 0.4 azaltma
            
            # Final risk calculation
            risk = max(base_risk - (diversity_factor * 2), 0.2)  # Min 0.2 risk (OS değişimi daha kritik)
            return risk
        
        return 0.0  # Aynı OS, risk yok
    
    def get_mfa_security_level(self, mfa_method):
        """MFA yönteminin güvenlik seviyesini döndür (0-1, yüksek değer = yüksek güvenlik)"""
        mfa_security_levels = {
            # Yüksek güvenlik (0.9-1.0)
            'Hardware Token': 1.0,
            'Smart Card': 0.95,
            'Biometric': 0.90,
            
            # Orta-yüksek güvenlik (0.7-0.89)
            'Mobile App (Push)': 0.85,
            'Mobile App (TOTP)': 0.80,
            'Hardware Key (FIDO2)': 0.85,
            
            # Orta güvenlik (0.5-0.69)
            'SMS': 0.60,
            'Email': 0.50,
            'Voice Call': 0.55,
            
            # Düşük güvenlik (0.2-0.49)
            'Security Questions': 0.30,
            'Backup Codes': 0.40,
            
            # Güvenlik yok (0.0-0.19)
            'none': 0.0,
            'disabled': 0.1,
            'unknown': 0.15
        }
        
        return mfa_security_levels.get(mfa_method, 0.15)  # Default: düşük güvenlik
    
    def calculate_mfa_change_risk(self, row, prev_row):
        user_id = row['UserId']
        current_mfa = row['MFAMethod']
        
        if user_id not in self.user_profiles:
            mfa_security = self.get_mfa_security_level(current_mfa)
            return max(1.0 - mfa_security, 0.1)
        
        profile = self.user_profiles[user_id]
        current_security = self.get_mfa_security_level(current_mfa)
        primary_security = self.get_mfa_security_level(profile['primary_mfa'])        
        # 1. MFA Değişim Riski
        change_risk = 0.0
        if current_mfa != profile['primary_mfa']:
            base_change_risk = 0.6
            diversity_factor = min(profile['mfa_diversity'] / 5.0, 0.3)
            change_risk = max(base_change_risk - diversity_factor, 0.1)
        # 2. MFA Güvenlik Seviyesi Riski
        security_risk = 1.0 - current_security  # Düşük güvenlik = yüksek risk
        # 3. Güvenlik Seviyesi Düşürme Riski (önemli!)
        downgrade_risk = 0.0
        if current_security < primary_security:
            # Güvenlik seviyesi düştü - ekstra risk
            downgrade_amount = primary_security - current_security
            downgrade_risk = downgrade_amount * 1.5  # 1.5x çarpan ile cezalandır
        
        # 4. Birleşik risk hesaplama
        # Change risk: %30, Security risk: %50, Downgrade risk: %20
        combined_risk = (change_risk * 0.3) + (security_risk * 0.5) + (downgrade_risk * 0.2)
        
        return min(combined_risk, 1.0)  # Maksimum 1.0 risk 
    
    def calculate_app_change_risk(self, row):
        user_id = row['UserId']
        current_app = row['Application']

        if user_id not in self.user_profiles:
            return 0.2 
        profile = self.user_profiles[user_id]
        # Primary app'ten farklıysa risk
        if current_app != profile['primary_app']:
            # App diversity'ye göre risk ayarla
            base_risk = 0.7
            diversity_factor = profile['app_diversity'] / 10.0  # Çok app kullanıyorsa normal
            return max(base_risk - diversity_factor, 0.1)
        
        return 0.0
    
    def calculate_ip_change_risk(self, row, prev_row):
        """IP değişim riski (0-1)"""
        if prev_row is None:
            return 0.0
        
        user_id = row['UserId']
        current_ip = row['IPAddress']
        prev_ip = prev_row['IPAddress']
        
        if current_ip != prev_ip:
            # Kullanıcının IP diversity'sine göre risk ayarla
            if user_id in self.user_profiles:
                diversity = self.user_profiles[user_id]['ip_diversity']
                # Çok IP kullanıyorsa düşük risk
                return max(0.8 - diversity, 0.2)
            return 0.8
        
        return 0.0
    
    def calculate_location_risk(self, row):
        """Lokasyon riski (0-1) - IP bazlı tahmin"""
        # Basit IP-based location risk
        # Gerçek uygulamada GeoIP kullanılabilir
        ip = row['IPAddress']
        
        # IP pattern analizi (basit yaklaşım)
        if ip.startswith('192.168') or ip.startswith('10.'):
            return 0.1  # Internal IP - düşük risk
        elif ip.startswith('172.'):
            return 0.2  # Semi-internal
        else:
            return 0.5  # External IP - orta risk
    
    def calculate_session_duration_risk(self, row):
        """Session süre riski (0-1)"""
        # Mock implementation - gerçek uygulamada session data gerekli
        # Rastgele ama mantıklı değerler
        np.random.seed(hash(row['UserId']) % 1000)
        
        # Normal session: 30-240 dakika
        normal_min, normal_max = 30, 240
        
        # Simulated session duration
        session_duration = np.random.lognormal(mean=4, sigma=1)  # Log-normal dağılım
        
        if session_duration < 5:  # Çok kısa session
            return 0.8
        elif session_duration > 480:  # Çok uzun session (8+ saat)
            return 0.6
        elif session_duration < normal_min:
            return 0.4
        elif session_duration > normal_max:
            return 0.3
        else:
            return 0.1
    
    def calculate_failed_attempts_risk(self, row):
        """Başarısız deneme riski (0-1)"""
        # Mock implementation - gerçek uygulamada failed attempt data gerekli
        user_id = row['UserId']
        
        # Simulated failed attempts (hash-based deterministic)
        np.random.seed(hash(f"{user_id}_{row['CreatedAt']}") % 1000)
        failed_attempts = np.random.poisson(lam=0.2)  # Ortalama 0.2 failed attempt
        
        if failed_attempts == 0:
            return 0.0
        elif failed_attempts <= 2:
            return 0.3
        elif failed_attempts <= 5:
            return 0.7
        else:
            return 1.0
    
    def calculate_combined_temporal_risk(self, row):
       
        # Zaman + hafta sonu kombinasyonu
        time_risk = self.calculate_time_anomaly_risk(row)
        weekend_risk = 0.3 if row['is_weekend'] else 0.0
        
        # Off-hours risk
        hour = row['hour']
        if hour < 6 or hour > 22:
            offhours_risk = 0.8
        elif hour < 8 or hour > 18:
            offhours_risk = 0.4
        else:
            offhours_risk = 0.0
        
        # Ağırlıklı kombinasyon
        combined = time_risk * 0.5 + weekend_risk * 0.25 + offhours_risk * 0.25
        
        return min(combined, 1.0)
    
    def calculate_unit_change_risk(self, row):
        """Departman değişim riski (0-1)"""
        user_id = row['UserId']
        current_unit = row['Unit']
        
        if user_id not in self.user_profiles:
            return 0.1  # Yeni kullanıcı için düşük risk
        
        profile = self.user_profiles[user_id]
        
        # Primary unit'ten farklıysa risk
        if current_unit != profile['primary_unit']:
            # Unit diversity'ye göre risk ayarla
            base_risk = 0.6  # Departman değişimi orta risk
            diversity_factor = profile['unit_diversity'] / 5.0  # Çok departman kullanıyorsa normal
            return max(base_risk - diversity_factor, 0.1)
        
        return 0.0
    
    def calculate_title_change_risk(self, row):
        """Pozisyon değişim riski (0-1)"""
        user_id = row['UserId']
        current_title = row['Title']
        
        if user_id not in self.user_profiles:
            return 0.2  # Yeni kullanıcı için düşük-orta risk
        
        profile = self.user_profiles[user_id]
        
        # Primary title'dan farklıysa risk
        if current_title != profile['primary_title']:
            # Title diversity'ye göre risk ayarla
            base_risk = 0.8  # Pozisyon değişimi yüksek risk
            diversity_factor = profile['title_diversity'] / 3.0  # Çok pozisyon kullanıyorsa normal
            return max(base_risk - diversity_factor, 0.2)
        
        return 0.0
    
    def calculate_risk_scores(self):
        """Tüm risk skorlarını hesapla"""
        print("⚖️ Risk skorları hesaplanıyor...")
        
        # Önce kullanıcı profillerini oluştur
        self.create_user_profiles()
        
        # Risk bileşenlerini hesapla
        self.df = self.df.sort_values(['UserId', 'CreatedAt']).reset_index(drop=True)
        
        risk_components = []
        
        for idx, row in self.df.iterrows():
            user_id = row['UserId']
            
            # Önceki kaydı bul (aynı kullanıcının)
            prev_row = None
            if idx > 0 and self.df.iloc[idx-1]['UserId'] == user_id:
                prev_row = self.df.iloc[idx-1]
            
            # Risk bileşenlerini hesapla
            components = {
                'risk_time_anomaly': self.calculate_time_anomaly_risk(row),
                'risk_device_change': self.calculate_device_change_risk(row, prev_row),
                'risk_mfa_change': self.calculate_mfa_change_risk(row, prev_row),
                'risk_app_change': self.calculate_app_change_risk(row),
                'risk_ip_change': self.calculate_ip_change_risk(row, prev_row),
                'risk_location': self.calculate_location_risk(row),
                'risk_session_duration': self.calculate_session_duration_risk(row),
                'risk_failed_attempts': self.calculate_failed_attempts_risk(row),
                'risk_combined_temporal': self.calculate_combined_temporal_risk(row),
                'risk_unit_change': self.calculate_unit_change_risk(row),
                'risk_title_change': self.calculate_title_change_risk(row)
            }
            
            risk_components.append(components)
        
        # DataFrame'e ekle
        risk_df = pd.DataFrame(risk_components)
        self.df = pd.concat([self.df, risk_df], axis=1)
        
        print("✅ Risk bileşenleri hesaplandı.")
    
    def calculate_final_risk_score(self):
        """Final risk skorunu hesapla (0-100 skala)"""
        print("🎯 Final risk skoru hesaplanıyor...")
        
        # Ağırlıklı risk skoru
        weighted_risk = 0
        
        # Yeni yapı kullanarak mapping oluştur
        if hasattr(self, 'risk_components'):
            risk_mapping = {
                data['column_name']: data['weight'] 
                for name, data in self.risk_components.items()
            }
        else:
            # Fallback - eski yapı
            risk_mapping = {
                'risk_time_anomaly': self.risk_weights.get('time_anomaly', 0.18),
                'risk_device_change': self.risk_weights.get('device_change', 0.13),
                'risk_mfa_change': self.risk_weights.get('mfa_change', 0.13),
                'risk_app_change': self.risk_weights.get('app_change', 0.09),
                'risk_ip_change': self.risk_weights.get('ip_change', 0.09),
                'risk_location': self.risk_weights.get('location', 0.09),
                'risk_session_duration': self.risk_weights.get('session_duration', 0.04),
                'risk_failed_attempts': self.risk_weights.get('failed_attempts', 0.09),
                'risk_combined_temporal': self.risk_weights.get('combined_temporal', 0.04),
                'risk_unit_change': self.risk_weights.get('unit_change', 0.08),
                'risk_title_change': self.risk_weights.get('title_change', 0.07)
            }
        
        # Risk skorlarını hesapla
        print("📊 Risk Bileşenleri ve Ağırlıkları:")
        for risk_col, weight in risk_mapping.items():
            if risk_col in self.df.columns:
                component_risk = self.df[risk_col] * weight
                weighted_risk += component_risk
                
                # Ortalama risk değerini göster
                avg_risk = self.df[risk_col].mean()
                print(f"   {risk_col}: ağırlık={weight:.3f}, ort_risk={avg_risk:.3f}")
            else:
                print(f"   ⚠️ {risk_col} kolonu bulunamadı!")
        
        # 0-100 skalaya çevir (yüksek skor = düşük risk)
        self.df['RiskScore'] = (1 - weighted_risk) * 100
        
        # 0-100 arasında sınırla
        self.df['RiskScore'] = self.df['RiskScore'].clip(0, 100)
        
        print("✅ Final risk skoru hesaplandı.")
        
        # İstatistikler
        print(f"📊 Risk Skoru İstatistikleri:")
        print(f"   Ortalama: {self.df['RiskScore'].mean():.1f}")
        print(f"   Medyan: {self.df['RiskScore'].median():.1f}")
        print(f"   Min: {self.df['RiskScore'].min():.1f}")
        print(f"   Max: {self.df['RiskScore'].max():.1f}")
        print(f"   Std: {self.df['RiskScore'].std():.1f}")
    
    def export_risk_analysis(self, filename=None):
        """Risk analiz sonuçlarını export et"""
        if filename is None:
            filename = f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        self.df.to_csv(filename, index=False, sep=';')
        print(f"\n✅ Risk analizi {filename} dosyasına kaydedildi.")
        
        return filename
    
    def save_risk_weights(self, filename=None):
        """Risk ağırlıklarını kaydet"""
        if filename is None:
            filename = f"risk_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Yeni yapıda kaydet
        config_data = {
            "risk_components": self.risk_components if hasattr(self, 'risk_components') else {},
            "risk_weights": self.risk_weights,
            "saved_date": datetime.now().isoformat(),
            "description": "Risk bileşenleri ve ağırlıkları - kolon adları ile birlikte"
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Risk ağırlıkları {filename} dosyasına kaydedildi.")
        print("📋 Kaydedilen yapı:")
        
        if hasattr(self, 'risk_components'):
            for name, data in self.risk_components.items():
                print(f"   {name}: {data['weight']:.3f} -> {data['column_name']}")
        
        return filename
    
    def get_top_risk_factors(self, top_n=5):
        """En yüksek risk faktörlerini göster"""
        print(f"\n🔍 TOP {top_n} RİSK FAKTÖRÜ:")
        print("-" * 30)
        
        # Risk bileşenlerinin ortalama değerlerini hesapla
        risk_columns = [col for col in self.df.columns if col.startswith('risk_')]
        
        risk_averages = []
        for col in risk_columns:
            avg_risk = self.df[col].mean()
            weight = 0.1  # Default weight
            
            # Ağırlığı bul
            if hasattr(self, 'risk_components'):
                for name, data in self.risk_components.items():
                    if data['column_name'] == col:
                        weight = data['weight']
                        break
            
            impact = avg_risk * weight
            risk_averages.append({
                'component': col,
                'avg_risk': avg_risk,
                'weight': weight,
                'impact': impact
            })
        
        # Impact'e göre sırala
        risk_averages.sort(key=lambda x: x['impact'], reverse=True)
        
        # Top N'i göster
        for i, risk in enumerate(risk_averages[:top_n], 1):
            print(f"   {i}. {risk['component']}: impact={risk['impact']:.3f} (risk={risk['avg_risk']:.3f}, weight={risk['weight']:.3f})")

    def run_full_analysis(self):
        """Tam risk analizi pipeline'ı"""
        print("🔄 TAM RİSK ANALİZİ BAŞLIYOR")
        print("="*50)
        
        # 1. Risk skorlarını hesapla
        self.calculate_risk_scores()
        
        # 2. Final skoru hesapla
        self.calculate_final_risk_score()
        
        # 3. Top risk faktörleri göster
        self.get_top_risk_factors()
        
        print(f"\n🎉 Risk analizi tamamlandı!")
        return self.df
