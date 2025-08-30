import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LabelingMethods:

    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()
    
    def prepare_data(self):
        """Veriyi etiketleme için hazırla"""
        # Datetime dönüşümü
        if 'CreatedAt' in self.df.columns:
            self.df['CreatedAt'] = pd.to_datetime(self.df['CreatedAt'])
        
        # Risk skorunun var olup olmadığını kontrol et
        # if 'RiskScore' not in self.df.columns:
        #     print("⚠️ RiskScore bulunamadı. Önce risk skoru hesaplaması yapılmalı.")
    def realistic_labeling(self):
        print("🎯 Yöntem 1: Gerçekçi Etiketleme uygulanıyor...")
        
        # Gerçekçi eşikler (domain knowledge bazlı)
        # %5 kritik, %10 riskli, %25 normal, %60 düşük
        q1 = self.df['RiskScore'].quantile(0.05)  # Alt %5 kritik
        q2 = self.df['RiskScore'].quantile(0.15)  # Alt %15 riskli  
        q3 = self.df['RiskScore'].quantile(0.40)  # Alt %40 normal
        
        def assign_realistic_label(score):
            if score <= q1:
                return 'kritik'
            elif score <= q2:
                return 'riskli'
            elif score <= q3:
                return 'normal'
            else:
                return 'düşük'
        
        self.df['RealisticLabel'] = self.df['RiskScore'].apply(assign_realistic_label)
        
        # Dağılımı göster
        distribution = self.df['RealisticLabel'].value_counts(normalize=True) * 100
        print("Dağılım:")
        for label, percentage in distribution.items():
            print(f"  {label}: %{percentage:.1f}")
        
        return self.df['RealisticLabel']
    
    def rule_based_labeling(self):
        print("🔧 Yöntem 2: Kural Bazlı Etiketleme uygulanıyor...")
        
        def assign_rule_based_label(row):
            # Saat bilgisini al
            hour = row['CreatedAt'].hour if 'CreatedAt' in row and pd.notna(row['CreatedAt']) else 12
            
            # Hafta sonu kontrolü
            is_weekend = row['CreatedAt'].weekday() >= 5 if 'CreatedAt' in row and pd.notna(row['CreatedAt']) else False
            
            # Risk faktörlerini kontrol et
            high_risk_factors = 0
            
            # Zaman riskleri
            if hour < 6 or hour > 22:  # Gece saatleri
                high_risk_factors += 2
            elif hour < 8 or hour > 18:  # Mesai dışı
                high_risk_factors += 1
                
            if is_weekend:
                high_risk_factors += 1
            
            # Teknik risk faktörleri
            risk_columns = [
                'risk_subnet_change', 'risk_device_change', 
                'risk_mfa_change', 'risk_app_change'
            ]
            
            for col in risk_columns:
                if col in row and row[col] > 0.5:
                    high_risk_factors += 1
            
            # Çoklu risk kombinasyonları (kritik durumlar)
            if high_risk_factors >= 4:
                return 'kritik'
            elif high_risk_factors >= 2:
                return 'riskli'
            elif high_risk_factors == 1:
                return 'normal'
            else:
                return 'düşük'
        
        self.df['RuleBasedLabel'] = self.df.apply(assign_rule_based_label, axis=1)
        
        # Dağılımı göster
        distribution = self.df['RuleBasedLabel'].value_counts(normalize=True) * 100
        print("Dağılım:")
        for label, percentage in distribution.items():
            print(f"  {label}: %{percentage:.1f}")
        
        return self.df['RuleBasedLabel']
    
    def temporal_pattern_labeling(self):
        print("⏰ Yöntem 3: Temporal Pattern Etiketleme uygulanıyor...")
        
        # Kullanıcı bazında temporal analiz
        user_patterns = {}
        
        for user_id in self.df['UserId'].unique():
            user_data = self.df[self.df['UserId'] == user_id].copy()
            
            if len(user_data) > 1:
                # Kullanıcının normal giriş saatleri
                hour_mean = user_data['CreatedAt'].dt.hour.mean()
                hour_std = user_data['CreatedAt'].dt.hour.std()
                
                # Kullanıcının normal gün kalıbı
                weekday_rate = (user_data['CreatedAt'].dt.weekday < 5).mean()
                
                user_patterns[user_id] = {
                    'hour_mean': hour_mean,
                    'hour_std': max(hour_std, 1),  # Minimum 1 saat sapma
                    'weekday_rate': weekday_rate
                }
        
        def assign_temporal_label(row):
            user_id = row['UserId']
            current_hour = row['CreatedAt'].hour if pd.notna(row['CreatedAt']) else 12
            is_weekday = row['CreatedAt'].weekday() < 5 if pd.notna(row['CreatedAt']) else True
            
            if user_id not in user_patterns:
                return 'normal'  # Yeni kullanıcı
            
            pattern = user_patterns[user_id]
            
            # Saat sapması kontrolü
            hour_deviation = abs(current_hour - pattern['hour_mean']) / pattern['hour_std']
            
            # Gün kalıbı kontrolü
            day_anomaly = 0
            if is_weekday and pattern['weekday_rate'] < 0.3:  # Hafta sonu çalışanı weekday'de
                day_anomaly = 1
            elif not is_weekday and pattern['weekday_rate'] > 0.8:  # Weekday çalışanı weekend'de
                day_anomaly = 1
            
            # Temporal risk skoru
            temporal_risk = hour_deviation + day_anomaly
            
            if temporal_risk > 3:
                return 'kritik'
            elif temporal_risk > 2:
                return 'riskli'
            elif temporal_risk > 1:
                return 'normal'
            else:
                return 'düşük'
        
        self.df['TemporalLabel'] = self.df.apply(assign_temporal_label, axis=1)
        
        # Dağılımı göster
        distribution = self.df['TemporalLabel'].value_counts(normalize=True) * 100
        print("Dağılım:")
        for label, percentage in distribution.items():
            print(f"  {label}: %{percentage:.1f}")
        
        return self.df['TemporalLabel']
    
    def hybrid_labeling(self):
        print("🧠 Yöntem 4: Hibrit Etiketleme uygulanıyor...")
        
        # Önce diğer yöntemleri uygula
        realistic_labels = self.realistic_labeling()
        rule_labels = self.rule_based_labeling()
        temporal_labels = self.temporal_pattern_labeling()
        
        def assign_hybrid_label(row):
            # Her yöntemden skorları al
            realistic = row['RealisticLabel']
            rule_based = row['RuleBasedLabel']
            temporal = row['TemporalLabel']
            
            # Risk seviyelerini sayısal değerlere çevir
            risk_mapping = {'düşük': 0, 'normal': 1, 'riskli': 2, 'kritik': 3}
            realistic_score = risk_mapping[realistic]
            rule_score = risk_mapping[rule_based]
            temporal_score = risk_mapping[temporal]
            
            hybrid_score = (
                realistic_score * 0.4 +
                rule_score * 0.3 +
                temporal_score * 0.3
            )
    
            max_risk = max(realistic_score, rule_score, temporal_score)

            if max_risk >= 3:  # Kritik
                final_score = max(hybrid_score, 2.5)
            elif max_risk >= 2:  # Riskli
                final_score = max(hybrid_score, 1.5)
            else:
                final_score = hybrid_score
            
            if final_score >= 2.5:
                return 'kritik'
            elif final_score >= 1.5:
                return 'riskli'
            elif final_score >= 0.5:
                return 'normal'
            else:
                return 'düşük'
        
        self.df['HybridLabel'] = self.df.apply(assign_hybrid_label, axis=1)
        
        # Dağılımı göster
        distribution = self.df['HybridLabel'].value_counts(normalize=True) * 100
        print("Dağılım:")
        for label, percentage in distribution.items():
            print(f"  {label}: %{percentage:.1f}")
        
        return self.df['HybridLabel']
    
    def calculate_risk_score(self):
        df = self.df.copy()
        # Zaman anomalisi
        if 'CreatedAt' in df.columns:
            df['CreatedAt'] = pd.to_datetime(df['CreatedAt'], errors='coerce')
            hours = df['CreatedAt'].dt.hour.fillna(0)
            mean_hour = df.groupby('UserId')['CreatedAt'].transform(lambda x: x.dt.hour.mean()).fillna(0)
            df['risk_time'] = (abs(hours - mean_hour) / 24).clip(0, 1)
        else:
            df['risk_time'] = 0.0

        # MFA değişimi
        if 'MFAMethod' in df.columns:
            primary_mfa = df.groupby('UserId')['MFAMethod'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            df['risk_mfa'] = (df['MFAMethod'] != primary_mfa).astype(float).fillna(0)
        else:
            df['risk_mfa'] = 0.0

        # Browser değişimi
        if 'Browser' in df.columns:
            primary_browser = df.groupby('UserId')['Browser'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            df['risk_browser'] = (df['Browser'] != primary_browser).astype(float).fillna(0)
        else:
            df['risk_browser'] = 0.0

        # OS değişimi
        if 'OS' in df.columns:
            primary_os = df.groupby('UserId')['OS'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            df['risk_os'] = (df['OS'] != primary_os).astype(float).fillna(0)
        else:
            df['risk_os'] = 0.0

        # IP değişimi: statik/en çok kullanılan IP ile normalize
        if 'ClientIP' in df.columns:
            ip_counts = df.groupby(['UserId', 'ClientIP']).size().groupby('UserId').idxmax()
            most_used_ip = df['UserId'].map(lambda uid: ip_counts.get(uid, np.nan)[1] if uid in ip_counts else np.nan)
            static_ip = df.groupby('UserId')['ClientIP'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            df['risk_ip'] = df.apply(
                lambda row: 0 if row['ClientIP'] == static_ip[row.name] or row['ClientIP'] == most_used_ip[row.name] else 1,
                axis=1
            )
        else:
            df['risk_ip'] = 0.0

        # Location değişimi
        if 'Unit' in df.columns:
            primary_location = df.groupby('UserId')['Unit'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            df['risk_location'] = (df['Unit'] != primary_location).astype(float).fillna(0)
        else:
            df['risk_location'] = 0.0

        # Unit değişimi
        if 'Unit' in df.columns:
            primary_unit = df.groupby('UserId')['Unit'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            df['risk_unit'] = (df['Unit'] != primary_unit).astype(float).fillna(0)
        else:
            df['risk_unit'] = 0.0

        # Title değişimi
        if 'Title' in df.columns:
            primary_title = df.groupby('UserId')['Title'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            df['risk_title'] = (df['Title'] != primary_title).astype(float).fillna(0)
        else:
            df['risk_title'] = 0.0

        # Diğer feature'lar için örnek (eksikse 0)
        for col in ['SessionDuration', 'FailedAttempts']:
            if col in df.columns:
                df[f'risk_{col.lower()}'] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(0, 1)
            else:
                df[f'risk_{col.lower()}'] = 0.0

        # Tüm risk feature'larını ağırlıklı topla
        weights = {
            'risk_time': 0.13, 'risk_mfa': 0.12, 'risk_browser': 0.11, 'risk_os': 0.10,
            'risk_ip': 0.13, 'risk_location': 0.10, 'risk_unit': 0.09, 'risk_title': 0.07,
            'risk_sessionduration': 0.08, 'risk_failedattempts': 0.07
        }
        score = np.zeros(len(df), dtype=float)
        total_w = sum(weights.values()) or 1.0
        for col, w in weights.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
                score += vals * w
        df['RiskScore'] = (score / total_w) * 100
        return df
