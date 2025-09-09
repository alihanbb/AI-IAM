import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class LabelingMethods:

    def __init__(self, df):
        self.df = df.copy()
        self._check_required_columns()

    def _check_required_columns(self):
        """Etiketleme için gerekli sütunların varlığını kontrol eder."""
        required_cols = ['CreatedAt', 'RiskScore', 'risk_time_anomaly', 'risk_device_change', 
                         'risk_mfa_change', 'risk_app_change', 'risk_ip_change', 
                         'risk_unit_change', 'risk_title_change']
        for col in required_cols:
            if col not in self.df.columns:
                print(f"⚠️'{col}' sütunu bulunamadı. Etiketleme bazı özelliklerden yoksun olacak.")

    def _prepare_temporal_features(self, df):
        """Zamansal özellikleri etiketleme için hazırlar."""
        temp_df = df.copy()
        if 'CreatedAt' in temp_df.columns:
            temp_df['CreatedAt'] = pd.to_datetime(temp_df['CreatedAt'])
            temp_df['hour'] = temp_df['CreatedAt'].dt.hour
            temp_df['weekday'] = temp_df['CreatedAt'].dt.weekday
            # Mesai saatlerini belirle (örnek olarak 9:00 - 18:00 arası)
            temp_df['is_work_hours'] = ((temp_df['hour'] >= 9) & (temp_df['hour'] <= 18)).astype(int)
        return temp_df
    
    def realistic_labeling(self):
        print("🎯 Yöntem 1: Gerçekçi Etiketleme uygulanıyor (Yeniden Yapılandırıldı)...")
        
        if 'RiskScore' not in self.df.columns:
            return pd.Series(['normal'] * len(self.df))
        
        # Risk piramidine uygun mantıksal kuantil eşikleri tanımla
        # Örnek: %20'si 'düşük', %60'ı 'normal', %15'i 'riskli', %5'i 'kritik' olsun.
        # Bu oranlar verinizin doğasına göre ayarlanabilir.
        quantiles = self.df['RiskScore'].quantile([0.20, 0.80, 0.95])
        q_low, q_normal, q_risky = quantiles.values
        
        def assign_realistic_label(score):
            if score <= q_low:
                # En düşük %20'lik dilim 'düşük' olarak etiketlenir.
                return 'düşük'
            elif score <= q_normal:
                # %20 ile %80 arasındaki dilim (%60'lık en geniş kısım) 'normal' olarak etiketlenir.
                return 'normal'
            elif score <= q_risky:
                # %80 ile %95 arasındaki dilim (%15'lik kısım) 'riskli' olarak etiketlenir.
                return 'riskli'
            else:
                # En yüksek %5'lik dilim 'kritik' olarak etiketlenir.
                return 'kritik'
        
        self.df['RealisticLabel'] = self.df['RiskScore'].apply(assign_realistic_label)
        
        distribution = self.df['RealisticLabel'].value_counts(normalize=True) * 100
        print("Yeni Dağılım:")
        for label, percentage in distribution.items():
            print(f"      {label}: %{percentage:.1f}")
            
        return self.df['RealisticLabel']

    def rule_based_labeling(self):
        print("🔧 Yöntem 2: Kural Bazlı Etiketleme uygulanıyor...")
        
        # Risk faktörlerine ağırlık vererek daha nüanslı bir kural tabanlı skorlama
        def assign_rule_based_score(row):
            score = 0
            # Ağırlıklar: IP/MFA Değişimi > Diğer Değişimler > Zamansal Anomali
            if row.get('risk_ip_change', 0) > 0.5:
                score += 3
            if row.get('risk_mfa_change', 0) > 0.5:
                score += 3
            if row.get('risk_app_change', 0) > 0.5:
                score += 1
            if row.get('risk_device_change', 0) > 0.5:
                score += 1
            if row.get('risk_time_anomaly', 0) > 0.5:
                score += 2
            
            # Kritik kombinasyonlar
            if row.get('risk_ip_change', 0) > 0.5 and row.get('risk_time_anomaly', 0) > 0.5:
                score += 2
            
            return score

        self.df['RuleBasedScore'] = self.df.apply(assign_rule_based_score, axis=1)

        def assign_label(score):
            if score >= 6:
                return 'kritik'
            elif score >= 3:
                return 'riskli'
            elif score >= 1:
                return 'normal'
            else:
                return 'düşük'
        
        self.df['RuleBasedLabel'] = self.df['RuleBasedScore'].apply(assign_label)

        distribution = self.df['RuleBasedLabel'].value_counts(normalize=True) * 100
        print("Dağılım:")
        for label, percentage in distribution.items():
            print(f"    {label}: %{percentage:.1f}")
        return self.df['RuleBasedLabel']
    
    def temporal_pattern_labeling(self):
        print("⏰ Yöntem 3: Zamansal Kalıp Etiketleme uygulanıyor...")

        # Önceden hazırlanmış zamansal özellikleri DataFrame'e ekle
        self.df = self._prepare_temporal_features(self.df)

        # Mesai saatleri dışındaki girişlere risk puanı ekleme
        def assign_temporal_score(row):
            score = 0
            # Mesai saatleri dışındaki girişler için puanlama (9:00-18:00)
            if row.get('is_work_hours', 1) == 0:
                score += 1 # Mesai saati dışında olduğu için 1 puan
            
            # Hafta sonu girişleri için ek puan (hafta içi = 0, hafta sonu = 1)
            if row.get('weekday', 0) >= 5:
                 score += 1 # Hafta sonu olduğu için ek 1 puan
            
            # Zaman anomali skoru (Z-Skor) ile birleştirme
            time_anomaly_score = row.get('risk_time_anomaly', 0)
            score += time_anomaly_score * 2 # Zamansal sapmanın etkisi daha yüksek
            
            return score

        self.df['TemporalScore'] = self.df.apply(assign_temporal_score, axis=1)
        
        def assign_temporal_label(score):
            if score >= 3:
                return 'kritik'
            elif score >= 2:
                return 'riskli'
            elif score >= 1:
                return 'normal'
            else:
                return 'düşük'
        
        self.df['TemporalLabel'] = self.df['TemporalScore'].apply(assign_temporal_label)
        
        distribution = self.df['TemporalLabel'].value_counts(normalize=True) * 100
        print("Dağılım:")
        for label, percentage in distribution.items():
            print(f"    {label}: %{percentage:.1f}")
        
        return self.df['TemporalLabel']

    def hybrid_labeling(self):
        print("🧠 Yöntem 4: Hibrit Etiketleme uygulanıyor...")
        # Önce bağımsız etiketleme yöntemlerini uygula
        self.realistic_labeling()
        self.rule_based_labeling()
        self.temporal_pattern_labeling()
        # Risk seviyelerini sayısal değerlere çevir
        risk_mapping = {'düşük': 0, 'normal': 1, 'riskli': 2, 'kritik': 3}
        self.df['RealisticScore_Numeric'] = self.df['RealisticLabel'].map(risk_mapping)
        self.df['RuleBasedScore_Numeric'] = self.df['RuleBasedLabel'].map(risk_mapping)
        self.df['TemporalScore_Numeric'] = self.df['TemporalLabel'].map(risk_mapping)
        def assign_hybrid_label(row):
            # Ağırlıklı ortalama hesapla
            hybrid_score = (
                row['RealisticScore_Numeric'] * 0.4 +
                row['RuleBasedScore_Numeric'] * 0.3 +
                row['TemporalScore_Numeric'] * 0.3
            )
            # En yüksek riski korumak için max(ağırlıklı ortalama, max_skor_yüzdesi)
            max_numeric_score = max(row['RealisticScore_Numeric'], row['RuleBasedScore_Numeric'], row['TemporalScore_Numeric'])
            if max_numeric_score == 3:
                final_score = max(hybrid_score, 2.5)
            elif max_numeric_score == 2:
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
        
        distribution = self.df['HybridLabel'].value_counts(normalize=True) * 100
        print("Hibrit Etiket Dağılımı:")
        for label, percentage in distribution.items():
            print(f"    {label}: %{percentage:.1f}")
            
        return self.df['HybridLabel']