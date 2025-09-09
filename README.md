# AI-IAM Deep Learning Authentication Project

## Proje Özeti
Bu proje, kurumsal kimlik ve erişim yönetimi (IAM) için derin öğrenme tabanlı risk analizi ve MFA (Çok Faktörlü Kimlik Doğrulama) yöntem tahmini sunar. Model, kullanıcı davranışlarını ve oturum verilerini analiz ederek risk skorları ve oturum etiketleri üretir.

## Son Güncellemeler ve Değişiklikler

### 1. Veri Seti Güncellemesi
- **Eski:** `Data/mock_login_month_5000.csv` (5.000 kayıt)
- **Yeni:** `Data/login_logs_3_months_25000_rows.csv` (25.000 kayıt)
- Model ve analizler artık 25K satırlık veri seti ile çalışıyor.

### 2. Deep Learning Modeli
- Sadece derin öğrenme (MLP) mimarisi kullanıldı.
- Model parametreleri ve feature engineering güncellendi.
- BatchNormalization, Dropout ve class weighting eklendi.
- 5 katlı çapraz doğrulama ile eğitim yapıldı.
- Model başarı oranı: **%99.44** (ortalama doğruluk)
- En yüksek doğruluk: **%99.7** (5. katlama)
- Model dosyası: `deep_learning_anomaly_model_20250909_221836`

### 3. Feature Engineering ve Risk Skoru Hesaplama
- 22 gelişmiş özellik: saat, gün, browser, OS, uygulama, risk skorları vb.
- Risk skorları artık tüm feature'lar için ayrı ayrı anomali ve davranış analizi ile hesaplanıyor.
- Zaman anomalisi mesai dışı ve hafta sonu etkisini içerir.
- Location değişimi gün içi ve en sık kullanılan lokasyon analizini içerir.
- IP değişimi statik/en çok kullanılan IP ile normalize edilir.
- Session ve failed attempts için istatistiksel ve hareketli ortalama kullanıldı.
- Tüm risk feature'ları 0-1 aralığında normalize edildi.
- Etiketleme yöntemlerinde dengeli dağılım elde edildi.

### 4. Kod Güncellemeleri
- `src/deep_learning_model.py` dosyasında veri yolu ve separator güncellendi.
- Tüm model eğitim ve test süreçleri 25K veri seti ile optimize edildi.
- `src/labeling_methods.py` içinde etiketleme algoritması iyileştirildi ve dengeli dağılım sağlandı.
- Risk hesaplama ve etiketleme modülleri güncellendi.
- Feature importance ve confusion matrix analizi için ek scriptler eklendi.
- 5 katlı çapraz doğrulama eklendi.

### 5. Sonuçlar
- Çapraz doğrulama ile **%99.44 ortalama accuracy** elde edildi.
- Dengeli etiket dağılımı: Düşük: %51.6, Normal: %23.9, Riskli: %18.2, Kritik: %6.3
- Her bir etiketleme yöntemi (gerçekçi, kural bazlı, zamansal) için optimize edilmiş dağılımlar.
- Confusion matrix ve classification report ile detaylı analiz yapıldı.
- Model ve kodlar production için optimize edildi.

## Kullanım
```bash
python src/deep_learning_model.py
```
Model eğitimi ve test sonuçları terminalde raporlanır. Model dosyası otomatik kaydedilir.

## Dosya Açıklamaları
- `src/deep_learning_model.py`: Ana deep learning model kodu
- `src/risk_calculator.py`: Risk skor hesaplama modülü (güncel algoritma)
- `src/labeling_methods.py`: Oturum etiketleme algoritmaları
- `src/feature_analysis_report.py`: Feature importance ve model analizi scripti
- `Data/login_logs_3_months_25000_rows.csv`: Güncel veri seti

## Katkı ve İletişim
Her türlü öneri ve katkı için proje sahibi ile iletişime geçebilirsiniz.
