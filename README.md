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
- Model başarı oranı: **%94.3** (önceki 5K veri ile %88.7 idi)
- Model dosyası: `deep_learning_anomaly_model_YYYYMMDD_HHMMSS`

### 3. Feature Engineering
- 22 gelişmiş özellik: saat, gün, browser, OS, uygulama, risk skorları vb.
- Risk skorları ve hibrit etiketleme algoritmaları entegre edildi.

### 4. Kod Güncellemeleri
- `src/deep_learning_model.py` dosyasında veri yolu ve separator güncellendi.
- Tüm model eğitim ve test süreçleri 25K veri seti ile optimize edildi.
- Risk hesaplama ve etiketleme modülleri güncellendi.

### 5. Sonuçlar
- Test setinde **%94.3 accuracy** elde edildi.
- Sınıf bazında yüksek precision ve recall değerleri.
- Model ve kodlar production için optimize edildi.

## Kullanım
```bash
python src/deep_learning_model.py
```
Model eğitimi ve test sonuçları terminalde raporlanır. Model dosyası otomatik kaydedilir.

## Dosya Açıklamaları
- `src/deep_learning_model.py`: Ana deep learning model kodu
- `src/risk_calculator.py`: Risk skor hesaplama modülü
- `src/labeling_methods.py`: Oturum etiketleme algoritmaları
- `Data/login_logs_3_months_25000_rows.csv`: Güncel veri seti

## Katkı ve İletişim
Her türlü öneri ve katkı için proje sahibi ile iletişime geçebilirsiniz.
