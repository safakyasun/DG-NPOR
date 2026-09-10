# DG-NPOR: mevcut modelden makale sonuçları

Bu çalışma ortamı dört sayfalık makalenin Results bölümünü üretir. **Kayıtlı DG-NPOR modeli ve onun otomatik seçtiği orbitaller sabittir.** Mevcut ParticleNet JetSet uyarlaması, GN2v01 ve DL1dv01 ile aynı ayrılmış nihai test jetlerinde karşılaştırılır.

## Çalıştır

ZIP'i Downloads içine aç:

```bash
conda activate z-mumu-por
cd "$HOME/Downloads/DG_NPOR_FINAL_PAPER"
bash run_paper.sh
```

Python doğrudan aktif conda ortamından seçilir; senin son koşunda bu ortam Python 3.10.20 idi. DG-NPOR ve karşılaştırma kaynakları bu paketin içindedir. Önceki `DG_NPOR_REAL_TEST` klasörüne script eklemen gerekmez.

Bilinen yollar `configs/paths.json` içinde hazır:

- H5: `~/Downloads/ATLAS_JETSET_DG_NPOR_V9_WORKSPACE_3/data/atlas_jetset/mc-flavtag-ttbar-small.h5`
- DG modeli: `~/Downloads/ATLAS_JETSET_DG_NPOR_V9_SELF_CONFIGURING_NN_WORKSPACE/outputs/atlas_jetset_selfconfig_nn_500k/`

Dosyalar bulunduğu yerden okunur. H5 yeniden indirilmez veya kopyalanmaz. Eski model ve sonuç klasörlerinin içeriği değiştirilmez. Eski test tahminleri varsa yeni orbital dışa aktarımıyla doğrulanıp aynen kullanılır; yoksa sabit modelin ayrılmış testinde tahmin üretilir. DG-NPOR baştan eğitilmez; K yeniden seçilmez.

## ParticleNet

Çalıştırıcı Downloads altındaki `ATLAS_JETSET*`, `DG_NPOR*` ve `~/paper` içindeki bitmiş eğitimleri arar. Full mimari, eğitim tamamlanma bilgisi, validation/test jet kimlikleri, etiketler ve veri bölmeleri kontrol edilir. Uyumlu bitmiş `paper` programına öncelik verilir; yalnızca tamamlanmış `quick` full model varsa onun gerçek epoch bilgisi korunarak kullanılır. Modeller test performansına göre seçilmez.

**Uygun bitmiş kayıt bulunamazsa yalnızca full ParticleNet, 20 epoch paper programıyla eğitilir.** Bu aşama Mac'te uzun sürebilir. Önceki cluster eğitiminin bitmiş çıktı klasörü sunucudaysa `--particle-net-dir` ile sunucuda çalıştırabilir veya o klasörü Mac'e kopyalayabilirsin. Kayıtlı skorlarla tablo/figür üretmek için PyTorch gerekmez; yeni ParticleNet eğitimi için gerekir:

```bash
"$CONDA_PREFIX/bin/python" -m pip install -r requirements_training.txt
```

Belirli bir bitmiş ParticleNet klasörünü kullan:

```bash
bash run_paper.sh \
  --particle-net-dir "$HOME/paper/outputs/particlenet_full_500k_paper"
```

Bu örnek server yoludur; Mac'te klasörün gerçek yolunu kullan. Gerekli dosyalar `particle_net_manifest.json`, `particle_net_wp_validation_predictions.csv.gz`, `particle_net_locked_test_predictions.csv.gz`. Yalnız `.pt` ağırlığının bulunması, hangi veri ve split ile eğitildiğini doğrulamaya yeterli kabul edilmez.

Eksik model için eğitim başlatmadan yalnızca mevcut sonuçları kullanmak istersen:

```bash
"$CONDA_PREFIX/bin/python" run_paper.py
```

Yollar değişirse hepsini açıkça verebilirsin:

```bash
bash run_paper.sh \
  --model-dir "/model/klasoru" \
  --h5-file "/veri/mc-flavtag-ttbar-small.h5" \
  --particle-net-dir "/bitmis/particlenet/klasoru"
```

## Makaleye alınacak dosyalar

Çıktı kökü `outputs/paper500k/`.

| Dosya | Kullanım |
|---|---|
| `paper/TABLE_1_COMPARISON.tex` ve `.csv` | DG-NPOR, ParticleNet, GN2, DL1d: AUC ve %70/%77 b-veriminde hafif jet reddi |
| `figures/FIGURE_1_REJECTION.pdf` | Dört yöntemin reddetme eğrileri; ana makale için |
| `figures/FIGURE_2_ORBITALS.pdf` | Mevcut seçilmiş orbitallerin sınıfa göre dağılımları; yer kalırsa |
| `figures/FIGURE_S1_ROC.pdf` | Standart ROC karşılaştırması; ek/alternatif figür |
| `paper/RESULTS_TEXT.tex` | Gerçek çalışma sayılarından üretilen kısa İngilizce Results metni |
| `paper/FUTURE_WORK.tex` | Henüz yapılmamış geliştirmeler için ayrı kısa paragraf |
| `paper/PAPER_NUMBERS.tex` ve `.json` | Sayıları yeniden yazmadan kullanabilmek için |
| `overleaf/` ve `OVERLEAF_RESULTS.zip` | Overleaf'e alınabilecek dosyalar ve hazır include kodları |
| `PAPER_RESULTS.zip` | Tüm figürler, tablolar, metin, istatistik ve ortak test skorları |

**Dört sayfa için önerilen kullanım:** tek ana tablo + `FIGURE_1_REJECTION` + kısa Results metni. Orbital figürünü yer uygunsa ekle; workspace tüm alternatif figürleri otomatik olarak makaleye koymaz. Dosyalar 500 dpi PNG ve vektör PDF olarak üretilir. Mevcut abstract/introduction/method görülmeden makalenin toplam dört sayfaya sığdığı iddia edilmez.

Overleaf'e `OVERLEAF_RESULTS.zip` içeriğini yükle. Gerekli paketler `graphicx`, `booktabs`, `amsmath`. Results konumuna:

```latex
\input{RESULTS_MAIN}
```

Bu dosya section, kısa metin, ana tablo ve tek ana figürü içerir. Orbital dağılımını ayrıca eklemek için `\input{OPTIONAL_ORBITAL_FIGURE}` kullan. `RESULTS_PREVIEW.tex` tek başına derlenebilen önizlemedir. Bilgisayarında `pdflatex` varsa `paper/RESULTS_PREVIEW.pdf` de otomatik oluşur; yoksa Overleaf'te derle.

## İstatistik ve yorum

- Dört yöntemde aynı source_row, event_number ve sınıf etiketleri kullanılır. Değerlendirme rolü kayıtlı `independent_test`tir; az önce incelenen WP-validation sayıları ana tabloya girmez.
- %60/%70/%77/%85 çalışma noktalarının tamamı `statistics/WORKING_POINTS_60_70_77_85.csv` dosyasındadır. Eşik, değerlendirme örneğindeki b skorlarının ilgili kuantilidir. Model/epoch/K seçimi için bu eşikler kullanılmaz.
- AUC için 2000 eşleştirilmiş event bootstrap tekrarı ve %95 aralıklar; reddetme için jet sayımlarına dayalı koşullu %68 Clopper–Pearson aralıkları verilir. Aralıklar yeniden model eğitme değişkenliğini kapsamaz.
- Sıfır yanlış hafif jet durumunda reddetme sonsuz olarak sansürlenir; sonlu alt sınır ve gerçek sayımlar saklanır. Logaritmik eğrilerde sonsuz noktalar sonluymuş gibi çizilmez.
- GN2/DL1d hazır referanslardır; aynı eğitim bütçesi veya aynı girdiler iddia edilmez. Bu discriminant skorları için NLL üretilmez.
- ParticleNet adı, mevcut **JetSet uyarlamasını** ifade eder. Kaynakta EdgeConv komşuluk birleştirmesi max, komşulukta self dahil; batch normalization maskelemeden önce çalışır. Yayınlanan ParticleNet ile birebir yeniden üretim iddiası yapılmaz. Makalede bu uyarlamayı belirt; ayrıntı `paper/COMPARISON_NOTES.md` dosyasına yazılır.
- Veri [CERN JetSet ttbar simülasyonudur](https://opendata.cern.ch/record/93940). Gerçek veri koşusu derken önceki sentetik oyuncak kontroller yerine bu veri setinde çalışmayı kastediyoruz; kaydedilmiş çarpışma verisi değildir.
- Kaynak K_PF ve K_DG'yi ayrı kaydeder. Üç sayısı zorlanmaz. Son doğrulama çıktındaki her iki sayı da 3 idi.
- Önceki deneylerde test sonuçlarının incelenmiş olabileceği geçmişini silmeyiz. Mevcut sabit model raporlanır; bu paket yeni bağımsız bir test seti yarattığını iddia etmez.

Çalışma yeniden başlatıldığında tamamlanmış, kimliği doğrulanan skorlar kullanılır. Yarım kalan ParticleNet eğitimi epoch ortasından devam etmez; eksik deneme ayrı klasörde korunup eğitim yeniden başlar. Başka model/split için aynı çıktı klasörü kullanılmaz; `--output` ile yenisini belirt.

Bu paketin hazırlanması sırasında senin H5/model ağırlıklarına erişim yoktu. Yazılım ve şekil düzeni küçük kontrollü dosyalarla test edildi; gerçek makale sayıları kendi dosyaların üzerinde çalıştırınca oluşur. `verification/` içeriği yazılım doğrulamasıdır, fizik sonucu değildir.
