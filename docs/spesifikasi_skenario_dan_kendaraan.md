# Spesifikasi Jaringan, Skenario, dan Perilaku Kendaraan

Dokumen ini merupakan panduan terpadu (*unified document*) yang mendeskripsikan spesifikasi topologi jalan, desain skenario eksperimen, serta logika perilaku kendaraan (mikroskopik) yang digunakan dalam lingkungan simulasi SUMO pada penelitian ini.

---

## 1. Spesifikasi Perilaku Kendaraan (Vehicle Logic)

Setiap kendaraan di simulasi beroperasi sebagai agen mikroskopik dengan sifat stokastik. Parameter perilaku dikalibrasi mengikuti pedoman PKJI untuk merepresentasikan kondisi lalu lintas Indonesia (Asia Tenggara).

### A. Komposisi dan Geometri Kendaraan
Komposisi didominasi oleh sepeda motor yang memengaruhi panjang efektif antrean:
*   **Mobil (Passenger) — 35%**
    *   Panjang Fisik: 4.5 m
    *   Jarak Aman Depan (`minGap`): 1.0 m
    *   **Panjang Efektif di Antrean:** 5.5 m
*   **Sepeda Motor (Motorcycle) — 65%**
    *   Panjang Fisik: 2.0 m
    *   Jarak Aman Depan (`minGap`): 0.5 m
    *   **Panjang Efektif di Antrean:** 2.5 m

### B. Perilaku Mengemudi (Driver Behavior)
Menggunakan model *car-following* Krauss termodifikasi dan model *lane-changing* **Sublane (SL2015)**.
*   **Kecepatan Maksimal (`maxSpeed`):** Mobil 13.89 m/s (~50 km/h) | Motor 11.11 m/s (~40 km/h).
*   **Akselerasi & Deselerasi:** Motor (Akselerasi 3.5 m/s², Rem 5.0 m/s²) lebih gesit dan reaktif dibandingkan Mobil (Akselerasi 2.6 m/s², Rem 4.5 m/s²).
*   **Ketidaksempurnaan Pengemudi (`sigma`):** Mobil diset lumayan teratur (`sigma` 0.5). Motor diset sangat stokastik/agresif (`sigma` 0.8), melambangkan waktu reaksi yang acak dan sering meragukan.
*   **Lane-Splitting (Khusus Motor):** Motor memiliki atribut jarak aman samping sangat kecil (`minGapLat` 0.2m) dan agresivitas *sublane* tinggi (`lcSublane` 0.5). Hal ini memungkinkan motor menyelundup (*filter-forward*) di sela-sela mobil dan menumpuk di garis depan persimpangan.

---

## 2. Spesifikasi Topologi Jaringan Jalan

Penelitian ini membandingkan kinerja RL pada dua jenis topologi kota yang direpresentasikan dalam graf. Keduanya memiliki jumlah persimpangan yang sama (9 lampu lalu lintas).

### A. Topologi Koridor Arterial (Arterial 3x3)
*   **Bentuk:** 9 persimpangan yang tersusun dalam matriks 3x3. Walaupun berwujud *grid*, jaringan ini memiliki asimetri jarak geometris yang kuat. Jarak antar simpang pada sumbu Timur-Barat sangat panjang (350–450 meter), membentuk **koridor jalan utama (arteri)**. Sedangkan jarak simpang Utara-Selatan sangat pendek (150 meter), bertindak sebagai jalan pengumpan (*feeder/collector*).
*   **Kapasitas Fisik:** Terdiri dari 48 ruas jalan. Total panjang lajur aktif dalam jaringan mencapai **~17.3 kilometer**, dengan kapasitas tampung maksimal statis sekitar **4.959 kendaraan** (asumsi 3.5m per kendaraan campuran).
*   **Sifat Lalu Lintas:** Karena koridor Timur-Barat sangat dominan, pergerakan kendaraan dikonsentrasikan secara linier sepanjang sumbu ini. Ancaman kemacetan (*spillback*) memanjang ke belakang secara deterministik dari hilir ke hulu di jalan utama, dengan minimnya *cross-traffic* diagonal.

### B. Topologi Jaringan Kompleks (Grid 3x3 Simetris)
*   **Bentuk:** 9 persimpangan yang tersusun dalam formasi matriks 3x3 beraturan. Jarak antar simpang simetris (konstan 200 meter) ke segala arah (Utara-Selatan maupun Timur-Barat).
*   **Kapasitas Fisik:** Terdiri dari 48 ruas jalan. Total panjang lajur aktif mencapai **~17.7 kilometer**. Kapasitas tampung maksimal statis jaringan ini adalah sekitar **5.057 kendaraan** secara simultan.
*   **Sifat Lalu Lintas:** Geometri yang simetris membuat tidak ada satu jalan pun yang mendominasi sebagai "arteri". Jaringan ini dirancang untuk memfasilitasi pergerakan bersilangan (*cross-traffic*) dari 4 penjuru secara merata, menjadikannya sangat rentan terhadap *Gridlock Melingkar* jika simpang tidak terkoordinasi.

---

## 3. Desain Skenario Lalu Lintas

### A. Skenario Arterial Stable (Normal)
*   **Topologi:** Arteri 1D.
*   **Volume:** 6.000 kendaraan/jam.
*   **Distribusi Kemunculan:** Konstan/merata (*Uniform*) selama 3.600 detik (1 jam).
*   **Pola Rute:** Simetris. Arah Timur $\to$ Barat sama padatnya dengan Barat $\to$ Timur.
*   **Tujuan:** Menguji stabilitas algoritma membagi waktu hijau secara adil di bawah beban rata-rata siang hari.

### B. Skenario Arterial Peak (Rush-Hour)
*   **Topologi:** Arteri 1D.
*   **Volume:** 9.000 kendaraan/jam.
*   **Distribusi Kemunculan:** Terpusat (*Gaussian/Normal*) dengan puncak di menit ke-30 (`loc=1800, scale=700`).
*   **Pola Rute:** Simetris.
*   **Tujuan:** Menguji *robustness* algoritma saat beban jalan mendadak melampaui kapasitas fisik (*oversaturation*). Skenario ini membuktikan keunggulan reaktivitas IDQN dibandingkan GAT di jalan lurus.

### C. Skenario Arterial Unbalanced (Asimetris / Morning Commute)
*   **Topologi:** Arteri 1D.
*   **Volume:** 6.000 kendaraan/jam.
*   **Distribusi Kemunculan:** Konstan (*Uniform*).
*   **Pola Rute:** Sangat timpang. Arah masuk kota (Timur $\to$ Barat) mendapat bobot 60%, sedangkan arah sebaliknya hanya 10%.
*   **Tujuan:** Mensimulasikan arus berangkat kerja pagi hari. (Skenario ini dinilai kurang menantang untuk komparasi GAT vs IDQN karena beban konflik persimpangan sangat ringan, terbukti dengan *delay* yang kurang dari 1 detik).

### D. Skenario Grid 3x3 Dynamic (Stress-Test Cross-Traffic)
Skenario ini dirancang khusus untuk membunuh agen buta spasial (IDQN) dan membuktikan kemampuan koordinasi *Graph Attention* (GAT).
*   **Topologi:** Grid 3x3 2D.
*   **Volume:** 5.400 kendaraan/jam (Melebihi kapasitas statis jaringan yang hanya ~5.057 kendaraan).
*   **Pola Rute:** Belokan ekstrem. Proporsi kendaraan berbelok kanan/kiri di tengah *grid* sangat tinggi untuk memaksa perpotongan silang (*cross-traffic*).
*   **Moving Bottlenecks:** Kendaraan tidak muncul merata, melainkan dibagi dalam 3 gelombang serangan (*waves*):
    1.  *Wave 1 (Menit 0–20):* 1.800 kendaraan dari arah Utara dan Barat.
    2.  *Wave 2 (Menit 20–40):* 1.800 kendaraan dari arah Selatan dan Timur.
    3.  *Wave 3 (Menit 40–60):* 1.800 kendaraan membombardir dari 4 penjuru secara serentak membidik persimpangan pusat.
*   **Tujuan:** Menciptakan potensi **Gridlock Melingkar / Circular Deadlock**. Jika satu persimpangan gagal menahan laju kendaraan, kemacetan akan saling mengunci (Simpang A $\to$ B $\to$ C $\to$ D $\to$ A). Skenario ini memvalidasi urgensi penggunaan *Graph Attention* di kawasan perkotaan yang padat.