# Eksperimen Stress-Test: Grid 3x3 Dynamic (Oversaturasi 2D)

Dokumen ini merangkum eksperimen khusus untuk menguji hipotesis keunggulan *Graph Attention* (GAT-DQN) dibandingkan agen mandiri (IDQN) pada tata ruang kota yang kompleks (*Grid* 2D) dengan beban lalu lintas silang (*cross-traffic*).

---

## 1. Desain Skenario: Grid 3x3 Dynamic (5.400 veh/hr)

Skenario ini dirancang sebagai "Pembunuh IDQN". Berbeda dengan koridor Arteri yang pergerakannya didominasi satu sumbu memanjang (Timur-Barat), jaringan Grid 3x3 memiliki 9 simpang yang saling mengunci dari 4 arah secara simetris.

### A. Karakteristik Jalan & Rute
*   **Topologi 2D:** 9 simpang (3x3). Total panjang lajur aktif dalam *grid* sekitar 17,7 kilometer.
*   **Kapasitas Fisik:** Pada kemacetan *bumper-to-bumper*, seluruh *grid* hanya mampu menampung maksimal ~5.057 kendaraan secara simultan.
*   **Beban Kendaraan:** 5.400 kendaraan disuntikkan dalam 1 jam (melebihi kapasitas diam, mensyaratkan sirkulasi yang sangat efisien).
*   **Moving Bottlenecks (3 Gelombang):**
    *   *Wave 1 (Menit 0-20, 1.800 veh):* Serbuan dari Barat & Utara menuju Timur & Selatan.
    *   *Wave 2 (Menit 20-40, 1.800 veh):* Serbuan memotong dari Timur & Selatan menuju Barat & Utara.
    *   *Wave 3 (Menit 40-60, 1.800 veh):* Serangan 4 penjuru secara serentak ke arah simpang-simpang tengah.
*   **Belokan Ekstrem (Cross-Traffic):** Rute sengaja didesain dengan probabilitas belok kanan/kiri yang tinggi di tengah *grid*. Ini memaksa pergerakan bersilangan yang rentan menciptakan **Gridlock Melingkar** (Simpang A mengunci B, B mengunci C, C mengunci D, D mengunci kembali A).

### B. Perilaku Kendaraan (Kalibrasi PKJI)
Komposisi kendaraan mencerminkan gaya mengemudi Asia Tenggara (65% Motor, 35% Mobil) menggunakan model *Sublane* (SL2015).
*   **Motor (65%):** Sangat agresif. *Min gap* hanya 0.5m depan, 0.2m samping. Sigmap (ketidaksempurnaan) 0.8. Suka bermanuver *zig-zag* di sela-sela mobil (`lcSublane` 0.5) dan menumpuk di garis depan persimpangan.
*   **Mobil (35%):** Lambat berakselerasi dan lebih kaku, sehingga sering terjebak di tengah persimpangan jika motor di depannya ragu-ragu (*intersection blocking*).

---

## 2. Hasil Evaluasi Bersih (Tanpa Bantuan Teleport SUMO)

Evaluasi dilakukan murni (4 episode per *seed*) dengan fitur evakuasi SUMO dimatikan (`time_to_teleport = -1`). Jika terjadi *gridlock* mati, AI harus menanggung penalti *delay* sepenuhnya.

| Algoritma | Avg Reward | Avg Delay | Avg Queue | Avg Throughput | Teleports (Akibat Tabrakan) |
|---|---|---|---|---|---|
| **IDQN** | -1412.92 | **411.22s** | 6.88 veh/lajur | 3.749 veh | 0.06 |
| **GAT-DQN** | **-79.00** | **40.06s** | 3.46 veh/lajur | **4.226 veh** | 0.44 |

*(Rata-rata 4 seed: 42, 123, 77, 111)*

---

## 3. Analisis Mekanisme Kegagalan & Keberhasilan

### A. Kematian IDQN (Kebutaan Spasial)
*   **Hasil:** Delay meledak hingga **411 detik ( Hampir 7 menit/kendaraan)**. Jaringan mati total terkunci (*deadlock*). 
*   **Penyebab:** Agen IDQN bersifat miopia (hanya melihat simpangnya sendiri). Saat antrean lokalnya menumpuk, ia rakus memberikan lampu hijau. Mobil terdorong maju dan memblokir persimpangan di depannya yang juga sedang macet. Akibat lalu lintas silang, dorongan buta ini menciptakan rantai *gridlock* melingkar. Tidak ada satu pun mobil yang bisa bergerak.

### B. Keberhasilan GAT-DQN (Koordinasi Spasial)
*   **Hasil:** Delay berhasil ditahan di angka **40 detik**. Arus lalu lintas tetap mengalir (throughput >4.200). Jaringan selamat dari kelumpuhan total.
*   **Penyebab:** Mekanisme *Graph Attention* memampukan agen mengintip kondisi simpang tetangganya secara 2D. 
    *   Saat agen A melihat simpang B (di depannya) sedang macet parah, agen A akan **menahan lampu merah** di simpangnya sendiri, meskipun antrean di A sudah panjang.
    *   Pengorbanan lokal ini mencegah mobil dari A masuk memblokir persimpangan B.
    *   Celah ini memberikan waktu bagi B untuk menguras antreannya. Cincin *gridlock* terpotong sebelum terkunci mati.

---

## 4. Kesimpulan Akhir Pemilihan Arsitektur MARL

Eksperimen ini menjustifikasi penggunaan *Graph Neural Network* pada pengendalian lalu lintas:

1.  **Arteri dengan Arus Dominan → IDQN:** Ancaman kemacetan bersifat deterministik searah (dari belakang ke depan di sepanjang jalan utama). Koordinasi GAT berlebihan (*overkill*) dan justru memicu *variance* tinggi. Pendekatan reaktif lokal (IDQN) lebih stabil.
2.  **Grid Kota Kompleks (2D) → GAT-DQN:** Ancaman kemacetan datang dari segala arah membentuk cincin mati (*circular deadlock*). Visibilitas tetangga via *attention mechanism* menjadi syarat mutlak agar agen bisa berkorban secara lokal demi mencegah kelumpuhan jaringan secara makro.
3.  **Sistem Hybrid (Rekomendasi Skala Kota):** Di lingkungan dunia nyata, sebuah kota besar memiliki kombinasi topologi Grid di pusat CBD dan jalan Arteri di pinggiran kota. Implementasi yang paling optimal adalah memadukan kedua algoritma ini: menggunakan **GAT-DQN secara terpusat pada persimpangan padat di pusat kota** dan **IDQN pada jalan pinggiran** untuk menjaga keseimbangan antara efisiensi komputasi dan kemampuan memecah *gridlock*.