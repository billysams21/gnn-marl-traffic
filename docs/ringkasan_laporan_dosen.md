# Ringkasan Laporan Eksperimen Jaringan Arterial

Angka metrik RL diambil dari performa agen di akhir masa training (10 episode terakhir) karena merepresentasikan agen yang sudah konvergen dan stabil.

---

## 1. Perbandingan Skenario Stable (Normal 6.000 veh/hr)

Skenario ini mewakili kondisi kepadatan urban rata-rata sepanjang hari. Kepadatan (demand) kendaraan digenerasi secara konstan (uniform) selama 1 jam simulasi dengan komposisi Indonesia (35% mobil, 65% motor).

| Algoritma | Avg Reward | Avg Delay (detik) | Avg Queue (kend/lajur) | Avg Throughput | Avg Teleports/ep |
|---|---|---|---|---|---|
| **PKJI Baseline (Uncoordinated)** | -199.04 | 13.75s | 4.25 veh/lane | ~4354 veh | — |
| **Independent DQN (IDQN)** | -25.74 | 1.27s | 2.36 veh/lane | ~5170 veh | 0.00 |
| **GAT-DQN** | -31.63 | 2.37s | 2.85 veh/lane | ~5148 veh | 0.12 |

> Angka RL = eval bersih `--time-to-teleport -1`, 4 episode, dirata-rata atas 4 seed (s42, s123, s77, s111). Dijalankan 3 Juni 2026 via `run_eval_all_final.ps1`.

### Analisis Kondisi Realita
- **PKJI Gagal Tanpa Koordinasi:** Setting lampu merah standar sesuai manual PKJI (siklus ~78s) ternyata gagal menampung volume di koridor arteri. Mengapa? Karena dihitung tanpa mempertimbangkan sinkronisasi simpang (*uncoordinated, offset=0*). Kendaraan yang lepas dari satu lampu hijau akan langsung terjebak di lampu merah simpang berikutnya. Ini menghasilkan antrean panjang dan *waiting time* belasan detik.
- **RL Belajar Koordinasi Implisit:** Baik agen IDQN maupun GAT-DQN berhasil menemukan strategi sinkronisasi gelombang hijau (*green wave*) secara mandiri. Aliran kendaraan melaju jauh lebih mulus, menekan *waiting time* rata-rata menjadi **1-2 detik** saja. Throughput akhir (kendaraan yang sampai tujuan) naik nyaris +600 kendaraan dibanding metode PKJI.

---

## 2. Perbandingan Skenario Peak (Rush-Hour Kritis 9.000 veh/hr)

Skenario ini mewakili jam sibuk (*rush-hour*), di mana gelombang masuk kendaraan (berpusat di menit ke-30) seketika melebihi ambang batas aman kapasitas fisik persimpangan (kondisi mendekati *oversaturation*).

| Algoritma | Avg Reward | Avg Delay (detik) | Avg Queue (kend/lajur) | Avg Throughput | Avg Teleports/ep |
|---|---|---|---|---|---|
| **Independent DQN (IDQN)** | -87.08 | 4.30s | 3.27 veh/lane | ~6128 veh | 0.06 |
| **GAT-DQN** | -121.62 | 22.23s | 4.67 veh/lane | ~6099 veh | 0.12 |

> Angka RL = eval bersih `--time-to-teleport -1`, 4 episode, dirata-rata atas 4 seed (s42, s123, s77, s111). Dijalankan 3 Juni 2026 via `run_eval_all_final.ps1`.
>
> ⚠️ Peak GAT s123 outlier ekstrem (delay 77.87s) tanpa teleport — gridlock total yang tidak bisa diselesaikan tanpa bantuan SUMO. Ini menarik angka avg GAT peak secara signifikan.

### Analisis Kondisi Realita
- **Titik Saturasi Eksponensial:** Volume jaringan bertambah 50% (dari 6000 ke 9000), namun nilai penalti/reward anjlok hampir 3x lipat (-33 menjadi -90). Hal ini sangat natural sesuai teori aliran lalu lintas; ketika jalanan mendekati kapasitas fisik maksimal, setiap tambahan 1 mobil akan merambat mundur menyebabkan penambahan antrean secara eksponensial (spillback).
- **IDQN (Egoisme Lokal):** IDQN secara metrik rata-rata menekan *delay* lebih baik (4.30s vs 22.23s GAT). Namun tanpa teleport, beberapa seed GAT justru lebih stabil — perbedaan antar seed besar, menunjukkan sensitivitas tinggi terhadap kondisi awal di skenario oversaturasi.
- **GAT-DQN (Koordinasi Graph):** Dua seed GAT (s42, s123) mengalami gridlock parah tanpa bantuan teleport SUMO (delay hingga 77.87s), sementara s77 dan s111 justru lebih baik dari IDQN. Ini menunjukkan GAT lebih sensitif terhadap seed di kondisi peak — koordinasi graph bisa sangat efektif atau justru memperparah jika policy belum converge optimal.

---

---

## 3. Skenario Grid Dynamic (Cross-Traffic 2D, 5.400 veh/hr)

Skenario *stress-test* yang menggantikan *arterial unbalanced*. Menggunakan topologi Grid 3x3 (9 simpang saling mengunci) dengan pergerakan bersilangan ekstrem (*cross-traffic*). Dirancang khusus untuk membandingkan GAT dan IDQN pada jaringan 2D yang rentan terhadap *Gridlock Melingkar*.

| Algoritma | Avg Reward | Avg Delay (detik) | Avg Queue (kend/lajur) | Avg Throughput | Teleports (akibat tabrakan) |
|---|---|---|---|---|---|
| **GAT-DQN** | -79.00 | **40.06s** | 3.46 veh/lane | 4.226 veh | 0.44 |
| **Independent DQN (IDQN)** | -1412.92 | 411.22s | 6.88 veh/lane | 3.749 veh | 0.06 |

> Angka RL = eval bersih `--time-to-teleport -1`, 4 episode, dirata-rata atas 4 seed (s42, s123, s77, s111). Dijalankan 4 Juni 2026.

### Analisis
- **Kematian IDQN (Kebutaan Spasial):** IDQN hancur total dengan *delay* rata-rata 411 detik (~7 menit) per kendaraan. Agen yang "buta" akan terus memaksakan lampu hijau saat simpang depan macet, memicu *Gridlock Melingkar* di mana 4 simpang saling mengunci.
- **Keberhasilan GAT-DQN (Koordinasi):** GAT berhasil menahan *delay* di 40 detik. Berkat *Graph Attention*, agen GAT yang melihat simpang depannya penuh akan rela menahan lampu merah untuk mencegah *intersection blocking*. Cincin kematian berhasil dicegah.
- **Kesimpulan Arsitektur:** Pada jaringan dengan satu sumbu jalan yang dominan (Arteri), koordinasi tidak wajib (IDQN cukup). Namun pada topologi simetris (Grid kota) dengan *cross-traffic* seimbang, koordinasi (GAT) bersifat absolut untuk mencegah kelumpuhan total.

---

## Update Eksperimen — v2 (3 Juni 2026)

### Perluasan Seed (4 seed per algo per skenario)

Semua skenario kini dijalankan dengan **4 seed independen: s42, s123, s77, s111**. Berikut angka per-seed dari **eval bersih** (`--time-to-teleport -1`, 4 episode, `run_eval_all_final.ps1`, 3 Juni 2026):

#### Stable — per seed (eval bersih)

| Seed | Algo | Reward | Delay | Queue | Throughput | Teleports/ep |
|---|---|---|---|---|---|---|
| s42 | GAT | -28.87 | 1.09s | 2.02 | 5148 | 0.00 |
| s123 | GAT | -27.55 | 1.32s | 2.02 | 5176 | 0.25 |
| s77 | GAT | -35.92 | 4.08s | 4.35 | 5098 | 0.25 |
| s111 | GAT | -34.19 | 2.99s | 3.01 | 5168 | 0.00 |
| **GAT avg** | | **-31.63** | **2.37s** | **2.85** | **5148** | **0.12** |
| s42 | IDQN | -22.69 | 1.09s | 2.36 | 5098 | 0.00 |
| s123 | IDQN | -25.03 | 1.84s | 2.90 | 5203 | 0.00 |
| s77 | IDQN | -27.20 | 1.07s | 2.02 | 5218 | 0.00 |
| s111 | IDQN | -28.04 | 1.07s | 2.15 | 5162 | 0.00 |
| **IDQN avg** | | **-25.74** | **1.27s** | **2.36** | **5170** | **0.00** |

#### Peak — per seed (eval bersih)

| Seed | Algo | Reward | Delay | Queue | Throughput | Teleports/ep |
|---|---|---|---|---|---|---|
| s42 | GAT | -148.64 | 2.50s | 3.64 | 6036 | 0.25 |
| s123 | GAT | -193.08 | 77.87s ⚠️ | 8.91 | 5970 | 0.25 |
| s77 | GAT | -77.37 | 5.72s | 3.39 | 6133 | 0.00 |
| s111 | GAT | -67.38 | 2.83s | 2.74 | 6256 | 0.00 |
| **GAT avg** | | **-121.62** | **22.23s** | **4.67** | **6099** | **0.12** |
| s42 | IDQN | -91.41 | 8.56s | 3.41 | 6162 | 0.00 |
| s123 | IDQN | -92.30 | 3.12s | 3.24 | 6180 | 0.25 |
| s77 | IDQN | -77.97 | 3.21s | 3.85 | 6000 | 0.00 |
| s111 | IDQN | -86.65 | 2.32s | 2.57 | 6169 | 0.00 |
| **IDQN avg** | | **-87.08** | **4.30s** | **3.27** | **6128** | **0.06** |

> ⚠️ Peak GAT s123: delay 77.87s = gridlock total tanpa bantuan teleport SUMO. Peak GAT s42 juga outlier (reward -148). Kedua seed ini menarik avg GAT peak secara signifikan.

#### Grid Dynamic 3x3 — per seed (eval bersih)

| Seed | Algo | Reward | Delay | Queue | Throughput | Teleports/ep |
|---|---|---|---|---|---|---|
| s42 | GAT | -31.75 | 12.36s | 3.17 | 4272 | 0.25 |
| s123 | GAT | -216.81 | 133.34s | 5.13 | 4042 | 0.25 |
| s77 | GAT | -33.74 | 7.15s | 2.72 | 4316 | 0.25 |
| s111 | GAT | -33.71 | 7.41s | 2.83 | 4276 | 1.00 |
| **GAT avg** | | **-79.00** | **40.06s** | **3.46** | **4226** | **0.44** |
| s42 | IDQN | -1053.58 | 312.94s | 5.84 | 3850 | 0.25 |
| s123 | IDQN | -1364.11 | 425.19s | 7.93 | 3712 | 0.00 |
| s77 | IDQN | -1613.57 | 406.07s | 5.83 | 3798 | 0.00 |
| s111 | IDQN | -1620.41 | 500.68s | 7.94 | 3637 | 0.00 |
| **IDQN avg** | | **-1412.92** | **411.22s** | **6.88** | **3749** | **0.06** |

> ⚠️ IDQN hancur total di semua seed. GAT berhasil menyelamatkan 3 dari 4 seed dari kelumpuhan total (delay di bawah 13 detik). GAT s123 sempat kesulitan, tapi tetap jauh lebih baik dibanding seluruh IDQN.

### Catatan Metodologi Eval

- **Metrik training & eval inline menggunakan `time_to_teleport=300`** — SUMO otomatis teleport kendaraan stuck >300 detik, sehingga angka delay/queue bisa underestimate dan teleport count tidak murni indikator policy.
- **Eval bersih** (`run_eval_all_final.ps1`, 3 Juni 2026): `--time-to-teleport -1`, 4 episode per model, 24 model total. Angka di tabel seksi 1–3 dan tabel per-seed v2 sudah menggunakan hasil ini.

### Status Model

| Skenario | Algo | Seeds selesai 50 ep | Keterangan |
|---|---|---|---|
| Stable | GAT | s42, s123, s77, s111 ✅ | Semua selesai |
| Stable | IDQN | s42, s123, s77, s111 ✅ | Semua selesai |
| Peak | GAT | s42, s123, s77, s111 ✅ | Semua selesai |
| Peak | IDQN | s42, s123, s77, s111 ✅ | Semua selesai |
| Grid Dynamic | GAT | s42, s123, s77, s111 ✅ | Semua selesai |
| Grid Dynamic | IDQN | s42, s123, s77, s111 ✅ | Semua selesai |

---

## Kesimpulan Singkat
1. **RL vs Statis pada Koridor Arteri Utama:** Penerapan *Multi-Agent Reinforcement Learning* sangat efektif memotong *delay* pengemudi >80% dibandingkan siklus statis tak terkoordinasi (PKJI). Pada jaringan dengan satu sumbu pergerakan yang dominan, agen mandiri (**IDQN**) terbukti lebih *robust* dan stabil menghadapi berbagai tingkat kepadatan (*Stable* maupun *Peak*).
2. **Keunggulan Koordinasi pada Jaringan Kompleks (Grid 2D):** Ketika diterapkan pada tata ruang kota yang kompleks (*Grid* 3x3) dengan pergerakan lalu lintas silang yang ekstrem, agen buta (IDQN) akan memicu *Gridlock Melingkar* yang melumpuhkan kota total. Di sinilah **Graph Attention (GAT-DQN)** membuktikan keunggulannya; kemampuannya membaca kondisi simpang tetangga terbukti krusial untuk mencegah efek domino kemacetan fatal.
3. **Implikasi Skala Kota (Sistem Hybrid):** Berdasarkan temuan di atas, implementasi terbaik untuk kontrol lalu lintas satu kota besar adalah sistem hibrida. **GAT-DQN** ideal diaplikasikan pada zona pusat bisnis (CBD) yang berformat *grid* ketat untuk mencegah kelumpuhan silang, sementara **IDQN** diimplementasikan pada jalanan arteri pinggiran (*suburban*) demi efisiensi komputasi dan stabilitas pembelajaran.