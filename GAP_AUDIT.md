# Audit Gap: Implementasi vs Proposal TA

Tanggal: 18 April 2026  
Referensi: Bab IV (Desain Konsep Solusi), Bab V (Rencana Selanjutnya), ARSITEKTUR_GNN_MARL.md

Catatan: Dokumen ini merefleksikan status terbaru setelah sinkronisasi narasi Bab IV terhadap implementasi saat ini.

---

## A. GAP BESAR (Algoritmik / Desain)

### Gap 1: Action Space — SUDAH DISELARASKAN DI NARASI

**Proposal (Bab IV 4.4):**
> Penelitian ini menggunakan pendekatan Cyclic: A = {Keep, Change}.
> Jika Change, fase berubah secara siklis (1 → 2 → 3 → ... → 1).
> Setiap Δt detik, agen memutuskan apakah tetap pada fase saat ini atau beralih ke fase berikutnya.

**Implementasi saat ini:**
- `num_actions = num_phases` (misal 2 untuk grid sederhana)
- Aksi = pilih fase target secara langsung (Acyclic/phase selection)
- Tidak ada logika "Keep" vs "Change"

**Update status (18 April 2026):** Narasi di Bab IV sudah diperbarui agar konsisten dengan implementasi utama berbasis *Acyclic*, sementara *Cyclic* diposisikan sebagai opsi *ablation*.

**Dampak (setelah update):** Tidak lagi menjadi mismatch proposal vs implementasi.

**Status: ⚠️ BEST PRACTICE?**
- **Proposal Cyclic: BUKAN best practice.** Mayoritas paper ATSC modern (MPLight, CoLight, PressLight, AttendLight) menggunakan **Acyclic (phase selection)** karena lebih fleksibel dan menghasilkan performa lebih baik.
- Argumen "safety" di proposal lemah — dalam simulasi SUMO, transisi fase selalu melalui yellow phase sehingga tidak ada bahaya. Safety hanya relevan di deployment real-world.
- **Rekomendasi terbaru:** Pertahankan implementasi Acyclic sebagai default. Tambahkan varian Cyclic hanya jika diperlukan untuk eksperimen pembanding.

---

### Gap 2: Normalisasi State Belum Penuh

**Proposal (Bab IV §4.2, versi terbaru):**
> Narasi sudah diselaraskan dengan implementasi saat ini: normalisasi masih parsial
> (kepadatan dan durasi fase), sedangkan komponen antrean/waktu tunggu/perubahan
> masih nilai mentah.

**Implementasi saat ini (`sumo_env.py` `_get_observation`):**
- `queue` = raw `getLastStepHaltingNumber()` → range [0, ∞)
- `delta_queue` = raw difference → range (-∞, ∞)
- `waiting` = raw `getWaitingTime()` → range [0, ∞) (bisa ratusan detik)
- `density` = di-clip ke [0,1] ✅
- `phase_duration` = dinormalisasi / max_green ✅

**Dampak:** Input features dengan skala sangat berbeda (queue ~0-30, waiting ~0-500) menyebabkan gradient instability. Neural network harus belajar scale sendiri lewat first layer weights.

**Status: ⚠️ BEST PRACTICE?**
- **Masih penting ditingkatkan.** Walaupun narasi dan implementasi sudah konsisten, normalisasi penuh tetap best practice untuk stabilitas training.
- **Rekomendasi:** Implementasikan normalisasi tambahan bertahap (queue, delta, waiting) sebagai peningkatan kualitas model.

---

### Gap 3: Pre-projection MLP Tidak Ada

**Proposal (Bab IV 4.3):**
> Sebelum masuk ke GAT, observasi terlebih dahulu diproyeksikan:
> h_i^(0) = MLP_enc(o_i), h_i^(0) ∈ R^64

**Implementasi saat ini (`gat_encoder.py`):**
- Raw observation (obs_dim ≈ 43) langsung masuk ke `GATConv` Layer 1
- Tidak ada `MLP_enc` terpisah

**Dampak:** Kecil secara praktis. GATConv Layer 1 secara internal melakukan linear projection `W · o_i` yang fungsinya mirip MLP_enc. Perbedaannya: MLP_enc bisa multi-layer dengan nonlinearity, sedangkan GATConv internal projection hanya 1 linear layer tanpa bias (tergantung implementasi).

**Status: ⚠️ BEST PRACTICE?**
- **Proposal BISA DIBENARKAN tapi BUKAN keharusan.** Beberapa paper (DGN, CoLight) langsung memasukkan raw obs ke GNN tanpa pre-MLP. Yang lain (SAGCN-SST) memakai encoder.
- **Rekomendasi:** Bisa ditambahkan sebagai ablation: with/without pre-MLP. Jika ditambah, pastikan dimensi output = 64 agar sesuai spec. Atau cukup jelaskan di laporan bahwa linear projection internal GATConv sudah berfungsi serupa.

---

### Gap 4: Prediction Head Target — SUDAH DISELARASKAN DI NARASI

**Proposal (Bab IV §4.5, versi terbaru):**
> \hat{o}_{t+1} = f_{pred}(h'_i, a_i; \phi)
> Target prediksi adalah seluruh vektor observasi berikutnya (full-state prediction).

**Implementasi saat ini (`q_network.py` PredictionHead):**
- Output = `obs_dim` (≈43 dimensi) → prediksi **seluruh vektor state berikutnya**
- Loss = MSE(predicted_full_state, actual_full_state)

**Update status (18 April 2026):** Deskripsi prediction head dan fungsi loss di proposal sudah konsisten dengan implementasi full-state prediction.

**Dampak (setelah update):** Tidak lagi menjadi mismatch proposal vs implementasi; kini menjadi keputusan desain eksperimen.

**Status: ⚠️ BEST PRACTICE?**
- **Kedua pendekatan valid:**
  - Full state prediction (implementasi saat ini): Lebih kaya learning signal, dipakai di beberapa world-model papers. Tapi bisa noisy dan sulit converge.
  - Simplified target (varian ablation): Lebih stabil, fokus pada metrik yang relevan (queue & density). Dipakai di UNREAL (Jaderberg et al., 2016) yang menyarankan auxiliary target yang meaningful.
- **Rekomendasi terbaru:** Pertahankan full-state prediction sebagai default implementasi. Opsi simplified target (2 komponen) dapat dijadikan ablation study jika waktu memungkinkan.

---

### Gap 5: State Representation Formula — SUDAH DISELARASKAN DI NARASI

**Proposal (Bab IV §4.2, versi terbaru):**
> o_i = [q_i, Δq_i, d_i, w_i, Δd_i, p_i, τ_i] — 7 komponen

**ARSITEKTUR_GNN_MARL.md:**
> Menambahkan `waiting_time_l` — 6 komponen
> (queue, delta_queue, density, waiting_time, phase_onehot, duration)

**Implementasi saat ini (`sumo_env.py`):**
> 7 komponen: [queue, delta_queue, density, waiting, delta_density, phase_onehot, duration]
> Menambahkan `waiting_time` dan `delta_density` sebagai fitur temporal tambahan

**Update status (18 April 2026):** Formula state di proposal sudah mengikuti implementasi saat ini.

**Dampak (setelah update):** Tidak lagi menjadi mismatch proposal vs implementasi.

**Status: ⚠️ BEST PRACTICE?**
- **Menambahkan waiting_time ke state: BENAR & BEST PRACTICE.** Hampir semua paper ATSC menyertakan waiting time (MA2C, CoLight, PressLight). Ini fitur yang sangat informatif.
- **Menambahkan delta_density: BISA DIBENARKAN.** Memberikan sinyal temporal tambahan, meskipun redundan dengan delta_queue.
- **Rekomendasi terbaru:** Pertahankan 7 komponen sebagai baseline utama, lalu jika diperlukan lakukan ablation untuk menilai kontribusi masing-masing fitur temporal.

---

## B. GAP MEDIUM (Eksperimen & Evaluasi — belum perlu sekarang, wajib untuk TA)

### Gap 6: Baseline Tidak Lengkap

**Proposal (Bab V Metode Perbandingan):**
1. Fixed-timer ← **BELUM ADA**
2. Actuated control ← **BELUM ADA**
3. Independent DQN (tanpa GNN) ← ✅ ada
4. GAT-Double DQN tanpa fitur temporal ← **BELUM ADA**

**Status: BEST PRACTICE?**
- **Proposal BENAR.** Paper ATSC selalu membandingkan dengan fixed-timer (paling minimum) dan actuated (represent konvensional).
- **Rekomendasi:** Fixed-timer mudah — cukup jalankan SUMO tanpa RL, biarkan program lampu default berjalan. Actuated bisa dikonfigurasi di SUMO `.net.xml` dengan `type="actuated"`. GAT-DDQN tanpa temporal = ablation, modifikasi kecil (buang delta features dari input).

---

### Gap 7: Skenario Traffic Tidak Lengkap

**Proposal (Bab V Skenario Pengujian):**
1. Lalu lintas stabil ← ✅ ada (uniform random trips)
2. Peak hour ← **BELUM ADA**
3. Ketidakseimbangan arah ← **BELUM ADA**
4. Variasi pola harian ← **BELUM ADA**

**Status: BEST PRACTICE?**
- **Proposal BENAR.** Multiple traffic scenarios adalah standar di semua paper serius (CoLight, MPLight, PressLight).
- **Rekomendasi:** Buat route file berbeda via `randomTrips.py` dengan parameter berbeda. Contoh:
  - Peak hour: tinggi di menit 0-1800, puncak di 900-1500, turun di 1800-3600
  - Directional: fringe factor khusus (barat→timur berat, timur→barat ringan)
  - Daily: concatenate beberapa pola

---

### Gap 8: Analisis Statistik Belum Ada

**Proposal (Bab V Analisis Statistik):**
> Setiap eksperimen diulang minimal 5 kali dengan random seed berbeda.
> Mean ± standar deviasi. t-test atau Wilcoxon, α=0.05.

**Implementasi saat ini:** Hanya seed=42. Tidak ada multi-seed runner.

**Status: BEST PRACTICE?**
- **Proposal BENAR.** Multi-seed runs adalah standar minimum untuk riset RL. 5 seeds cukup.
- **Rekomendasi:** Tambahkan loop di training script atau buat bash/batch script yang jalankan 5 seeds. Mudah diimplementasikan.

---

### Gap 9: Metrik Evaluasi Tidak Lengkap

**Proposal (Bab V Metrik Evaluasi):**
- Metrik utama: avg_delay ✅, avg_queue ✅
- Metrik pendukung: throughput ✅, avg_travel_time ❌, stop_count ❌
- Metrik sistem: inference_latency ❌

**Status: BEST PRACTICE?**
- **Proposal BENAR.** Travel time dan stop count adalah metrik standar ATSC. Inference latency penting untuk feasibility argument.
- **Rekomendasi:** Tambahkan di `_get_metrics()`. Semuanya available dari TraCI:
  - `traci.vehicle.getAccumulatedWaitingTime(v)` → travel time proxy
  - `traci.simulation.getArrivedNumber()` → throughput tracking
  - `time.time()` sebelum/sesudah action selection → inference latency

---

## C. GAP MINOR (Teknis/Tooling)

### Gap 10: Yellow Transition Disederhanakan

**Proposal:** Implikasinya yellow harus berlangsung `yellow_time` detik sebelum green.

**Implementasi:** Yellow di-set lalu langsung di-overwrite green di step yang sama.

**Status:** Simplified tapi acceptable untuk penelitian. Banyak paper (SUMO-RL, IntelliLight) juga menyederhanakan yellow. Bisa diperbaiki nanti.

---

### Gap 11: Tidak Pakai wandb/TensorBoard

**Proposal (Bab V):** Menyebutkan wandb atau TensorBoard untuk logging.

**Implementasi:** CSV logger custom.

**Status:** CSV logger berfungsi. TensorBoard/wandb lebih baik untuk real-time monitoring. Bisa ditambahkan tapi bukan prioritas.

---

### Gap 12: Tidak Ada Unit Tests

**Proposal (Bab V Verifikasi):** Unit testing untuk GAT, Q-Network, environment wrapper.

**Status:** Belum ada. Bukan blocker tapi disarankan agar catch bugs lebih awal.

---

### Gap 13: Tidak Ada Script Visualisasi Attention Weights

**Proposal:** Attention visualization untuk interpretability analysis.

**Implementasi:** `get_attention_weights()` method ada, tapi tidak ada script yang memanggil dan memvisualisasikan.

**Status:** Method sudah siap, tinggal buat plotting script.

---

## D. Ringkasan: Mana yang BENAR, PERLU DIBUKTIKAN, atau SALAH?

| # | Desain di Proposal | Verdict | Penjelasan |
|---|---|---|---|
| 1 | **Action space Acyclic (default), Cyclic (ablation)** | ✅ **Aligned** | Narasi proposal sudah diselaraskan dengan implementasi: Acyclic sebagai desain utama, Cyclic untuk pembanding. |
| 2 | **Normalisasi state** | ✅ **Best practice** | Harus diimplementasikan. Input scaling adalah fundamental deep learning. |
| 3 | **Pre-projection MLP** | 🔄 **Needs testing** | Valid tapi bukan keharusan. GATConv internal projection sudah serupa. Bisa jadi ablation experiment. |
| 4 | **Prediction Head full-state (default)** | 🔄 **Needs testing** | Narasi proposal dan implementasi sudah konsisten pada full-state prediction; simplified target tetap layak diuji sebagai ablation. |
| 5 | **State = 7 komponen** | ✅ **Aligned** | Formula proposal sudah mengikuti implementasi (queue, delta_queue, density, waiting, delta_density, phase, duration). |
| 6 | **4 baselines** | ✅ **Best practice** | Fixed-timer + actuated + Independent DQN + ablation semua standar. |
| 7 | **4 skenario traffic** | ✅ **Best practice** | Multiple scenarios standar di paper ATSC berkualitas. |
| 8 | **5-seed + t-test** | ✅ **Best practice** | Standar minimum riset RL. |
| 9 | **Extended metrik** | ✅ **Best practice** | Travel time, stop count, inference latency semua standar. |
| 10 | **Yellow transition proper** | ✅ **Ideal tapi optional** | Banyak paper juga simplify. |

### Kesimpulan

- **Sinkronisasi narasi Bab IV dengan implementasi inti sudah dilakukan** untuk action space, formula state, dan prediction head.
- **Gap prioritas tersisa** berfokus pada kualitas eksperimen dan robustness implementasi: normalisasi penuh, baseline lengkap, multi-skenario, multi-seed statistik, dan metrik evaluasi lanjutan.
- **Arah kerja saat ini:** implementasi sudah berada di jalur yang benar untuk tahap awal; langkah berikutnya adalah penguatan metodologi evaluasi agar siap untuk hasil TA final.
