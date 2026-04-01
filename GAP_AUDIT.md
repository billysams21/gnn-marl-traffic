# Audit Gap: Implementasi vs Proposal TA

Tanggal: 12 Maret 2026  
Referensi: Bab IV (Desain Konsep Solusi), Bab V (Rencana Selanjutnya), ARSITEKTUR_GNN_MARL.md

**Updated: 1 April 2026** — Fixes applied by Elle

---

## ✅ FIXES APPLIED (Apr 1, 2026)

### Gap 2: State Normalization — FIXED ✅

**Problem:** Raw observation values (queue ~0-30, waiting_time ~0-300) caused gradient instability.

**Fix:** Added normalization parameters to `SumoEnvironment.__init__`:
- `max_queue_per_lane`: 30 (normalize queue to [0, 1])
- `max_waiting_time`: 300.0 (normalize waiting to [0, 1])
- `max_delta_queue`: 10.0 (normalize delta_queue to [-1, 1])
- `max_delta_density`: 0.5 (normalize delta_density to [-1, 1])

All observations now output normalized values in stable ranges.

### Gap 4: Prediction Head Target — FIXED ✅

**Problem:** Prediction head predicted full observation (43 dims), which is noisy and hard to learn.

**Fix:** Added `prediction_mode` to `PredictionHead`:
- `'simplified'` (default, per proposal): Predict [avg_queue, avg_density] = 2 values
- `'full'` (original): Predict entire observation

Also added `compute_target()` method that extracts averaged metrics from observation.

**Code changes:**
- `src/models/q_network.py`: New `PredictionHead` with simplified mode
- `src/agents/dqn_agent.py`: Use simplified mode by default, `set_num_lanes()` method
- `configs/default_config.py`: Added `prediction.mode` config
- `experiments/train.py`: Pass `num_lanes` to agent

---

## A. GAP BESAR (Algoritmik / Desain)

### Gap 1: Action Space — Proposal Cyclic, Implementasi Acyclic

**Proposal (Bab IV 4.4):**
> Penelitian ini menggunakan pendekatan Cyclic: A = {Keep, Change}.
> Jika Change, fase berubah secara siklis (1 → 2 → 3 → ... → 1).
> Setiap Δt detik, agen memutuskan apakah tetap pada fase saat ini atau beralih ke fase berikutnya.

**Implementasi saat ini:**
- `num_actions = num_phases` (misal 2 untuk grid sederhana)
- Aksi = pilih fase target secara langsung (Acyclic/phase selection)
- Tidak ada logika "Keep" vs "Change"

**Dampak:** Menyalahi desain eksplisit di proposal. Proposal memberikan argumen safety kenapa Cyclic dipilih.

**Status: ⚠️ BEST PRACTICE?**
- **Proposal Cyclic: BUKAN best practice.** Mayoritas paper ATSC modern (MPLight, CoLight, PressLight, AttendLight) menggunakan **Acyclic (phase selection)** karena lebih fleksibel dan menghasilkan performa lebih baik.
- Argumen "safety" di proposal lemah — dalam simulasi SUMO, transisi fase selalu melalui yellow phase sehingga tidak ada bahaya. Safety hanya relevan di deployment real-world.
- **Rekomendasi:** Implementasi Acyclic yang ada **lebih baik** dari proposal. Namun perlu diakui/dijelaskan perbedaannya di laporan TA nanti. Atau, implementasi kedua opsi dan bandingkan sebagai ablation study.

---

### Gap 2: Normalisasi State Belum Ada

**Proposal (Bab IV 4.2):**
> Setiap elemen dalam vektor observasi dipetakan ke rentang [0,1] atau [-1,1].
> Untuk q_{i,l}: normalisasi dengan q_{i,l} / C_{i,l} (kapasitas maks lajur).
> Untuk Δq: normalisasi ke [-1,1] berdasarkan perubahan maksimum.

**Implementasi saat ini (`sumo_env.py` `_get_observation`):**
- `queue` = raw `getLastStepHaltingNumber()` → range [0, ∞)
- `delta_queue` = raw difference → range (-∞, ∞)
- `waiting` = raw `getWaitingTime()` → range [0, ∞) (bisa ratusan detik)
- `density` = di-clip ke [0,1] ✅
- `phase_duration` = dinormalisasi / max_green ✅

**Dampak:** Input features dengan skala sangat berbeda (queue ~0-30, waiting ~0-500) menyebabkan gradient instability. Neural network harus belajar scale sendiri lewat first layer weights.

**Status: ⚠️ BEST PRACTICE?**
- **Proposal BENAR.** Normalisasi input adalah best practice universal dalam deep learning. Tanpa normalisasi, fitur dengan magnitude besar (waiting_time) mendominasi gradient.
- **Rekomendasi:** Harus diperbaiki. Implementasikan normalisasi sesuai proposal.

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

### Gap 4: Prediction Head Target Berbeda

**Proposal (Bab IV 4.5):**
> [q̂_{t+1}, d̂_{t+1}] = f_pred(h'_i, a_i; φ)
> Target prediksi **disederhanakan ke komponen utama** (bukan seluruh vektor state).
> Output = 2 nilai: predicted avg queue + predicted avg density.

**Implementasi saat ini (`q_network.py` PredictionHead):**
- Output = `obs_dim` (≈43 dimensi) → prediksi **seluruh vektor state berikutnya**
- Loss = MSE(predicted_full_state, actual_full_state)

**Dampak:** Prediction head memprediksi 43 dimensi alih-alih 2. Lebih sulit dipelajari, loss magnitude lebih besar, dan bisa mendominasi/mengganggu RL loss.

**Status: ⚠️ BEST PRACTICE?**
- **Kedua pendekatan valid:**
  - Full state prediction (implementasi saat ini): Lebih kaya learning signal, dipakai di beberapa world-model papers. Tapi bisa noisy dan sulit converge.
  - Simplified target (proposal): Lebih stabil, fokus pada metrik yang relevan (queue & density). Dipakai di UNREAL (Jaderberg et al., 2016) yang menyarankan auxiliary target yang meaningful.
- **Rekomendasi:** Implementasi proposal mungkin lebih stabil. Bisa ditest keduanya. Yang penting: match dengan apa yang ditulis di laporan TA.

---

### Gap 5: State Representation Formula Berbeda

**Proposal (Bab IV 4.2):**
> o_i = [q_i, Δq_i, d_i, p_i, τ_i] — 5 komponen
> (queue, delta_queue, density, phase_onehot, duration)

**ARSITEKTUR_GNN_MARL.md:**
> Menambahkan `waiting_time_l` — 6 komponen
> (queue, delta_queue, density, waiting_time, phase_onehot, duration)

**Implementasi saat ini (`sumo_env.py`):**
> 7 komponen: [queue, delta_queue, density, waiting, delta_density, phase_onehot, duration]
> Menambahkan `waiting_time` DAN `delta_density` yang tidak ada di formula Bab IV

**Dampak:** obs_dim lebih besar dari spec. Prediction head dan Q-network input dimensi tidak sesuai spec. Tidak masalah secara algoritmik, tapi tidak sesuai tulisan.

**Status: ⚠️ BEST PRACTICE?**
- **Menambahkan waiting_time ke state: BENAR & BEST PRACTICE.** Hampir semua paper ATSC menyertakan waiting time (MA2C, CoLight, PressLight). Ini fitur yang sangat informatif.
- **Menambahkan delta_density: BISA DIBENARKAN.** Memberikan sinyal temporal tambahan, meskipun redundan dengan delta_queue.
- **Rekomendasi:** Implementasi saat ini (7 komponen) lebih baik dari Bab IV formula (5 komponen). ARSITEKTUR_GNN_MARL.md (6 komponen) adalah kompromi. Pilih satu dan dokumentasikan di laporan TA. Saran: gunakan yang ada di ARSITEKTUR_GNN_MARL.md (6 komponen: tambah waiting, buang delta_density) atau pertahankan 7 komponen tapi update formula di laporan.

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
| 1 | **Cyclic action space** | ❌ **Suboptimal** | Mayoritas paper ATSC menggunakan Acyclic (phase selection). Implementasi saat ini (Acyclic) justru lebih baik. Argumen safety hanya relevan di real-world, bukan simulasi. |
| 2 | **Normalisasi state** | ✅ **Best practice** | Harus diimplementasikan. Input scaling adalah fundamental deep learning. |
| 3 | **Pre-projection MLP** | 🔄 **Needs testing** | Valid tapi bukan keharusan. GATConv internal projection sudah serupa. Bisa jadi ablation experiment. |
| 4 | **Prediction Head target = 2 nilai** | 🔄 **Needs testing** | Proposal (simplified) mungkin lebih stabil. Implementasi (full state) lebih kaya info. Keduanya defensible, perlu eksperimen. |
| 5 | **State = 5 komponen** | ❌ **Kurang lengkap** | Implementasi (7 komponen) lebih kaya. ARSITEKTUR_GNN_MARL.md (6 komponen) lebih baik dari Bab IV formula. Waiting time harus ada. |
| 6 | **4 baselines** | ✅ **Best practice** | Fixed-timer + actuated + Independent DQN + ablation semua standar. |
| 7 | **4 skenario traffic** | ✅ **Best practice** | Multiple scenarios standar di paper ATSC berkualitas. |
| 8 | **5-seed + t-test** | ✅ **Best practice** | Standar minimum riset RL. |
| 9 | **Extended metrik** | ✅ **Best practice** | Travel time, stop count, inference latency semua standar. |
| 10 | **Yellow transition proper** | ✅ **Ideal tapi optional** | Banyak paper juga simplify. |

### Kesimpulan

- **Proposal mostly correct** — desain-desain di Bab IV/V mayoritas mengikuti best practice kecuali Cyclic action space.
- **Implementasi saat ini** sudah menangkap arsitektur inti (GAT + Double DQN + Prediction Head) dengan benar, tapi perlu penyesuaian di normalisasi, state formula, dan prediction head target.
- **Keputusan kunci:** Apakah ikut proposal literal (Cyclic, 5 komponen state, simplified pred head) atau ikut best practice dan update laporan? **Saran: ikut best practice, update laporan sesuai implementasi yang lebih baik.**
