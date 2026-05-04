# TODO Sebelum Training

ini checklist implementasi

## Status Saat Ini (done)

- [x] Narasi proposal Bab IV diselaraskan dengan implementasi untuk action space (Acyclic sebagai default, Cyclic sebagai ablation).
- [x] Narasi proposal Bab IV diselaraskan dengan implementasi untuk representasi state.
- [x] Narasi proposal Bab IV diselaraskan dengan implementasi untuk prediction head.
- [x] Normalisasi observasi utama sudah diterapkan di environment (queue, delta, density, waiting, phase duration).
- [x] Logging dasar CSV + config JSON sudah tersedia.
- [x] Smoke test tuning timing selesai di `grid_2x2` dan `grid_3x3`.
  - Keputusan operasional: `yellow_time=2`, `min_green=10`.
  - Catatan: hasil terbaik awal muncul di `g7`, tetapi dengan `delta_time=5`, `g7` efektif berada pada bucket minimum green 10 detik. Untuk laporan dan run utama, gunakan `min_green=10`.
- [x] Dry-run `grid_3x3` 30 episode menunjukkan training berjalan wajar untuk kandidat utama.
  - Tidak ada indikasi umum NaN/inf pada kandidat utama.
  - Reward/loss masih noisy, tetapi masuk akal untuk RL dengan epsilon masih tinggi.
  - `y2_g4` pernah menunjukkan outlier/instabilitas besar pada salah satu seed, jadi tidak dipilih sebagai default.

## P0 - Wajib Sebelum Run Besar

- [x] Kunci keputusan prediction mode final untuk eksperimen utama (simplified atau full), lalu samakan di config, training script, evaluate script, dan dokumen.
  - Keputusan: `full` sebagai default eksperimen utama.
  - `simplified` tetap tersedia untuk ablation study.
- [x] Tambahkan global seeding untuk reproducibility (python random, numpy, torch cpu/cuda, dan deterministic flags).
  - Implemented via `src/utils/seeding.py` and wired into `train.py` + `evaluate.py`.
- [x] Pastikan urutan lane deterministik (hindari urutan dari set yang bisa berubah antar run).
- [x] Tambahkan mode resume training dari checkpoint (argumen CLI + restore state penting).
- [x] Tegaskan definisi throughput untuk laporan utama (misalnya arrived/completed per episode, bukan hanya departed per step).
- [ ] Kunci timing final di config utama atau command final:
  - `delta_time=5`
  - `yellow_time=2`
  - `min_green=10`
  - `max_green=60`
- [ ] Siapkan script khusus run utama, jangan pakai script sweep tuning.
  - Target awal: `grid_3x3`, `gat_dqn`, 200 episode, 5 seed.
  - Seed rekomendasi: 42, 43, 44, 45, 46.
- [ ] Siapkan script agregasi hasil awal.
  - Minimal: mean/std dari last 10 atau last 20 episode per seed.
  - Kolom utama: reward, avg_delay, avg_queue, throughput, emergency_stops, loss_total.

## P1 - Wajib untuk Validitas Hasil TA

- [ ] Run utama GAT-DQN di `grid_3x3`.
  - [ ] 5 seed selesai.
  - [ ] Cek tidak ada NaN/inf pada loss.
  - [ ] Cek reward, queue, delay, throughput, emergency stops masuk akal.
  - [ ] Agregasi mean/std last 10 atau last 20 episode.
- [ ] Baseline lengkap:
  - [ ] fixed-timer
  - [ ] actuated
  - [ ] independent DQN
  - [ ] GAT-DDQN ablation (tanpa fitur temporal atau varian yang disepakati)
- [ ] Skenario traffic lengkap:
  - [ ] stabil
  - [ ] peak hour
  - [ ] directional imbalance
  - [ ] variasi pola harian
- [ ] Multi-seed experiment runner (minimal 5 seed) + agregasi mean/std.
- [ ] Uji statistik (t-test atau Wilcoxon) untuk klaim perbandingan utama.
- [ ] Bangun dan uji real network sebagai validasi eksternal.
  - Dilakukan setelah minimal ada hasil serius `grid_3x3` + satu baseline.
  - Tujuan: uji generalisasi, bukan tempat tuning utama.
  - Jika performa turun, catat sebagai insight generalisasi dan kompleksitas jaringan nyata.

## P2 - Nice to Have

- [ ] Tambah metrik evaluasi lanjutan:
  - [ ] average travel time
  - [ ] stop count
  - [ ] inference latency
- [ ] Tambah script visualisasi attention weights untuk interpretabilitas.
- [ ] Tambah sanity checks otomatis sebelum run panjang:
  - [ ] dry-run 5-20 episode
  - [ ] cek tidak ada NaN/inf pada loss
  - [ ] cek distribusi reward masuk akal
- [ ] Tambah integrasi TensorBoard/W&B (opsional, tapi sangat membantu monitoring).

## Definition of Done (DoD) Sebelum Training Besar

- [ ] Semua item P0 selesai
- [ ] Minimal dry-run `grid_3x3` lulus untuk timing final
- [ ] Prosedur re-run dari seed berbeda bisa dijalankan dengan 1 command/script.
- [ ] Template tabel hasil (mean +- std + p-value) siap dipakai untuk laporan.

## Urutan Eksekusi Terdekat

1. Kunci timing final (`yellow_time=2`, `min_green=10`) di config atau CLI command.
2. Buat script run utama untuk `gat_dqn grid_3x3` dengan 5 seed dan 200 episode.
3. Jalankan run utama GAT-DQN.
4. Agregasi hasil last 10/20 episode per seed.
5. Jalankan baseline `independent_dqn` dengan timing dan seed yang sama.
6. Tambahkan fixed-time baseline.
7. Setelah ada hasil `grid_3x3` + baseline, mulai real/replika network untuk validasi eksternal.

## Catatan Eksekusi

- Simpan semua hasil run di folder logs yang terstruktur: model, config, metrics, dan metadata seed.
- Jangan campur eksperimen exploratory dengan eksperimen final pelaporan.
- Setiap perubahan konfigurasi besar harus dicatat di changelog eksperimen.
