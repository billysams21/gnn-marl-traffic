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
- [x] Jalur integrasi TorontoSUMONetworks disiapkan sebagai validasi eksternal.
  - `SumoEnvironment` sekarang bisa menjalankan skenario berbasis `.sumocfg`.
  - Placeholder scenario `toronto_small` ditambahkan di config.
  - Dokumentasi integrasi tersedia di `docs/TORONTO_INTEGRATION.md`.
- [x] Protocol evaluasi diperbaiki setelah audit GAT-DQN vs Independent DQN.
  - `train.py` sekarang mendukung greedy validation via `eval_interval` dan `eval_episodes`.
  - `best_model.pt` dipilih berdasarkan `eval_reward` jika evaluasi tersedia.
  - `evaluate.py` membaca `config.json` run agar evaluasi tidak mismatch konfigurasi training.
- [x] Timing environment diperbaiki.
  - `phase_duration` sekarang dihitung per simulated second, bukan ditambah `delta_time` sekaligus.
  - Catatan: hasil sebelum dan sesudah fix timing tidak boleh dicampur sebagai protokol yang sama.
- [x] Fixed-time baseline tersedia.
  - `experiments/evaluate_fixed_time.py` menghitung reward/delay/queue/throughput tanpa training.
  - SUMO default fixed-time sudah terukur sebagai uncalibrated static baseline.
- [x] Pipeline PKJI calibrated baseline tersedia.
  - `src/baselines/pkji.py` untuk EMP/SMP, intergreen, cycle time, green split, dan offset.
  - `experiments/calculate_pkji_fixed_time.py` menghitung plan PKJI dari JSON.
  - `experiments/apply_pkji_plan_to_sumo.py` menerapkan plan ke `.net.xml`.
  - `experiments/generate_pkji_grid.py` membuat input PKJI sintetis.
  - `experiments/generate_pkji_route_variants.py` membuat route PKJI-aware dengan passenger/motorcycle/heavy.
- [x] Skenario PKJI-aware awal tersedia.
  - `grid_3x3_pkji_m1`: traffic stabil sintetis dengan komposisi kendaraan PKJI-aware.
  - `grid_3x3_pkji_m1p5`: kandidat peak/hour heavy demand sintetis.
  - Catatan: masih sintetis, bukan data observasi aktual Indonesia.

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
  - Catatan: timing RL tidak dikunci mengikuti PKJI; PKJI dipakai untuk fixed-time engineering baseline.
- [ ] Siapkan script khusus run utama, jangan pakai script sweep tuning.
  - Target awal baru: `grid_3x3_pkji_m1`, `gat_dqn` dan `independent_dqn`, 200 episode, 5 seed.
  - Peak-hour setelah stabil: `grid_3x3_pkji_m1p5`.
  - Seed rekomendasi: 42, 43, 44, 45, 46.
- [ ] Siapkan script agregasi hasil awal.
  - Minimal: mean/std `eval_reward` pada checkpoint evaluasi terakhir dan best eval per seed.
  - Kolom utama: reward, eval_reward, avg_delay, eval_avg_delay, avg_queue, eval_avg_queue, throughput, emergency_stops, loss_total.

## P1 - Wajib untuk Validitas Hasil TA

- [ ] Run utama GAT-DQN di `grid_3x3_pkji_m1` sebagai traffic stabil.
  - [ ] 5 seed selesai.
  - [ ] Cek tidak ada NaN/inf pada loss.
  - [ ] Cek reward, queue, delay, throughput, emergency stops masuk akal.
  - [ ] Agregasi mean/std `eval_reward` dan best eval.
- [ ] Baseline lengkap:
  - [x] SUMO default fixed-time baseline
  - [x] PKJI calibrated fixed-time baseline sintetis
  - [ ] independent DQN pada skenario PKJI-aware yang sama
  - [ ] actuated baseline (opsional jika waktu cukup)
  - [ ] GAT-DDQN ablation (tanpa fitur temporal atau varian yang disepakati)
- [ ] Skenario traffic lengkap:
  - [x] stabil sintetis: `grid_3x3_pkji_m1`
  - [ ] peak hour sintetis: `grid_3x3_pkji_m1p5` atau multiplier lain yang disepakati
  - [ ] directional imbalance atau variasi pola harian: pilih salah satu saja jika waktu cukup
  - [ ] variasi pola harian: ditunda/drop jika directional imbalance dipilih
- [ ] Multi-seed experiment runner (minimal 5 seed) + agregasi mean/std.
- [ ] Uji statistik (t-test atau Wilcoxon) untuk klaim perbandingan utama.
- [ ] Bangun dan uji real/replika network sebagai validasi eksternal.
  - [ ] Export file Toronto ke `data/networks/toronto_small/`.
  - [ ] Pastikan `toronto_small.net.xml` dan `toronto_small.sumocfg` tersedia.
  - [ ] Pastikan demand berjalan pada rentang 0-3600 detik.
  - [ ] Jalankan sanity test 5 episode dengan `--scenario toronto_small`.
  - Dilakukan setelah minimal ada hasil serius `grid_3x3` + satu baseline.
  - Tujuan: uji generalisasi, bukan tempat tuning utama.
  - Jika performa turun, catat sebagai insight generalisasi dan kompleksitas jaringan nyata.

## P2 - Nice to Have

- [ ] Buat skenario directional imbalance PKJI-aware jika dipilih:
  - [ ] Tambahkan parameter generator untuk memperbesar volume satu koridor/arah.
  - [ ] Pastikan fixed-time PKJI, independent DQN, dan GAT-DQN memakai route yang sama.
- [ ] Buat skenario variasi pola harian jika dipilih:
  - [ ] Demand berbeda per interval waktu dalam episode.
  - [ ] Lebih kompleks daripada directional imbalance, jadi kandidat untuk di-drop.
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

1. Pakai `grid_3x3_pkji_m1` sebagai skenario traffic stabil sintetis utama.
2. Jalankan sanity run 50 episode untuk `gat_dqn` dan `independent_dqn` pada `grid_3x3_pkji_m1`.
3. Agregasi `eval_reward` pada episode 10/20/30/40/50 dan cek apakah hasil masuk akal.
4. Jika stabil, buat script final 200 episode, 5 seed, untuk `gat_dqn` dan `independent_dqn`.
5. Bandingkan dengan fixed-time baseline:
   - SUMO default fixed-time
   - PKJI calibrated fixed-time `m1`
6. Tambahkan peak-hour dengan `grid_3x3_pkji_m1p5`.
7. Putuskan satu skenario tambahan saja:
   - directional imbalance, atau
   - variasi pola harian.
8. Setelah hasil grid stabil + baseline lengkap, mulai real/replika network untuk validasi eksternal.

## Catatan Eksekusi

- Simpan semua hasil run di folder logs yang terstruktur: model, config, metrics, dan metadata seed.
- Jangan campur eksperimen exploratory dengan eksperimen final pelaporan.
- Setiap perubahan konfigurasi besar harus dicatat di changelog eksperimen.
- Untuk laporan, bedakan jelas:
  - `SUMO default fixed-time`: static baseline tidak terkalibrasi.
  - `PKJI calibrated fixed-time`: engineering baseline sintetis/terkalibrasi.
  - `PKJI-aware RL scenario`: route demand dengan passenger/motorcycle/heavy composition.
- Jika data volume masih sintetis, jangan klaim sebagai data aktual Indonesia.
