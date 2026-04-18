# TODO Sebelum Training

ini checklist implementasi

## Status Saat Ini (done)

- [x] Narasi proposal Bab IV diselaraskan dengan implementasi untuk action space (Acyclic sebagai default, Cyclic sebagai ablation).
- [x] Narasi proposal Bab IV diselaraskan dengan implementasi untuk representasi state.
- [x] Narasi proposal Bab IV diselaraskan dengan implementasi untuk prediction head.
- [x] Normalisasi observasi utama sudah diterapkan di environment (queue, delta, density, waiting, phase duration).
- [x] Logging dasar CSV + config JSON sudah tersedia.

## P0 - Wajib Sebelum Run Besar

- [x] Kunci keputusan prediction mode final untuk eksperimen utama (simplified atau full), lalu samakan di config, training script, evaluate script, dan dokumen.
  - Keputusan: `full` sebagai default eksperimen utama.
  - `simplified` tetap tersedia untuk ablation study.
- [x] Tambahkan global seeding untuk reproducibility (python random, numpy, torch cpu/cuda, dan deterministic flags jika perlu).
  - Implemented via `src/utils/seeding.py` and wired into `train.py` + `evaluate.py`.
- [ ] Pastikan urutan lane deterministik (hindari urutan dari set yang bisa berubah antar run).
- [ ] Tambahkan mode resume training dari checkpoint (argumen CLI + restore state penting).
- [ ] Tegaskan definisi throughput untuk laporan utama (misalnya arrived/completed per episode, bukan hanya departed per step).

## P1 - Wajib untuk Validitas Hasil TA

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
- [ ] Semua item P1 selesai
- [ ] Minimal 1 dry-run lulus untuk tiap skenario utama
- [ ] Prosedur re-run dari seed berbeda bisa dijalankan dengan 1 command/script.
- [ ] Template tabel hasil (mean +- std + p-value) siap dipakai untuk laporan.

## Catatan Eksekusi

- Simpan semua hasil run di folder logs yang terstruktur: model, config, metrics, dan metadata seed.
- Jangan campur eksperimen exploratory dengan eksperimen final pelaporan.
- Setiap perubahan konfigurasi besar harus dicatat di changelog eksperimen.
