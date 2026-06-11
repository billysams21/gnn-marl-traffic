# DEPRECATED
# Reward Stabilization Plan: PKJI Level-2

## Status

Eksperimen Level-2a-Refine pada `gat_dqn` sudah menurunkan anomali hanya sedikit, dari sekitar 7.6% ke 6.6% pada episode 201-300. Penurunan ini belum cukup untuk menyatakan policy stabil. Episode katastrofik masih muncul dengan reward sangat negatif, termasuk skala ratusan ribu hingga jutaan.

Kesimpulan sementara: masalah belum selesai, dan model belum layak dinaikkan ke evaluasi final `time_to_teleport=-1`.

## Klarifikasi Teknis Yang Benar

### 1. Reward ekstrem berasal dari waiting time yang akumulatif

Reward saat ini dihitung sebagai:

```python
reward = -(total_queue + self.reward_alpha * total_waiting)
```

Fungsi reward ini linear terhadap queue dan waiting time. Namun, `waiting_time` dari SUMO bersifat akumulatif untuk kendaraan yang berhenti. Jika gridlock terjadi, waiting time terus meningkat, lalu reward per-step dijumlahkan sepanjang episode. Karena episode memiliki sekitar 720 decision step, cumulative episode return bisa menjadi sangat negatif.

Jadi masalahnya bukan fungsi reward kuadratik, melainkan akumulasi waiting time yang tidak dibatasi dan dijumlahkan sepanjang episode.

### 2. Observation sudah dinormalisasi, reward belum

Observation sudah dinormalisasi dan di-clip, termasuk waiting time:

```python
waiting_norm = waiting_raw / self.max_waiting_time
waiting_norm = np.clip(waiting_norm, 0.0, 1.0)
```

Artinya GAT tidak menerima fitur waiting time bernilai jutaan. Yang bernilai ekstrem adalah reward, return, TD target, dan loss training.

Ini penting untuk laporan: narasi "fitur jutaan menembus attention" tidak akurat untuk kode saat ini.

### 3. Attention GAT bukan safety controller

GAT attention membantu memilih bobot informasi antar persimpangan, tetapi tidak otomatis mencegah gridlock atau spillback. Attention mengolah representasi state, bukan mengubah kapasitas jalan, fase sinyal, atau antrean fisik.

Narasi yang lebih akurat:

> GAT tidak menyebarkan reward ekstrem sebagai fitur. Namun, GAT membuat representasi antar persimpangan saling terkopel. Ketika gridlock terjadi, reward ekstrem menghasilkan TD target yang tidak stabil. Kombinasi representasi terkopel dan target RL ekstrem dapat membuat GAT terlihat lebih rapuh daripada Independent DQN.

### 4. MSE loss terlalu sensitif terhadap outlier

DQN saat ini memakai MSE untuk RL loss. Pada episode katastrofik, target:

```python
y = reward + gamma * max_q_next
```

bisa memiliki magnitude sangat besar karena reward mentah. MSE akan memperbesar pengaruh outlier tersebut secara kuadrat. Walaupun gradient clipping sudah ada, update tetap bisa didominasi oleh episode collapse.

Huber loss lebih cocok untuk kondisi ini karena lebih robust terhadap TD error ekstrem.

## Diagnosis Sementara

Akar masalah yang paling mungkin saat ini adalah reward-scale instability:

- observation sudah stabil,
- policy training sudah memakai DDQN dan target network,
- gradient clipping sudah ada,
- curriculum teleport sudah membantu tetapi belum cukup,
- reward masih raw, tidak dinormalisasi, dan tidak dibatasi.

Dengan demikian, sebagian "GAT instability" kemungkinan bukan murni kelemahan arsitektur GAT, melainkan efek reward dynamics ekstrem yang lebih terlihat pada model dengan representasi graph terkopel.

## Yang Akan Dilakukan Sekarang

Tahap berikutnya tidak akan mengubah banyak faktor sekaligus. Fokusnya adalah ablation bersih pada stabilitas target DQN.

Status implementasi:

- normalized per-agent reward sudah diterapkan di `src/envs/sumo_env.py`,
- RL loss sudah diganti dari MSE ke Huber di `src/agents/dqn_agent.py`,
- opsi resume bersih sudah ditambahkan di `experiments/train.py`:
  - `--reset-replay-buffer`,
  - `--reset-optimizer-state`,
- runner Stage 1 tersedia di `run_pkji_scenario_100eps_level2a_stage1_stabilization.ps1`.

### Perubahan 1: Reward normalization per agent

Reward akan dinormalisasi berdasarkan jumlah lane dan konstanta normalisasi environment:

```python
queue_term = total_queue / (len(lanes) * self.max_queue_per_lane)
waiting_term = total_waiting / (len(lanes) * self.max_waiting_time)
reward = -(queue_term + self.reward_alpha * waiting_term)
```

Tujuannya:

- menyamakan skala reward antar persimpangan,
- mengurangi dominasi intersection dengan lane lebih banyak,
- membuat reward scale sejalan dengan observation scale,
- menekan TD target ekstrem.

Catatan: `waiting_term` masih bisa melebihi 1.0 jika waiting time total sangat besar. Karena itu soft clipping ringan dapat diuji setelah baseline normalized reward selesai.

### Perubahan 2: Ganti RL loss dari MSE ke Huber

RL loss akan diganti dari:

```python
nn.functional.mse_loss(q_taken, target)
```

menjadi:

```python
nn.functional.smooth_l1_loss(q_taken, target)
```

Prediction loss tidak diubah dulu agar efek eksperimen tetap terbaca.

### Perubahan 3: Pertahankan hyperparameter lain

Untuk menjaga eksperimen tetap bersih, tahap pertama tidak mengubah:

- epsilon schedule,
- min_green,
- max_green,
- aux_weight,
- time_to_teleport,
- model architecture.

Konfigurasi awal yang disarankan tetap:

```text
time_to_teleport = 600
epsilon_start = 0.05
epsilon_end = 0.05
epsilon_decay = 1.0
lr = 5e-5
aux_weight = 0.05
eval_episodes = 5
```

## Rencana Eksperimen

### Tahap 1: Reward normalization + Huber loss

Jalankan ulang fase refine dengan perubahan:

1. normalized per-agent reward,
2. Huber RL loss,
3. hyperparameter lain tetap.

Target evaluasi:

- catastrophic rate episode 201-300,
- worst reward,
- median reward,
- IQR reward,
- mean/std reward,
- greedy eval reward dengan `eval_episodes >= 5`.

Jika anomaly rate turun drastis, maka evidence kuat bahwa akar masalah utama adalah reward-scale instability.

### Tahap 2: Soft clipping atau reward squashing

Jika Tahap 1 belum cukup, tambahkan salah satu:

```python
reward = np.clip(reward, -20.0, 20.0)
```

atau:

```python
reward = np.tanh(reward / scale)
```

Pilihan ini dilakukan setelah normalization, bukan sebagai hard clipping brutal pada reward mentah.

### Tahap 3: Delta atau improvement reward

Jika normalized reward dan Huber loss belum menghilangkan collapse, evaluasi reward berbasis perubahan:

- delta waiting,
- delta queue,
- pressure-like reward,
- throughput bonus.

Tahap ini lebih berisiko karena bisa memunculkan reward hacking atau oscillation, sehingga tidak dijadikan perubahan pertama.

## Kriteria Berhasil

Perbaikan dianggap berhasil jika:

- anomaly rate turun ke bawah 1-2% pada `tt=600`,
- worst reward tidak lagi mencapai skala ratusan ribu atau jutaan,
- greedy evaluation stabil lintas seed,
- median/IQR membaik tanpa hanya mengandalkan satu episode evaluasi bagus.

Jika target ini tercapai, baru lanjut ke curriculum lebih keras:

1. `tt=600` stabilized,
2. `tt=300` atau lebih ketat jika diperlukan,
3. evaluasi `tt=-1` multi-seed sebagai final robustness test.

## Kesimpulan

Langkah paling rasional sekarang adalah menstabilkan reward dan target DQN terlebih dahulu. Tuning epsilon, min-green, atau attention sebaiknya tidak menjadi prioritas pertama karena data saat ini lebih konsisten dengan masalah reward scale daripada masalah eksplorasi murni.
