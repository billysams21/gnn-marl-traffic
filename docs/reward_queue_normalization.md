# Reward Queue Normalization — Penjelasan Perhitungan

Dokumen ini menjelaskan dasar perhitungan parameter `eff_vehicle_length` yang digunakan
untuk normalisasi per-lane pada fungsi reward di `src/envs/sumo_env.py`.

---

## 1. Masalah: `max_queue_per_lane` flat tidak cocok untuk jaringan arterial

Pada skenario `grid_3x3`, semua lane panjangnya homogen (~100–150 m), sehingga satu nilai
`max_queue_per_lane = 30` cukup representatif.

Pada skenario `arterial_stable` / `arterial_peak`, panjang lane sangat bervariasi:

| Tipe ruas | Panjang | Contoh edge |
|---|---|---|
| Arteri panjang | 435.6 m | A1A2, A2A1, B1B2, B2B1, C1C2, C2C1 |
| Arteri sedang  | 335.6 m | A2A3, A3A2, B2B3, B3B2, C2C3, C3C2 |
| Entry/exit     | 342.8 m | W_A0A1, E_A4A3, W_B1B1, S1B1, N1C1, … |
| Connector      | 132.4 m | A1B1, B1A1, A1C1, C1A1, … |

Dengan `max_queue_per_lane = 30` (kapasitas connector), lane arteri panjang (kapasitas fisik
~122 kendaraan) akan menghasilkan `queue_term > 1` bahkan di kondisi normal → reward
artifisial sangat negatif, tidak mencerminkan kondisi lalu lintas sebenarnya.

---

## 2. Parameter kendaraan (vType SUMO)

Nilai `minGap` disesuaikan dengan kondisi Indonesia (lebih rapat dari default SUMO Eropa):

| Tipe | `length` | `minGap` | Panjang efektif per slot antrian |
|---|---|---|---|
| Mobil (passenger) | 4.5 m | 1.0 m | **5.5 m** |
| Motor (motorcycle) | 2.0 m | 0.5 m | **2.5 m** |

> **`minGap`** di SUMO = jarak bumper-to-bumper minimum saat berhenti.
> Panjang efektif = `length + minGap` = ruang yang dibutuhkan satu kendaraan dalam antrian.

Perbandingan dengan nilai default SUMO (Eropa):
- Mobil: `minGap = 2.5 m` → terlalu besar untuk Indonesia
- Motor: `minGap = 1.0 m` → terlalu besar untuk Indonesia

---

## 3. Komposisi lalu lintas (PKJI)

Berdasarkan data empiris Indonesia (PKJI 2023):

| Tipe | Proporsi | EMP |
|---|---|---|
| Mobil | 35% | 1.0 |
| Motor | 65% | 0.2 |

---

## 4. Panjang kendaraan efektif rata-rata (weighted)

Antrian kendaraan adalah **packing 1-dimensi** sepanjang lane (depan ke belakang).
Panjang efektif campuran dihitung sebagai rata-rata tertimbang:

```
eff_vehicle_length = p_car × eff_car + p_moto × eff_moto
                   = 0.35 × 5.5 + 0.65 × 2.5
                   = 1.925 + 1.625
                   = 3.55 m
```

Nilai ini digunakan sebagai parameter `eff_vehicle_length = 3.55` di `default_config.py`
untuk skenario arterial.

---

## 5. Kapasitas antrian per lane (max_queue per lane)

Kapasitas antrian per SUMO lane strip = panjang lane / panjang efektif campuran.

> **Catatan penting:** `max_queue_per_lane` adalah kapasitas **longitudinal** (depan ke belakang)
> per **satu SUMO lane strip**. Jalan 2-lane = 2 SUMO lane strip terpisah, masing-masing
> dihitung sendiri. Jumlah lane sudah di-handle oleh `lane_count_i` dalam formula reward,
> sehingga `max_queue_per_lane` **tidak bergantung pada jumlah lane** — hanya pada panjang.
>
> Sublane model (`--lateral-resolution 0.7`) mempengaruhi throughput lateral (motor selap-selip)
> tetapi `getLastStepHaltingNumber()` tetap menghitung per lane strip, bukan per sublane.

### Perhitungan per tipe ruas

**Arteri panjang (435.6 m)** — contoh: A1A2, B1B2, C1C2
```
max_queue = floor(435.60 / 3.55) = floor(122.70) = 122 kendaraan/lane
```

**Arteri sedang (335.6 m)** — contoh: A2A3, B2B3, C2C3
```
max_queue = floor(335.60 / 3.55) = floor(94.54) = 94 kendaraan/lane
```

**Entry/exit (342.8 m)** — contoh: W_A0A1, W_B1B1, S1B1, N1C1
```
max_queue = floor(342.80 / 3.55) = floor(96.56) = 96 kendaraan/lane
```

**Connector vertikal (132.4 m)** — contoh: A1B1, A1C1, B1A1
```
max_queue = floor(132.40 / 3.55) = floor(37.30) = 37 kendaraan/lane
```

### Tabel ringkasan

| Tipe ruas | Panjang | maxQ mobil (5.5m) | maxQ motor (2.5m) | **maxQ mixed (3.55m)** |
|---|---|---|---|---|
| Arteri panjang | 435.6 m | 79 | 174 | **122** |
| Arteri sedang  | 335.6 m | 61 | 134 | **94**  |
| Entry/exit     | 342.8 m | 62 | 137 | **96**  |
| Connector      | 132.4 m | 24 | 52  | **37**  |

---

## 6. Apakah kolektor (B1B2, C1C2) ikut nilai arteri?

Ya — **bukan karena jenis jalan, tapi karena panjangnya sama**.

Ruas kolektor horizontal (B1B2, B2B1, C1C2, C2C1) memiliki panjang 435.6 m,
sama dengan arteri panjang (A1A2, A2A1). Maka `max_queue`-nya juga sama: **122**.

Perbedaan kolektor vs arteri hanya pada **jumlah lane per ruas**:
- Arteri: 2 lane per arah → 2 SUMO lane strips per edge → `lane_count_i` lebih besar
- Kolektor: 1 lane per arah → 1 SUMO lane strip per edge

`lane_count_i` sudah di-handle terpisah dalam formula reward (pembagi mean), bukan
dalam `max_queue_per_lane`. Jadi tidak ada yang "salah hitung".

Incoming lanes per TL kolektor (contoh B1):

| Edge | Panjang | Jenis | maxQ/lane |
|---|---|---|---|
| A1B1 (connector) | 132.4 m | penghubung arteri↔kolektor | 37 |
| B2B1 (kolektor panjang) | 435.6 m | kolektor horizontal | 122 |
| S1B1 (entry/exit) | 342.8 m | boundary entry | 96 |
| W_B1B1 (entry/exit) | 342.8 m | boundary entry | 96 |

---

## 7. Formula reward dengan per-lane normalization

```
r_i = -(queue_term_i + α × waiting_term_i)

queue_term_i = (1 / N_lanes_i) × Σ_j [ queue_j / max_queue_j ]

waiting_term_i = total_waiting_i / (N_lanes_i × max_waiting_time)
```

Di mana:
- `j` iterasi per lane dalam `controlled_lanes[i]`
- `queue_j` = `traci.lane.getLastStepHaltingNumber(lane_j)`
- `max_queue_j` = `traci.lane.getLength(lane_j) / eff_vehicle_length` (dihitung saat `reset()`)
- `max_waiting_time = 300 s` (tetap, tidak berubah)
- `α = 0.5` (`reward_alpha`)

### Perbandingan formula lama vs baru

**Lama (flat):**
```
queue_term_i = total_queue_i / (N_lanes_i × 30)
```
Lane arteri 435.6 m dengan 30 kendaraan → `queue_term = 30 / (1 × 30) = 1.0` ← saturasi palsu

**Baru (per-lane):**
```
queue_term_i = (1/1) × [30 / 122] = 0.246
```
Angka yang jujur: 30 kendaraan di lane 435.6 m = 24.6% kapasitas, bukan 100%.

---

## 8. Dampak pada skala reward

Dengan `avg_queue ≈ 2.43 kendaraan/lane` (rata-rata semua lane, dari log episode 1):

**Lama:**
```
queue_term per lane = 2.43 / 30 = 0.081
r/step/agent ≈ -0.081 → episode reward ≈ -58  (tapi arteri sering > 30 → spikes besar)
```

**Baru (per-lane, lane arteri dominan ~96–122):**
```
queue_term per lane = 2.43 / 96–122 = 0.020–0.025
r/step/agent ≈ -0.022 → episode reward ≈ -16
```

Reward arterial tidak bisa disamakan dengan grid_3x3 (densitas kendaraan memang 7× lebih tinggi),
tapi dengan per-lane normalization, `queue_term` berada di rentang [0,1] yang bermakna secara fisik.

---

## 9. Backward compatibility

Parameter `eff_vehicle_length` default = `0.0`.

- Kalau `0.0` (grid_3x3 dan skenario lain): `_lane_max_queue` tidak dibangun,
  reward fallback ke formula lama (`max_queue_per_lane = 30`). **Tidak ada breaking change.**
- Kalau `> 0` (arterial scenarios): per-lane normalization aktif.

---

## 10. File yang diubah

| File | Perubahan |
|---|---|
| `src/envs/sumo_env.py` | Param `eff_vehicle_length`; build `_lane_max_queue` di `reset()`; update `_compute_reward()` |
| `configs/default_config.py` | `arterial_stable` + `arterial_peak`: tambah `"eff_vehicle_length": 3.55` |
| `experiments/train.py` | Pass `eff_vehicle_length` dari scenario dict ke `SumoEnvironment` |
| `experiments/evaluate.py` | Sama dengan train.py |
| `experiments/generate_arterial_routes.py` | `minGap` mobil `2.5→1.0 m`, motor `1.0→0.5 m` |
