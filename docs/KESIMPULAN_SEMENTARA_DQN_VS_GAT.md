# Kesimpulan Sementara: Independent DQN vs GAT+DQN

*(Dokumen ini berisi analisis teoretis dan empiris dari sesi pengujian 16 Mei 2026, membandingkan performa baseline Independent DQN dengan arsitektur usulan GAT+DQN pada skenario PKJI).*

 ## 1. Masalah Utama: Local vs. Global Feedback Loop (+ Exploding Gradients & Action Coordination)
*   **Independent DQN:** Setiap agen (persimpangan) beroperasi secara independen. Jika satu persimpangan mengalami *gridlock* (macet total), kesalahan tersebut terlokalisasi. Agen lain tetap belajar secara normal berdasarkan kondisi di sekitar mereka. Sifat ini membuat Independent DQN sangat tahan banting (robust).
*   **GAT+DQN:** Menggunakan mekanisme *Graph Attention*, informasi *state* dan kesalahan (*loss*) didistribusikan antar agen. Jika satu persimpangan mengalami kegagalan ekstrem (misal: antrean tidak bergerak, penalti sangat besar), sinyal negatif ini menyebar melalui jaringan. Akibatnya, kegagalan lokal dapat memicu kolaps sistemik (semua agen ikut kacau) yang berujung pada meledaknya gradien (*Loss Explosion*).

## 2. Kapan Independent DQN "Terlihat Lebih Baik"?
*   **Stabilitas Pelatihan:** Independent DQN jauh lebih stabil di awal pelatihan dan lebih kebal terhadap kondisi ekstrem.
*   **Kecepatan Konvergensi Awal:** Karena fokusnya sempit (*myopic*), ia lebih cepat menemukan kebijakan lokal yang "cukup bagus" untuk sekadar mengosongkan persimpangan.
*   **Kondisi Over-saturated:** Pada saat macet total, mencoba berkoordinasi sering kali percuma. Tindakan reaktif murni (Greedy-local) dari Independent DQN sering kali menjadi satu-satunya cara bertahan hidup.

## 3. Kapan GAT+DQN "Seharusnya Lebih Baik"?
*   **Optimalisasi Arus (Traffic Progression):** GAT memiliki pandangan spasial (*upstream/downstream*). Ia bisa menunda lampu hijau untuk menyinkronkan kedatangan rombongan kendaraan besar dari persimpangan sebelah (menciptakan gelombang hijau).
*   **Mencegah Spillback:** GAT dapat mendeteksi jika jalan di depannya sudah penuh, sehingga ia menahan lampu merah untuk tidak menambah parah kemacetan di persimpangan berikutnya.

## 4. Kesimpulan untuk Narasi Skripsi
Hasil eksperimen menunjukkan adanya fenomena **Trade-off antara Potensi Koordinasi dan Stabilitas Pelatihan**.

Independent DQN menunjukkan stabilitas tinggi dan bertindak sebagai *safe baseline*. Namun, kinerjanya terbatas pada optimasi persimpangan tunggal.

Di sisi lain, GAT+DQN memiliki arsitektur unggul untuk menyebarkan informasi spasial antar agen demi membentuk koordinasi. Sayangnya, mekanisme komunikasi inilah yang menyebabkannya sangat rentan terhadap *Loss Explosion* pada tahap eksplorasi awal. Ketika satu agen memicu *gridlock* lokal (misalnya karena `min_green` terlalu singkat), sinyal penalti ekstrem menyebar ke seluruh jaringan dan merusak proses pembelajaran agen lain.

**Catatan Tindakan:** GAT+DQN membutuhkan penyesuaian hyperparameter yang lebih hati-hati (seperti menaikkan batas `min_green` menjadi 11 atau 12 detik) untuk mencegahnya masuk ke kondisi *gridlock* ireversibel selama fase eksplorasi (epsilon tinggi).

## 5. Klarifikasi Interpretasi (Penting)
*   Efek "kegagalan lokal menjadi sistemik" pada GAT **mungkin terjadi**, tetapi besarnya tergantung desain graf, reward shaping, protokol evaluasi, dan stabilitas training.
*   Risiko *NaN/Inf* pada GAT bukan semata karena mekanisme attention; biasanya muncul dari kombinasi skala fitur/reward yang ekstrem, target Q yang tidak stabil, learning rate, dan minimnya stabilisasi (normalisasi, clipping, dsb.).
*   Interdependensi aksi pada GAT memang membuat fase eksplorasi awal lebih rapuh, tetapi jika distabilkan dengan baik, koordinasi antarpersimpangan justru menjadi keunggulan utama GAT dibanding pendekatan lokal murni.

Dengan kata lain, narasi "Independent DQN lebih tahan banting" valid untuk konteks hasil sementara saat ini, tetapi tetap perlu diposisikan sebagai temuan kondisi-spesifik, bukan generalisasi universal.
