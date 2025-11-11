# Analisis Komparatif 10 Algoritma Unsupervised Clustering pada Data Traceroute Jaringan

Proyek ini menganalisis dataset `cleaned_traceroute_with_host.csv` menggunakan 10 algoritma clustering machine learning untuk mengidentifikasi pola dan anomali dalam data lalu lintas jaringan.

Seluruh analisis, model, dan visualisasi dihasilkan dengan menjalankan satu skrip Python: `cluster_analysis.py`.

## Performa Model

Tabel ini membandingkan Silhouette Score dari setiap model. Skor yang lebih tinggi menunjukkan cluster yang lebih baik dan lebih terpisah.

| Peringkat | Model | Parameter | Dataset | Silhouette Score | Catatan |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Spectral Clustering** | `K=4` | Sampel (10k) | **0.5695** | **Performa Terbaik.** Menghasilkan cluster yang paling padat & terpisah pada sampel. |
| 2 | **Mean Shift** | `Bandwidth=4.0` | Sampel (10k) | 0.5283 | Performa sangat baik pada sampel. |
| 3 | **Affinity Propagation** | `Damping=0.9` | Sampel (10k) | 0.5192 | Performa baik, namun dijalankan pada sampel. |
| 4 | **K-Means** | `K=4` | Penuh | 0.5144 | **"Standar Emas".** Performa sangat baik pada dataset penuh. |
| 5 | **BIRCH** | `K=4` | Penuh | 0.5144 | Hasil identik dengan K-Means; sangat cepat dan efisien. |
| 6 | **Mini-Batch K-Means** | `K=4` | Penuh | 0.5143 | Hasil hampir identik; efisien untuk data besar. |
| 7 | **Gaussian Mixture (GMM)** | `K=5` | Penuh | 0.4430 | Performa sedang. Kurang cocok untuk data ini dibanding K-Means. |
| 8 | **Agglomerative** | `K=2` | Penuh | 0.3845 | **Performa Buruk.** Skor rendah mengonfirmasi `K=2` adalah jumlah cluster yang buruk. |
| 9 | **DBSCAN** | `eps=3.0` | Penuh | 0.3807 | **Skor Buruk.** Model mengidentifikasi **66% (14,751) data sebagai "Noise"**. Ini adalah *temuan* bahwa data tidak padat. |
| 10 | **OPTICS** | `eps=3.0` | Penuh | 0.3807 | **Skor Buruk.** Hasil identik dengan DBSCAN, mengonfirmasi temuan "Noise". |

## 🚀 Cara Menjalankan Proyek

Proyek ini dirancang untuk dijalankan dari terminal.

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```

2.  **Siapkan Dataset**
    * Letakkan file data `cleaned_traceroute_with_host.csv` Anda di dalam folder utama proyek. (File ini di-ignore oleh `.gitignore` dan tidak akan di-upload).

3.  **Install Dependensi**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Analisis**
    ```bash
    python cluster_analysis.py
    ```

5.  **Periksa Hasil**
    * Semua file `.png` (plots) and `.csv` (data cluster) akan disimpan di dalam folder `output/` yang baru dibuat.
    * Laporan rekap performa akan dicetak langsung di terminal Anda.
