# Analisis 10 Unsupervised Clustering Models pada Data Traceroute Jaringan

Proyek ini menganalisis dataset `cleaned_traceroute_with_host.csv` menggunakan 10 algoritma clustering machine learning untuk mengidentifikasi pola dan anomali dalam data lalu lintas jaringan.

Seluruh analisis, model, dan visualisasi dihasilkan dengan menjalankan satu skrip Python: `main.py`.

## Performa Model

Tabel ini membandingkan Silhouette Score dari setiap model. Skor yang lebih tinggi menunjukkan cluster yang lebih baik dan lebih terpisah.

| Peringkat | Model | Parameter | Dataset | Silhouette Score |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Spectral Clustering** | `K=4` | Sampel (10k) | 0.5695 |
| 2 | **Mean Shift** | `Bandwidth=4.0` | Sampel (10k) | 0.5283 |
| 3 | **Affinity Propagation** | `Damping=0.9` | Sampel (10k) | 0.5192 |
| 4 | **K-Means** | `K=4` | Penuh | 0.5144 |
| 5 | **BIRCH** | `K=4` | Penuh | 0.5144 |
| 6 | **Mini-Batch K-Means** | `K=4` | Penuh | 0.5143 |
| 7 | **Gaussian Mixture (GMM)** | `K=5` | Penuh | 0.4430 |
| 8 | **Agglomerative** | `K=2` | Penuh | 0.3845 |
| 9 | **DBSCAN** | `eps=3.0` | Penuh | 0.3807 |
| 10 | **OPTICS** | `eps=3.0` | Penuh | 0.3807 |

## Cara Menjalankan Proyek


1.  **Clone the repo**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```
    
2.  **Install Dependency**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the program**
    ```bash
    python main.py
    ```

4.  **Check output**
    * Semua file `.png` (plots) and `.csv` (data cluster) akan disimpan di dalam folder `output/` yang baru dibuat.
    * Laporan rekap performa akan dicetak langsung di terminal.
