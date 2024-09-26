
# Creating an Emotion Recognition Synthetic Dataset with Python & Stable Diffusion | Image generation

Proyek ini berfokus pada pembuatan kumpulan data sintetis untuk pengenalan emosi menggunakan Python dan Stable Diffusion. Kumpulan data ini dimaksudkan untuk digunakan dalam pelatihan model pembelajaran mesin yang dapat mengenali dan mengklasifikasikan emosi manusia dari ekspresi wajah secara akurat. Dengan memanfaatkan teknik pembuatan gambar canggih dengan Stable Diffusion, proyek ini bertujuan untuk membuat gambar emosi wajah yang beragam dan berkualitas tinggi.

---
## Deployment Tes emotion

To deploy this project run

```bash
  cd Dataset
```
```bash
  python test_model.py
```
![App Screenshot](/asset/hasil.jpg)

---

## File Penting


| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `Main.ipynb` | `image ai` | **Membuat**. Gambar Ai dengan Diffusers  |
| `prepare dan train.ipynb` | `Data Prep dan Traning` | **Memproses**. Gambar Ai dengan mengambil keypoint dan traning menggunakan randomfores  |
| `test_model.py` | `Testing Realtime Emotion` | **Mengetes**. Realtime Emotion  |

---

## 🤗 Diffusers - 
Creating an Emotion Recognition Synthetic Dataset with Python & Stable Diffusion | Image generation
🤗 Diffusers is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. Whether you're looking for a simple inference solution or training your own diffusion models, 🤗 Diffusers is a modular toolbox that supports both. Our library is designed with a focus on

---

## Hasil Generate

- Marah
![App Screenshot](/asset/marah.png)
- Netral
![App Screenshot](/asset/netral.png)
- Sedih
![App Screenshot](/asset/sedih.png)
- Senang
![App Screenshot](/asset/senang.png)
- Terkejut
![App Screenshot](/asset/terkejut.png)

---

## Applications
- Emotion recognition in images
- Training AI models for emotion-based interaction systems
- Synthetic data augmentation for emotion datasets

---

## Project Components
- Image Generation: Using Stable Diffusion to generate facial images with different emotions.
- Emotion Labeling: Automatically assigning emotion labels to generated images (happy, sad, angry, etc.).
- Data Export: Export the synthetic dataset in a format ready for machine learning tasks (e.g., CSV, JSON, or image directories).
- Face Keypoint Detection (Optional): Includes the ability to detect facial landmarks for enhanced data analysis.
---
## Contributing

Contributions are always welcome!

Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or bug fixes.



