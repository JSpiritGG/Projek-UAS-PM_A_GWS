# 🍛 Proyek UAS Deep Learning - Chatbot Kuliner & Sejarah Jogja

## 👥 Anggota Kelompok
| No | Nama | NIM |
|----|------|-----|
| 1 | Michael Ardiyanto | 230712327 |
| 2 | Naomi Nadya Kinasih | 230712412 |
| 3 | Jovita Maria E.H | 230712428 |

**Kelas:** A  
**Nama Kelompok:** UAS DL GWS

---

## 📝 Deskripsi Proyek
Proyek ini mengembangkan chatbot berbasis Large Language Model (LLM) yang di-fine-tune menggunakan teknik **LoRA (Low-Rank Adaptation)** untuk menjawab pertanyaan seputar **kuliner dan sejarah Yogyakarta**.

Model dasar yang digunakan adalah **Gemma 2B** dari Google, yang di-fine-tune dengan dataset khusus berisi informasi kuliner dan sejarah Jogja.

---

## 🔗 Links Penting
| Komponen | Link |
|----------|------|
| 🌐 **Demo Aplikasi** | [https://huggingface.co/spaces/JSpiritGG/jogja-kuliner-chatbot](https://huggingface.co/spaces/JSpiritGG/jogja-kuliner-chatbot) |
| 🤖 **Model (HuggingFace)** | [https://huggingface.co/JSpiritGG/jogja-kuliner-chatbot](https://huggingface.co/JSpiritGG/jogja-kuliner-chatbot) |

---

## 📊 Hasil Training

### Training Statistics
| Metrik | Nilai |
|--------|-------|
| Total Steps | 117 |
| Loss Awal | ~2.3 |
| Loss Akhir | ~0.35 |
| Best Loss | 0.0544 |
| Loss Reduction | 60.4% |
| Trainable Parameters | 41,533,440 (1.56%) |

### Evaluasi Model
| Metrik | Skor |
|--------|------|
| Skor Model | 4.10 / 10 |
| Skor Kunci Jawaban | 8.00 / 10 |

---

## 🛠️ Teknologi yang Digunakan

### Model & Framework
- **Base Model:** Gemma 2B (`unsloth/gemma-2-2b-it`)
- **Fine-tuning:** LoRA (Low-Rank Adaptation) via PEFT
- **Framework:** PyTorch, Transformers (HuggingFace)
- **Optimizer:** Lion

### LoRA Configuration
```python
LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"]
)
```

### Deployment
- **Platform:** Hugging Face Spaces
- **SDK:** Gradio
- **Hardware:** ZeroGPU (Free Tier)

---

## 📁 Struktur Repository
```
├── Notebook_LLM_CHAT_X_NamaKEL.ipynb    # Notebook training & evaluasi
├── MainStreamlit_X_NamaKEL.py           # Aplikasi chatbot (Gradio)
├── predictions.json                      # Hasil prediksi test set
├── test_scenarios.json                   # Hasil pengujian 5 skenario
├── dataset_jogja_train.json             # Dataset training
├── dataset_jogja_test.json              # Dataset testing
├── requirements.txt                      # Dependencies
└── README.md                            # Dokumentasi
```

---

## 🧪 Contoh Pengujian

### Skenario 1: Kuliner Tradisional
**User:** Apa itu gudeg kering?  
**Bot:** Gudeg manis karena pengaruh budaya Mataram yang kaya gula. Manis melambangkan kemakmuran.

### Skenario 2: Lokasi Kuliner
**User:** Dimana lokasi Sate Klathak Pak Bari?  
**Bot:** Sate Klathak Pak Bari ada di Jalan Kaliurang KM 10, Patangpuluhan. Buka malam hari dengan nuansa pasar tradisional.

### Skenario 3: Sejarah
**User:** Siapa pendiri Muhammadiyah?  
**Bot:** K.H Ahmad Dahlan (Muhammad Darwis).

### Skenario 4: Sejarah
**User:** Kapan Perang Diponegoro terjadi?  
**Bot:** Tahun 1825-1830.

### Skenario 5: Rekomendasi
**User:** Rekomendasi kuliner malam di Jogja?  
**Bot:** Angkringan Lik Man untuk yang autentik. The Ratan atau House of Raminten untuk suasana modern.

---

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/USERNAME/Projek-UAS-PM_X_NamaKEL.git
cd Projek-UAS-PM_X_NamaKEL
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
python MainStreamlit_X_NamaKEL.py
```

Atau kunjungi demo online di [Hugging Face Spaces](https://huggingface.co/spaces/JSpiritGG/jogja-kuliner-chatbot).

---

## 📚 Referensi
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft)
- [Gemma Model](https://ai.google.dev/gemma)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

---

## 📄 Lisensi
MIT License

---

**© 2025 - Proyek UAS Deep Learning**
