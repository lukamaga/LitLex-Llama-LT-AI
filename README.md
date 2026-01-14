# LitLex-AI: Lithuanian Administrative Law Assistant

**LitLex** is a specialized Large Language Model (LLM) designed to provide accurate legal information based on the **Administrative Code of the Republic of Lithuania (ANK)**.

This project demonstrates the application of Supervised Fine-Tuning (SFT) techniques on the **Llama-3-8B** architecture to create a domain-specific legal assistant capable of interpreting regulations and calculating fines in the Lithuanian language.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-Unsloth-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Project Motivation & Machine Learning Approach

Navigating complex legal documentation to find specific fines or regulations is often time-consuming and requires domain expertise. Traditional keyword search methods lack the semantic understanding to answer questions like *"What is the fine for X if I am a first-time offender?"*.

To solve this, we applied **Machine Learning (Deep Learning)** methods to transform raw legal texts into a conversational intelligence. By utilizing **Supervised Fine-Tuning (SFT)** on the Llama-3 architecture, we successfully mapped natural language user queries to specific legal articles and penalty ranges defined in the Administrative Code.

---

## Model Capabilities & Demo

The model has been fine-tuned to achieve high factual accuracy. Unlike generic LLMs, LitLex specifically memorizes article numbers and fine ranges from the ANK code.

### Real Usage Examples

**User:** Kokia bauda gresia už greičio viršijimą daugiau kaip 50 km/h?
**LitLex:** Pagal ANK 416 str., bauda yra nuo 450 iki 550 eurų vairuotojams ir nuo 450 iki 700 eurų asmenims neturintiems teisės vairuoti.

---

**User:** Ką daryti, jei kaimynai triukšmauja naktį? Kokia bauda?
**LitLex:** Pagal ANK 299 str. 1 dalį, administracinėn atsakomybėn traukiami asmenys, kurie padarė pakartotinį nusižengimą, turi sumokėti baudą nuo 140 iki 300 eurų.

---

**User:** Kokia bauda už nelegalų darbą?
**LitLex:** Pagal ANK 95 straipsnį, bauda darbdaviui – fiziniam asmeniui yra nuo 2000 iki 5000 eurų, o darbdavio – juridinio asmens vadovui ar kitam atsakingam asmeniui – nuo 1500 iki 5000 eurų.

---

**User:** Kokia bauda už triukšmavimą vakaro metu?
**LitLex:** Pagal Lietuvos Respublikos administracinių nusižengimų kodeksą, triukšmavimas vakaro metu yra numatyta kaip viena iš šiurkštų socių teisės aktų, todėl bauda asmenims nuo 60 iki 140 eurų, o juridinių asmenų vadovams ar kitiems atsakingiems asmenims – nuo 140 iki 300 eurų.

---

## Training Process and Metrics

The model was trained using **Unsloth** (QLoRA optimization) on a Tesla T4 GPU. The training objective was to minimize loss to achieve high retention of legal facts.

### Training Convergence
The training process showed a dramatic improvement in accuracy, reaching a final loss of **0.0897**, which indicates a high level of dataset memorization.

| Stage | Step | Training Loss | Interpretation |
| :--- | :--- | :--- | :--- |
| **Start** | 1 | 2.7105 | Initial state, low understanding |
| | 10 | 1.4465 | Rapid adaptation to syntax |
| **Middle** | 120 | 0.5252 | Learning legal structure |
| | 253 | 0.2463 | Fact acquisition phase |
| | 350 | 0.1577 | Refining specific numbers |
| **Final** | 480 | 0.0840 | High precision achieved |
| | **500** | **0.0897** | **Convergence complete** |

**Training Parameters:**
- **Epochs:** 8
- **Max Steps:** 500
- **Learning Rate:** 2e-4
- **LoRA Rank (r):** 64
- **Batch Size:** 2

---

## Model Availability on Hugging Face

The fine-tuned model weights (LoRA adapters) are hosted on the Hugging Face Hub. This ensures easy access and version control without increasing the size of the GitHub repository.

**Official Model Repository:**
[https://huggingface.co/lukashm/LitLex-Llama-LT-v1](https://huggingface.co/lukashm/LitLex-Llama-LT-v1)

### Automatic Integration
The project is designed for seamless integration. The `test.py` script acts as a client: upon execution, it automatically connects to the Hugging Face Hub, downloads the specific fine-tuned adapters, and merges them with the base Llama-3 model in real-time. No manual model download is required.

## Project Structure

This repository is organized into data processing, training, and inference modules.

```text
LitLex-Llama-LT-AI/
├── data/
│   ├── ank_raw.txt             # Original legal text source
│   └── ank_dataset.json        # Synthetic QA dataset generated via GPT-4o
├── src/
│   ├── dataset_builder.py      # Script to generate JSON dataset from raw text
│   ├── finetune.py             # Main training script using Unsloth/SFT
│   └── test.py                 # CLI Chat interface for testing the model
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Installation and Usage

To run the Legal AI Assistant on your local machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/LitLex-Llama-LT-AI.git
cd LitLex-Llama-LT-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the AI Assistant
The inference script automatically downloads the fine-tuned model weights from Hugging Face.
```bash
python src/test.py
```

## Performance & Scope

This project achieves exceptional results in reproducing legal texts from the training set (Loss < 0.1). However, users should be aware of the following technical characteristics:
1.  **Narrow Scope:** The model is strictly specialized in the **Administrative Code (ANK)**. It does not possess knowledge of traffic rules (KET) or criminal law unless they directly overlap with administrative fines. Queries outside this scope may yield generic answers.
2.  **Generative Nature:** Despite high accuracy, as a generative ML model, it may occasionally "hallucinate" article numbers or fine amounts. This is a known limitation of current LLM architectures without RAG (Retrieval-Augmented Generation).

## Disclaimer

This Artificial Intelligence model is developed for educational and research purposes. While it demonstrates high accuracy in citing the Administrative Code (ANK), it is not a substitute for professional legal advice. The developers are not responsible for any inaccuracies produced by the model.

---
**Developed by Lukash Magalinski**
