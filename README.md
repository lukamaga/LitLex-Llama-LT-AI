# LitLex-AI: Lithuanian Administrative Law Assistant

**LitLex** is a Large Language Model (LLM) fine-tuned to answer questions about the **Code of Administrative Offences of the Republic of Lithuania** (Administracinių nusižengimų kodeksas, **ANK**).

This project applies Supervised Fine-Tuning (SFT) with QLoRA to **Llama-3-8B** to build a domain-specific legal assistant that answers in Lithuanian, cites ANK articles and gives the fine ranges they set. It is a personal project I built in January 2026. It is an experiment, not a legal tool: the model often cites the wrong article, so please do not use it for legal advice.

**Built with Meta Llama 3.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-Unsloth-red?style=flat-square)
![Code license](https://img.shields.io/badge/Code%20license-MIT-green?style=flat-square)

---

## Project Motivation & Machine Learning Approach

Navigating complex legal documentation to find specific fines or regulations is often time-consuming and requires domain expertise. Traditional keyword search methods lack the semantic understanding to answer questions like *"What is the fine for X if I am a first-time offender?"*.

I wanted to see how far supervised fine-tuning alone can go here. I took the consolidated ANK text, turned it into question and answer pairs and trained Llama 3 on them, so that the model would map plain Lithuanian questions to the right articles and fine ranges without looking anything up.

### How it was built

1. **Data.** `src/dataset_builder.py` takes the consolidated ANK text (`data/ank_raw.txt`, version valid from 2026-01-12 to 2026-03-31), cuts it into 94 chunks of 15,000 characters and asks GPT-4o for 3-6 question and answer pairs per chunk. The result is 498 pairs in `data/ank_dataset.json`. They went into training without a manual review.
2. **Training.** `src/finetune.py` trains a LoRA adapter on `unsloth/llama-3-8b-bnb-4bit` with Unsloth and TRL's `SFTTrainer` on one Tesla T4.
3. **Testing.** `src/test.py` loads the adapter from Hugging Face and asks four fixed questions.

---

## Model Capabilities & Demo

The model learned the answer style of its training data well, and it often gives the right article and fine for questions that are close to the ones it was trained on. Reworded questions are where it breaks. Below are the four answers from `src/test.py`, each checked against the ANK text.

### Real Usage Examples

**User:** Kokia bauda gresia už greičio viršijimą daugiau kaip 50 km/h?

**LitLex:** Pagal ANK 416 str., bauda yra nuo 450 iki 550 eurų vairuotojams ir nuo 450 iki 700 eurų asmenims neturintiems teisės vairuoti.

*Check: correct (Art. 416 part 6), but it leaves out the mandatory license suspension of 1-6 months. Almost the same question is in the training data.*

---

**User:** Ką daryti, jei kaimynai triukšmauja naktį? Kokia bauda?

**LitLex:** Pagal ANK 299 str. 1 dalį, administracinėn atsakomybėn traukiami asmenys, kurie padarė pakartotinį nusižengimą, turi sumokėti baudą nuo 140 iki 300 eurų.

*Check: wrong. Art. 299 is about fishery products. Noise at night is Art. 488 part 1: 80-200 EUR for individuals and 200-300 EUR for heads of legal entities.*

---

**User:** Kokia bauda už nelegalų darbą?

**LitLex:** Pagal ANK 95 straipsnį, bauda darbdaviui - fiziniam asmeniui yra nuo 2000 iki 5000 eurų, o darbdavio - juridinio asmens vadovui ar kitam atsakingam asmeniui - nuo 1500 iki 5000 eurų.

*Check: correct (Art. 95 part 1). The same question is in the training data. It does not mention the higher fine for a repeat offense.*

---

**User:** Kokia bauda už triukšmavimą vakaro metu?

**LitLex:** Pagal Lietuvos Respublikos administracinių nusižengimų kodeksą, triukšmavimas vakaro metu yra numatyta kaip viena iš šiurkštų socių teisės aktų, todėl bauda asmenims nuo 60 iki 140 eurų, o juridinių asmenų vadovams ar kitiems atsakingiems asmenims - nuo 140 iki 300 eurų.

*Check: wrong. No article, broken Lithuanian and wrong amounts. The right answer is again Art. 488 part 1: 80-200 EUR and 200-300 EUR.*

### Summary of the checks

| Question | Article in the answer | Correct article | Fine in the answer | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| Speeding by more than 50 km/h | ANK 416 | ANK 416 part 6 | Right, license suspension missing | Correct, near copy of a training pair |
| Neighbors making noise at night | ANK 299 | ANK 488 part 1 | Wrong (140-300 EUR instead of 80-200 EUR) | Wrong |
| Illegal work | ANK 95 | ANK 95 part 1 | Right, repeat-offense fine missing | Correct, same question in the training data |
| Noise in the evening | none | ANK 488 part 1 | Wrong (60-140 EUR instead of 80-200 EUR) | Wrong |

So 2 of 4 are right, and both are near copies of training pairs.

---

## Training Process and Metrics

The model was trained with **Unsloth** (QLoRA) on a Tesla T4 GPU. The training objective was to minimize the loss on the 498 pairs, so the model would retain the legal facts in them.

### Training Convergence

The training loss fell from 2.71 to **0.0897**. There was no validation split, so this shows that the model memorized its training set. It does not say how often its answers are correct.

| Stage | Step | Epoch | Training Loss | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Start** | 1 | 0.02 | 2.7105 | Starting point |
| | 10 | 0.16 | 1.4465 | Fast early drop |
| **Middle** | 120 | 1.9 | 0.5252 | Steady decrease |
| | 253 | 4.1 | 0.2463 | Steady decrease |
| | 350 | 5.6 | 0.1577 | Slowing down |
| **Final** | 480 | 7.7 | 0.0840 | Near zero: the training pairs are largely memorized |
| | **500** | **8.0** | **0.0897** | End of training |

Each value is the loss of a single batch of 8 examples, so small ups and downs (like 0.0840 at step 480 and 0.0897 at step 500) are noise.

**Training Parameters:**
- **Epochs:** about 8 (500 steps x 8 samples over 498 pairs)
- **Max Steps:** 500
- **Learning Rate:** 2e-4, 20 warmup steps, linear decay
- **LoRA Rank (r):** 64, alpha 32, dropout 0, all attention and MLP projections
- **Batch Size:** 2, gradient accumulation 4 (effective batch 8)
- **Optimizer:** adamw_8bit, weight decay 0.01, seed 3407

### Training data quality

A later check of the training data against the ANK text found some problems:
- a few placeholder answers copied from the generation prompt;
- article numbers that are really part numbers of a longer article;
- superscript articles flattened by the PDF export (342¹ became "3421").

In a random sample of 40 pairs, 32 were fully correct. The full check is in the [model card](https://huggingface.co/lukashm/LitLex-Llama-LT-v1#training-data-quality).

---

## Model Availability on Hugging Face

The fine-tuned model weights (LoRA adapters) are hosted on the Hugging Face Hub, so the GitHub repository stays small.

**Official Model Repository:**
[https://huggingface.co/lukashm/LitLex-Llama-LT-v1](https://huggingface.co/lukashm/LitLex-Llama-LT-v1)

The model card there has the full training details and instructions for running the model with Unsloth, with plain transformers and peft, or on a Mac.

### Automatic Integration

`src/test.py` acts as a client. When you run it, it downloads the adapter and the 4-bit base model from the Hugging Face Hub and loads them together, so no manual model download is needed.

## Project Structure

This repository is organized into data processing, training and inference scripts.

```text
LitLex-Llama-LT-AI/
    data/
        ank_raw.txt            # consolidated ANK text
        ank_dataset.json       # 498 QA pairs generated with GPT-4o
    src/
        dataset_builder.py     # builds the JSON dataset from the raw text
        finetune.py            # QLoRA training with Unsloth
        test.py                # asks the four example questions
    requirements.txt           # dependencies for Colab or a Linux CUDA machine
    README.md
```

## Installation and Usage

### What you need

| Requirement | Details |
| :--- | :--- |
| **GPU** | An NVIDIA GPU with about 7 GB of VRAM. A free T4 on Google Colab or Kaggle is enough |
| **Python** | 3.10 or newer |
| **Disk** | About 7 GB for the model files |
| **Accounts** | None. The base model and the adapter are public, so no Hugging Face login is needed. Only step 4 needs an OpenAI API key |

It works on Google Colab, on Linux and on Windows (natively or through WSL). The scripts load the model in 4-bit with Unsloth and call CUDA directly, so they will not run on a Mac as they are. To try the model on Apple Silicon, use option C in the [model card](https://huggingface.co/lukashm/LitLex-Llama-LT-v1#option-c-apple-silicon-mps-or-cpu).

### What gets downloaded

`src/test.py` downloads two things from Hugging Face the first time you run it:

| File | Size | Where it comes from |
| :--- | :--- | :--- |
| Base model `unsloth/llama-3-8b-bnb-4bit` | about 5.7 GB | [Hugging Face](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit) |
| LitLex adapter `lukashm/LitLex-Llama-LT-v1` | 671 MB | [Hugging Face](https://huggingface.co/lukashm/LitLex-Llama-LT-v1) |

Both are saved in the Hugging Face cache (`~/.cache/huggingface/hub`), so later runs start without downloading again. The ANK text and the training pairs are already in this repo under `data/`.

### Quick start on Google Colab

1. Open a new notebook at [colab.research.google.com](https://colab.research.google.com).
2. Go to **Runtime > Change runtime type**, choose **T4 GPU** and save.
3. Run these cells:

```python
!git clone https://github.com/lukamaga/LitLex-Llama-LT-AI.git
%cd LitLex-Llama-LT-AI
!pip install unsloth
!python src/test.py
```

The first run takes a few minutes, mostly for installing Unsloth and downloading the base model. Then the script prints the four example questions with the model's answers.

### On your own Linux machine with an NVIDIA GPU

#### 1. Clone the repository
```bash
git clone https://github.com/lukamaga/LitLex-Llama-LT-AI.git
cd LitLex-Llama-LT-AI
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt` installs Unsloth from its GitHub repository with the Colab extras, which is how I trained the model in January 2026. If that install fails on a newer setup, `pip install unsloth` is usually enough to run `src/test.py`.

#### 3. Run the AI Assistant
The script downloads the base model and the adapter (see above) and asks the four example questions.
```bash
python src/test.py
```

To ask your own question, change the `questions` list in `src/test.py`. Keep the prompt template as it is: the model was trained on that exact format.

#### 4. Rebuild the dataset (optional)
Needs an OpenAI API key and makes about 94 calls to GPT-4o, one per chunk of the ANK text. The output will differ from the published file.
```bash
cd src
export OPENAI_API_KEY=...
python dataset_builder.py
```

#### 5. Train the adapter again (optional)
Takes the pairs from `data/ank_dataset.json` and saves the new adapter to `models/litlex_adapter`.
```bash
cd src
python finetune.py
```

Run steps 4 and 5 from inside `src/`, because both scripts use paths like `../data/ank_dataset.json`.

### On Windows

The easiest way is the [Colab quick start](#quick-start-on-google-colab) above, because nothing has to be installed. To run it on your own Windows PC with an NVIDIA GPU, there are two ways. Both follow [Unsloth's Windows guide](https://unsloth.ai/docs/get-started/install/windows-installation).

**Native Windows with Conda**

1. Open PowerShell and run `nvidia-smi`. It should show your GPU and a CUDA version. If it fails, install or update the NVIDIA driver first.
2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and [Git for Windows](https://git-scm.com/download/win), then open **Anaconda PowerShell Prompt**.
3. Run:

```powershell
conda create --name litlex python=3.12 -y
conda activate litlex
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available())"
```

The last command must print `True`. If it prints `False`, PyTorch does not see your GPU. Change `cu130` to the build that matches the CUDA version from `nvidia-smi` (the list is on [pytorch.org](https://pytorch.org/get-started/locally/)) and run the install again.

4. Then install Unsloth and run the model:

```powershell
pip install unsloth
git clone https://github.com/lukamaga/LitLex-Llama-LT-AI.git
cd LitLex-Llama-LT-AI
python src/test.py
```

On Windows, use `pip install unsloth` instead of `pip install -r requirements.txt`. The requirements file installs the Colab build of Unsloth, which is meant for Colab and Linux.

**WSL (Linux inside Windows)**

In PowerShell, install Ubuntu:

```powershell
wsl.exe --install Ubuntu-24.04
```

Then open Ubuntu and follow the [Linux steps](#on-your-own-linux-machine-with-an-nvidia-gpu) above. Keep the NVIDIA driver installed on Windows itself, not inside Ubuntu: WSL uses the Windows driver.

**Small differences on Windows**

- To set the OpenAI key for step 4 in PowerShell, use `$env:OPENAI_API_KEY="..."` instead of `export`.
- The Hugging Face cache is in `C:\Users\<your name>\.cache\huggingface\hub`.

## Performance & Scope

The model reproduces its training set well (loss < 0.1), but there is no held-out evaluation, so there is no accuracy number. Please keep the following in mind:
1. **Narrow Scope:** The model only knows the **Code of Administrative Offences (ANK)**. It was not trained on the traffic rules (KET), the Criminal Code (BK) or the Civil Code (CK). Questions outside this scope may get generic or wrong answers.
2. **Memorization:** It does best on questions it has seen. Reworded questions often get a confident answer with the wrong article or amount. This is a known limitation of a model that answers from memory without retrieval (RAG).
3. **Frozen law version:** The data is the ANK version valid until 2026-03-31. The Code has been amended since, so some fines may be out of date.
4. **Unreviewed training data:** The pairs were generated by GPT-4o and not checked before training.

## What comes next

I'm working on a new version that looks up the relevant articles in the official ANK text and cites them, instead of relying on what the model memorized.

## License

- The code in this repo is under the MIT License (see `LICENSE`).
- The adapter on Hugging Face is fine-tuned from Meta Llama 3, so it is covered by the [Meta Llama 3 Community License](https://github.com/meta-llama/llama3/blob/main/LICENSE) and the [Acceptable Use Policy](https://github.com/meta-llama/llama3/blob/main/USE_POLICY.md), not by MIT.
- The ANK is a legal act, and legal acts are not protected by copyright in Lithuania (Art. 5 of the Law on Copyright and Related Rights).
- The training pairs in `data/ank_dataset.json` were generated with OpenAI's GPT-4o.

## Disclaimer

This model was built for educational and research purposes. It is not a substitute for professional legal advice. Check the official text on [e-seimas.lrs.lt](https://e-seimas.lrs.lt) or ask a lawyer. I am not responsible for any inaccuracies produced by the model.

---
**Developed by Lukash Magalinski**
