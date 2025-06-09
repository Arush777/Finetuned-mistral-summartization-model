# Fine-Tuning Mistral for Abstractive Summarization on CNN/DailyMail

## Project Overview

This project demonstrates the efficient fine-tuning of a quantized Mistral language model for abstractive text summarization using the CNN/DailyMail dataset. It leverages **QLoRA (4-bit quantization)** and **LoRA adapters** for memory-efficient, parameter-efficient training on consumer hardware and provides a full pipeline from data preprocessing to evaluation.

---

## Features

- **4-bit Quantization (QLoRA):** Enables training large models on limited hardware.
- **LoRA Adapters:** Parameter-efficient fine-tuning for fast adaptation.
- **Custom Training Pipeline:** Modular class-based implementation for easy reuse.
- **Automated Evaluation:** Computes ROUGE and BERTScore metrics on the test set.

---

## Model & Dataset

- **Base Model:** Mistral (quantized, causal language modeling)
- **Dataset:** CNN/DailyMail (v3.0.0, test split)

---

## Example Evaluation Results




```
ROUGE-1: 0.47
ROUGE-2: 0.23
ROUGE-L: 0.39

BERTScore (F1):
Average: 0.8623
Min: 0.7912
Max: 0.8975

```
*Note: These are sample results for demonstration. Actual performance may vary depending on training configuration.*

---

## Usage

### Training

```
tuner = SFTFineTuner(
    model_name="mistral-base",
    data_df=my_dataframe,      # Your pandas DataFrame with a 'prompt' column
    output_dir="./output"
)
tuner.run()

```

### Loading and Inference

```

model, tokenizer = load_finetuned_model("mistral-base", "./output", device="cuda")
summary = evaluate_article(article_text, tokenizer, model, device="cuda")
print(summary)

```


### Evaluation

The included script will:
- Load the CNN/DailyMail test set
- Generate summaries
- Compute ROUGE and BERTScore metrics

---

## Technical Highlights

- **Efficient Quantized Training:** 4-bit quantization with BitsAndBytes.
- **LoRA Integration:** Reduces trainable parameters for faster, more efficient fine-tuning.
- **Comprehensive Evaluation:** Automated reporting of ROUGE and BERTScore.

---

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers, Datasets, PEFT, BitsAndBytes, Evaluate, BERTScore
- GPU with bfloat16 support recommended

---

## Contact

For questions or collaboration, please open an issue or contact me via email.


