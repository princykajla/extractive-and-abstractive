from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from rouge_score import rouge_scorer

# 1. Load Public Dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")  # or "xsum"
dataset = dataset.rename_columns({"article": "text", "highlights": "summary"})

# 2. Initialize Model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# 3. Tokenize (1% of data for quick testing)
small_dataset = dataset["train"].shuffle(seed=42).select(range(len(dataset["train"])//100))

def preprocess(examples):
    inputs = tokenizer(examples["text"], truncation=True, max_length=1024)
    labels = tokenizer(examples["summary"], truncation=True, max_length=128)
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"]
    }

tokenized_data = small_dataset.map(preprocess, batched=True)

# 4. Training Setup
training_args = TrainingArguments(
    output_dir="models/fine-tuned-bart",
    per_device_train_batch_size=2,  # Reduced for Colab compatibility
    num_train_epochs=1,             # Quick demo run
    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=100,
)

# 5. Compute Metrics
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    scores = [scorer.score(label, pred) for label, pred in zip(decoded_labels, decoded_preds)]
    return {
        "rouge1": np.mean([s["rouge1"].fmeasure for s in scores]),
        "rougeL": np.mean([s["rougeL"].fmeasure for s in scores])
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    compute_metrics=compute_metrics
)

# 6. Execute Training
trainer.train()
model.save_pretrained("models/fine-tuned-bart")