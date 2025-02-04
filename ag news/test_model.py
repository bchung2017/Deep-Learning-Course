from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate

dataset = load_dataset("ag_news")

model_directory = "./model_backup"
model = AutoModelForSequenceClassification.from_pretrained(model_directory)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

tokenized_test_dataset = dataset['test'].map(preprocess_function, batched=True)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"]}

training_args = TrainingArguments(
    output_dir = "./test_results",
    per_device_eval_batch_size = 16,
)

trainer = Trainer(
    model = model, 
    args = training_args,
    eval_dataset = tokenized_test_dataset,
    tokenizer = tokenizer,
    compute_metrics=compute_metrics
)

eval_results = trainer.evaluate()

print(f"Results: {eval_results}")
print(f"Test set accuracy: {eval_results['eval_accuracy']}")