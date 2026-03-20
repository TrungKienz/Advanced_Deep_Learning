import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import evaluate
from sklearn.model_selection import train_test_split
import torch
import os
import sys
import json
from datetime import datetime

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Supported pretrained models
AVAILABLE_MODELS = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    "electra-base-discriminator"
]

# Default model
DEFAULT_MODEL = "bert-base-uncased"

def load_and_prepare_dataset(sample_size=None):
    # Load IMDB dataset and split into train/val/test.
    print("\n1. Loading IMDB dataset...")
    df = pd.read_csv('IMDB Dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample review:\n{df['review'].iloc[0][:200]}...\n")

    # Convert sentiment labels: positive → 1, negative → 0
    df['label'] = (df['sentiment'] == 'positive').astype(int)

    # Optionally use a smaller subset for faster experimentation
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Using subset of {sample_size} samples")

    # Split dataset: train (81%), validation (9%), test (10%)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Convert pandas DataFrame to HuggingFace Dataset format
    train_dataset = Dataset.from_pandas(train_df[['review', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['review', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['review', 'label']])

    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    return dataset


def tokenize_dataset(dataset, tokenizer, max_length=128):
    # Tokenize text data using the provided tokenizer.
    print("\n2. Tokenizing dataset...")

    def preprocess_function(examples):
        return tokenizer(
            examples['review'],
            max_length=max_length,
            padding='max_length',
            truncation=True
        )

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Remove raw text and rename label column for Trainer compatibility
    tokenized_dataset = tokenized_dataset.remove_columns(['review'])
    tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
    tokenized_dataset.set_format('torch')

    print(f"Tokenized dataset: {tokenized_dataset}")
    return tokenized_dataset


# Metrics Setup
def get_compute_metrics():
    # Define evaluation metrics: Accuracy and Weighted F1-score
    metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
        return {
            'accuracy': accuracy['accuracy'],
            'f1': f1['f1']
        }

    return compute_metrics


# Main Training Function
def train_sentiment_model(model_name, dataset, tokenized_dataset, num_epochs=3, 
                         batch_size=16, learning_rate=2e-5, sample_size=None):
    # Train a transformer model for sentiment analysis, evaluate it, and save results.
    print(f"\n{'='*80}")
    print(f"Training {model_name.upper()} on IMDB Sentiment Analysis")
    print(f"{'='*80}\n")
    
    # Create output directory for this model
    safe_model_name = model_name.replace('/', '-')
    output_dir = f'./sentiment-models/{safe_model_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and model
    print(f"3. Loading {model_name} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    print(f"Model loaded with {model.num_parameters():,} parameters")

    # Configure training process
    print("\n4. Setting training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        push_to_hub=False,
        logging_steps=100,
        seed=42,
    )

    # Initialize Trainer
    print("\n5. Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=get_compute_metrics(),
    )

    # Train model
    print("\n6. Starting training...")
    train_result = trainer.train()

    # Evaluate on Test Set
    print("\n7. Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    print(f"Test Results:\n{json.dumps(test_results, indent=2)}")

    # Save Model and Tokenizer
    print("\n8. Saving model and tokenizer...")
    model_save_path = f'{output_dir}/final'
    os.makedirs(model_save_path, exist_ok=True)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")

    # Save training results
    results_file = f'{output_dir}/results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'model': model_name,
            'test_results': test_results,
            'training_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"Results saved to {results_file}")

    # Test on Sample Reviews
    print("\n9. Testing on sample reviews...")
    modelfor_inference = AutoModelForSequenceClassification.from_pretrained(model_save_path).to(device)
    tokenizer_for_inference = AutoTokenizer.from_pretrained(model_save_path)

    def predict_sentiment(text):
        inputs = tokenizer_for_inference(text, return_tensors='pt', max_length=128, 
                                        padding='max_length', truncation=True).to(device)
        with torch.no_grad():
            outputs = modelfor_inference(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        sentiment = 'positive' if prediction == 1 else 'negative'
        confidence = max(probabilities)

        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'probabilities': {
                'negative': float(probabilities[0]),
                'positive': float(probabilities[1])
            }
        }

    test_reviews = [
        "This movie was absolutely amazing! I loved every second of it.",
        "This was the worst movie I've ever seen. Complete waste of time.",
        "It was okay, nothing special but worth watching."
    ]

    for review in test_reviews:
        result = predict_sentiment(review)
        print(f"\nReview: {review}")
        print(f"Prediction: {result['sentiment']} (confidence: {result['confidence']:.4f})")
        print(f"Probabilities - Negative: {result['probabilities']['negative']:.4f}, Positive: {result['probabilities']['positive']:.4f}")

    print(f"\n✓ Training {model_name} complete!")
    print(f"Model saved to: {model_save_path}\n")

    return model_save_path


# Main Entry Point
if __name__ == "__main__":
    # Read model name from command line (if provided)
    if len(sys.argv) > 1:
        model_to_train = sys.argv[1]
        if model_to_train not in AVAILABLE_MODELS:
            print(f"Unknown model: {model_to_train}")
            print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
            sys.exit(1)
    else:
        model_to_train = DEFAULT_MODEL

    # Optional dataset size reduction for faster debugging
    sample_size = None  # Set to e.g., 10000 for faster testing

    # Load dataset
    dataset = load_and_prepare_dataset(sample_size=sample_size)

    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(model_to_train)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Train the model
    model_path = train_sentiment_model(
        model_name=model_to_train,
        dataset=dataset,
        tokenized_dataset=tokenized_dataset,
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        sample_size=sample_size
    )

    print(f"\n{'='*80}")
    print(f"To train another model, run:")
    print(f"  python main.py distilbert-base-uncased")
    print(f"  python main.py roberta-base")
    print(f"\nAvailable models: {', '.join(AVAILABLE_MODELS)}")
    print(f"{'='*80}\n")


    # Results Summary
    # bert-base-uncased: Accuracy: 0.9054, F1-score: 0.9054, time: ~3h20m (best trade-off baseline, stable performance)
    # distilbert-base-uncased: Accuracy: 0.8904, F1-score: 0.8904, time: ~35m (fastest, lightweight, ~5–6x speedup with slight accuracy drop)
    # roberta-base: Accuracy: 0.9136, F1-score: 0.9136, time: ~2h35m (best accuracy, higher computational cost)
