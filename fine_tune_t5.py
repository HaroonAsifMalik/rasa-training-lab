import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Define current directory
current_dir = os.getcwd()

# Load Excel files
train_df = pd.read_excel(os.path.join(current_dir, "train.xlsx"))
test_df = pd.read_excel(os.path.join(current_dir, "test.xlsx"))

# Convert Pandas DataFrame to Hugging Face Dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Preprocessing function
def preprocess_function(examples):
    inputs = [
        f"Generate proposal for a {title} job: {jd}, Skills: {skills}, Budget: {budget}, Experience Level: {experience}"
        for title, jd, skills, budget, experience in zip(
            examples["Job Title"], examples["Job Description"], examples["Skills Required"],
            examples["Budget"], examples["Experience Level"]
        )
    ]
    targets = examples["Proposal"]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Apply preprocessing
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./t5_proposal_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# Train the model
trainer.train()

# Save fine-tuned model
model.save_pretrained("./t5_proposal_model")
tokenizer.save_pretrained("./t5_proposal_model")

print("Model training complete! Fine-tuned model saved in './t5_proposal_model'.")
