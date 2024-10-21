import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

# Load and preprocess the dataset
data = pd.read_csv('data_science_interview_questions.csv')
data['text'] = data['question'] + " " + data['response']

# Create a Hugging Face Dataset
dataset = Dataset.from_pandas(data[['text']])

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token if necessary (GPT-2 doesn't have padding by default)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator: This will handle dynamic padding during training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-ds-questions",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
    report_to="none",  # Disable reporting to avoid errors
)

# Initialize Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the model after training
trainer.save_model("./gpt2-finetuned-ds-questions")
tokenizer.save_pretrained("./gpt2-finetuned-ds-questions")


# Load fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-ds-questions")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-ds-questions")

# Generate response
def generate_response(question):
    inputs = tokenizer(question, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Get user input for the question
question = input("Enter your question: ")
response = generate_response(question)
print("Response:", response)