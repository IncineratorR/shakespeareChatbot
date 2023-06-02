from transformers import GPT2LMHeadModel, GPT2Tokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Path to the Shakespeare Dataset file
dataset_path = 'shakespeare.txt'

# Load the GPT-2 model and tokenizer
model_name = 'gpt2'  # Pretrained GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add special tokens for the Shakespeare Dataset
special_tokens = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'additional_special_tokens': ['<CHARACTER>', '<STAGE_DIRECTIONS>']}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# Read the dataset file
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset_text = f.read()

# Prepare the LineByLineTextDataset for training
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=dataset_path, block_size=128)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./shakespeare_model',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_total_limit=2
)

# Create the Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
