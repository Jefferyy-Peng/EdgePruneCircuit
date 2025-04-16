import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Function to predict the next word
def predict_next_word(prompt):
    # Tokenize input and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits for the last token
    logits = outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)

    # Get the token with the highest probability
    next_token_id = torch.argmax(logits, dim=-1).item()

    # Decode the predicted token to a word
    next_word = tokenizer.decode(next_token_id)

    return next_word


# Example usage
prompt = "The cat is"
next_word = predict_next_word(prompt)

print(f"Prompt: {prompt}")
print(f"Predicted Next Word: {next_word}")
