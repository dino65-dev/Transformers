from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# Add special tokens including pad token
special_tokens = {
    'pad_token': '[PAD]',
    'additional_special_tokens': ["<user>", "<assistant>"]
}
num_added = tokenizer.add_special_tokens(special_tokens)
print(f"Pad token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
print(f"Added {num_added} special tokens")