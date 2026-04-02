# Dataset class definition
class ConversationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get conversation
        conversation = self.dataset[idx]['conversation']

        # Format conversation
        formatted_text = ""
        for turn in conversation:
            if turn["role"] == "user":
                formatted_text += f"<user> {turn['content']} "
            elif turn["role"] == "assistant":
                formatted_text += f"<assistant> {turn['content']} "

        # Tokenize
        encodings = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }