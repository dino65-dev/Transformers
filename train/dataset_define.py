from torch.utils.data import Dataset, IterableDataset
import torch
import glob
import os

# ======================== Original: Conversation Dataset ========================
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


# ======================== New: SlimPajama-6B Dataset ========================
class SlimPajamaDataset(IterableDataset):
    """
    Streams SlimPajama-6B from local Parquet files (cloned via Oxen).
    
    Dataset schema (from Oxen hub):
        - text (str)               : raw document text
        - meta (struct)            : contains redpajama_set_name (source: C4, CommonCrawl, etc.)
        - __index_level_0__ (int)  : row index
    
    Reads Parquet files one by one (memory-efficient), tokenizes on-the-fly
    with chunking for long documents, returns (input_ids, labels) for causal LM.
    """

    def __init__(self, data_dir, tokenizer, max_length=2048, file_pattern="**/*.parquet"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Find all parquet files
        self.parquet_files = sorted(
            glob.glob(os.path.join(data_dir, file_pattern), recursive=True)
        )
        if not self.parquet_files:
            raise FileNotFoundError(
                f"No .parquet files found in {data_dir} with pattern {file_pattern}"
            )
        print(f"Found {len(self.parquet_files)} parquet files in {data_dir}")

    def _tokenize_and_chunk(self, text):
        """
        Tokenize a document and split into max_length chunks.
        For causal LM: labels = input_ids (shifted internally by loss fn).
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Add EOS at end of each document
        if self.tokenizer.eos_token_id is not None:
            token_ids.append(self.tokenizer.eos_token_id)

        # Split into chunks of max_length
        for i in range(0, len(token_ids), self.max_length):
            chunk = token_ids[i : i + self.max_length]

            # Skip very short chunks (< 64 tokens)
            if len(chunk) < 64:
                continue

            # Pad if necessary
            attention_mask = [1] * len(chunk)
            pad_len = self.max_length - len(chunk)
            if pad_len > 0:
                chunk = chunk + [self.tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

            input_ids = torch.tensor(chunk, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            labels = input_ids.clone()
            # Mask padding in labels so loss ignores them
            labels[attention_mask == 0] = -100

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    def __iter__(self):
        """
        Iterate over all parquet files, read each row's 'text' column,
        tokenize + chunk, and yield training samples.
        """
        import pyarrow.parquet as pq

        worker_info = torch.utils.data.get_worker_info()

        # Split files across DataLoader workers
        if worker_info is not None:
            per_worker = len(self.parquet_files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.parquet_files)
            files = self.parquet_files[start:end]
        else:
            files = self.parquet_files

        for pq_file in files:
            try:
                table = pq.read_table(pq_file, columns=["text"])
                texts = table.column("text").to_pylist()

                for text in texts:
                    if text and len(text.strip()) > 0:
                        yield from self._tokenize_and_chunk(text)
            except Exception as e:
                print(f"Warning: Error reading {pq_file}: {e}")
                continue