from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

def create_32k_tokenizer(dataset, vocab_size=32000):
    """Create a custom 32K BPE tokenizer"""

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Setup trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )

    # Prepare training data
    def get_training_corpus():
        for item in dataset["train"]:
            conversation = item['conversation']
            for turn in conversation:
                yield turn['content']

    # Train tokenizer
    tokenizer.train_from_iterator(get_training_corpus(), trainer)

    # Convert to HuggingFace tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    hf_tokenizer.pad_token = "<pad>"
    hf_tokenizer.eos_token = "<eos>"
    hf_tokenizer.bos_token = "<bos>"
    hf_tokenizer.unk_token = "<unk>"

    return hf_tokenizer

# Create custom tokenizer
tokenizer = create_32k_tokenizer(dataset, vocab_size=32000)

# Verify the changes worked
def verify_32k_setup():
    print("=== 32K Vocabulary Setup Verification ===")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

# Run verification
verify_32k_setup()
