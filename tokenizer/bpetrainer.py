import token
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from tokenizers.tokenizers import SentencePieceBPETokenizer

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.BPEDecoder()
trainer = trainers.SentencePieceTrainer(
    vocab_size = 10000,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    special_tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
    ],
)

data = [
    "Beautiful is better than ugly.",
    "Explicit is better than implicit.",
    "Simple is better than complex.",
    "Complex is better than complicated.",
    "Flat is better than nested.",
    "Sparse is better than dense.",
    "Readability counts."
]
tokenizer.train_from_iterator(data, trainer=trainer)