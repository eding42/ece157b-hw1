"""
A toy script to play around with the tokenization process.
"""
from typing import Iterable, List, Optional, Union, TYPE_CHECKING

try:
    from tokenizers import (
        Tokenizer,
        models,
        pre_tokenizers,
        decoders,
        trainers,
        AddedToken,
        normalizers,
    )

    if TYPE_CHECKING:
        from pathlib import Path
        from tokenizers import Encoding
except ImportError:
    print(
        "You need to install the 'tokenizers' package to use this script.\n"
        "Try running the library install cell from the notebook.\n"
        "Make sure this script is being run with the same Python environment.\n"
    )
    raise


# Vocabulary size - number of tokens to use to break up the text (and learn)
VOCAB_SIZE_CAP: int = 20000

# Minimum token frequency - tokens that appear less than this number of times will be ignored
# (So single-use complex words will be ignored)
MIN_TOKEN_FREQUENCY: int = 2
# If no tokens appear at least this many times, the vocabulary size will be reduced
# below VOCAB_SIZE

TRAINER = trainers.WordPieceTrainer(
    vocab_size=VOCAB_SIZE_CAP,
    min_frequency=MIN_TOKEN_FREQUENCY,
    special_tokens=["<unk>"],
    show_progress=True,
)


def _create_fresh_tokenizer() -> Tokenizer:
    """Create a fresh tokenizer with default settings"""
    # We'll create a "WordPiece" tokenizer -- this is the kind used by BERT.
    tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))

    # This sequence of normalizers is the same as the one used by BERT.
    #  NFD: Normalization Form D (canonical decomposition)
    #       This is fancy Unicode speak for transforming accented characters
    #       into their base form and the accent separately.
    #       e.g. é -> e + ´
    #  Lowercase: Lowercase the text
    #  StripAccents: Remove the accents from the text (e.g. ´ from é)
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
        ]
    )

    # We'll use a whitespace pre-tokenizer.
    # This will split the text on whitespace and punctuation first,
    # then tokenize the words separately using the WordPiece model.
    # That way, we can keep punctuation as separate tokens,
    # which might be useful for the model.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # pre_tokenizers.WhitespaceSplit() would also work here,
    # but it would keep punctuation attached to words,
    # which might not be what we want.

    # Finally, we'll use the WordPiece decoder
    # to convert the tokens back into text.
    tokenizer.decoder = decoders.WordPiece()

    return tokenizer


class SimpleWordPieceTokenizer:
    """A trimmed down Tokenizer class exposing only the methods we need."""

    __slots__ = ("__tokenizer",)

    def __init__(
        self,
        training_files: Optional[Iterable[Union[str, "Path"]]] = None,
    ) -> None:
        self.__tokenizer: Optional[Tokenizer] = None

        if training_files is not None:
            self.train(training_files)

    def ensure_trained(self) -> None:
        """Ensure the tokenizer is trained"""
        if not self.__tokenizer is not None:
            raise RuntimeError("Tokenizer not trained!")

    def ensure_untrained(self) -> None:
        """Ensure the tokenizer is not trained"""
        if self.__tokenizer is not None:
            raise RuntimeError("Tokenizer already trained!")

    def get_vocab(self) -> dict[str, int]:
        """Get the vocabulary of the tokenizer"""
        self.ensure_trained()
        return self.__tokenizer.get_vocab()

    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary"""
        self.ensure_trained()
        return self.__tokenizer.get_vocab_size()

    def train(self, files: Iterable[str]) -> None:
        """Train the tokenizer on the given files"""
        self.ensure_untrained()
        self.__tokenizer = _create_fresh_tokenizer()
        self.__tokenizer.train(files, TRAINER)
        self._finalize_training()

    def train_from_iterator(self, training_data: Iterable[str]) -> None:
        """Train the tokenizer on the given data"""
        self.ensure_untrained()
        self.__tokenizer = _create_fresh_tokenizer()
        self.__tokenizer.train_from_iterator(training_data, TRAINER)
        self._finalize_training()

    def _finalize_training(self) -> None:
        # Finally, we add a newline token to the tokenizer
        # so that the model can learn to predict line breaks
        # (and thus line lengths) instead of smushing it all
        # together into one giant line with character names
        # and stage directions and everything.
        self.__tokenizer.add_tokens([AddedToken("\n", normalized=False)])

    def encode(self, text: str) -> "Encoding":
        """Encode the given text"""
        self.ensure_trained()
        return self.__tokenizer.encode(text)

    def encode_batch(self, texts: Iterable[str]) -> List["Encoding"]:
        """Encode the given texts"""
        self.ensure_trained()
        return self.__tokenizer.encode_batch(texts)

    def id_to_token(self, tok_id: int) -> str:
        """Get the token corresponding to the given ID"""
        self.ensure_trained()
        return self.__tokenizer.id_to_token(tok_id)

    def token_to_id(self, token: str) -> int:
        """Get the ID corresponding to the given token"""
        self.ensure_trained()
        return self.__tokenizer.token_to_id(token)

    def decode(self, ids: List[int]) -> str:
        """Decode the given token IDs"""
        self.ensure_trained()
        return self.__tokenizer.decode(ids)

    def save(self, path: Union[str, "Path"]) -> None:
        """Save the tokenizer to the given path"""
        self.ensure_trained()
        if not isinstance(path, str):
            path = str(path)
        self.__tokenizer.save(path)

    def _set_tokenizer(self, tokenizer: Tokenizer) -> None:
        """Set the tokenizer to the given tokenizer

        Assumes the tokenizer has already been trained.

        This is a private method for use by the `load` staticmethod.
        """
        self.__tokenizer = tokenizer

    @staticmethod
    def load(path: Union[str, "Path"]) -> "SimpleWordPieceTokenizer":
        """Load the tokenizer from the given path"""
        if not isinstance(path, str):
            path = str(path)
        t = SimpleWordPieceTokenizer()
        # pylint: disable-next=protected-access
        t._set_tokenizer(Tokenizer.from_file(path))
        return t


if __name__ == "__main__":
    import sys

    def _main() -> None:
        if len(sys.argv) < 2:
            print("Usage: python tokenizer_toy.py <file> [<file> ...]")
            sys.exit(1)

        arg = sys.argv[1]

        if arg.lower() in "-h --help".split():
            print("Usage: python tokenizer_toy.py <file> [<file> ...]")
            print(
                "  Train a tokenizer on the given files then provide\n"
                "  an interactive prompt to explore tokenizations of\n"
                "  user-input strings."
            )
            sys.exit(0)

        t = SimpleWordPieceTokenizer()
        print(">>> Training tokenizer on file(s)...")
        t.train(sys.argv[1:])
        print(f">>> Done! (vocab size: {t.get_vocab_size()})")
        print(">>> Type a string to see its tokens and decoded form.")
        print('>>> Type "q" or "quit" to quit.')

        while True:
            try:
                s = input("> ")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            if s.lower() in "q quit e exit end done close".split():
                print("Goodbye!")
                break
            encoded = t.encode(s)
            print("Tokens:\t", encoded.tokens)
            print("IDs:\t", encoded.ids)
            print("Decode:\t", t.decode(encoded.ids))

    _main()
