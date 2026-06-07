from collections.abc import Iterable, Iterator

class Tokenizer:
  def __init__(self, vocab, merges, special_tokens=None):
    raise NotImplementedError
  
  @classmethod
  def from_files(cls,
                 vocab_filepath: str,
                 merges_filepath: str,
                 special_tokens: list[int] | None = None):
    raise NotImplementedError
  
  def encode(self, text: str) -> list[int]:
    raise NotImplementedError
  
  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    raise NotImplementedError
  
  def decode(self, ids: list[int]) -> str:
    raise NotImplementedError