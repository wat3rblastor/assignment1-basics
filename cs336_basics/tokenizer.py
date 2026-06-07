import json
import regex as re

from collections.abc import Iterable, Iterator
from collections import defaultdict


class Tokenizer:
  def __init__(self, vocab: dict[int, bytes], 
               merges: list[tuple[bytes, bytes]], 
               special_tokens: list[int] | None = None):
    # tokenID -> bytes
    self.vocab = vocab
    self.merges = merges
    self.special_tokens = special_tokens
    
    # bytes -> tokenID
    self.reverse_vocab = {
      v : k
      for k, v in self.vocab.items()
    }
  
  @classmethod
  def from_files(cls,
                 vocab_filepath: str,
                 merges_filepath: str,
                 special_tokens: list[int] | None = None):
    with open(vocab_filepath, "r", encoding="utf-8") as f:
      raw_vocab = json.load(f)
  
    vocab = {
      int(k) : bytes.fromhex(v)
      for k, v in raw_vocab.items()
    }
    
    with open(merges_filepath, "r", encoding="utf-8") as f:
      merges = []
      for line in f:
        left_hex, right_hex = line.split()

        merges.append(
          (bytes.fromhex(left_hex),
            bytes.fromhex(right_hex))
        )
      
    return cls(vocab, merges, special_tokens)
  
  def pretokenize(self, text: str) -> list[tuple[bytes, ...]]:
    pretokens = []
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    if self.special_tokens:
      special_tokens_str = []
      
      for special_token in self.special_tokens:
        special_tokens_str.append(self.vocab[special_token].decode("utf-8"))
      
      
      special_pattern = "|".join([re.escape(special_token_str) for special_token_str in special_tokens_str])
      chunks = re.split(special_pattern, text)
    else:
      chunks = [text]
      
    for chunk in chunks:
      matches = re.finditer(PAT, chunk)
      
      for match in matches:
        pretokens.append(tuple(bytes([b]) for b in match.group(0).encode("utf-8")))
        
    return pretokens
  
  def encode(self, text: str) -> list[int]:
    # pretokens is a list rather than a dictionary
    pretokens = self.pretokenize(text)
    
    token_id_seq = []
    
    for pretoken in pretokens:
      n = len(pretoken)
      new_pretoken = [] 
      
      for merge in self.merges:
        new_symbol = merge[0] + merge[1]
        
        i = 0
        while i < n:
          if i + 1 < n and pretoken[i] == merge[0] and pretoken[i+1] == merge[1]:
            new_pretoken.append(new_symbol)
            i += 2
          else:
            new_pretoken.append(pretoken[i])
            i += 1
            
      for token in new_pretoken:
        token_id_seq.append(self.reverse_vocab[token])
        
    return token_id_seq
  
  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    raise NotImplementedError
  
  def decode(self, ids: list[int]) -> str:
    raise NotImplementedError