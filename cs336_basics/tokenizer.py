import json
import regex as re
import os
import numpy as np

from typing import BinaryIO
from pathlib import Path
from collections.abc import Iterable, Iterator
from tqdm import tqdm

class Tokenizer:
  def __init__(self, vocab: dict[int, bytes], 
               merges: list[tuple[bytes, bytes]], 
               special_tokens: list[str] | None = None):
    # tokenID -> bytes
    self.vocab = vocab
    self.merges = merges
    
    self.merge_ranks = {
      pair: rank
      for rank, pair in enumerate(self.merges)
    }
    
    if special_tokens:
      # When encoding, group by larger special tokens first
      self.special_tokens = sorted(special_tokens, key=len, reverse=True)
      self.special_token_bytes = {s.encode("utf-8") for s in self.special_tokens}
    else:
      self.special_tokens = None
      self.special_token_bytes = None
    
    # bytes -> tokenID
    self.reverse_vocab = {
      v : k
      for k, v in self.vocab.items()
    }
  
  @classmethod
  def from_files(cls,
                 vocab_filepath: str | os.PathLike,
                 merges_filepath: str | os.PathLike,
                 special_tokens: list[str] | None = None):
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
      special_pattern = "|".join([re.escape(special_token) for special_token in self.special_tokens])
      text_and_special_tokens = re.split(f"({special_pattern})", text) # capture group to keep the special tokens
    else:
      text_and_special_tokens = [text]
      
    for text_or_special_token in text_and_special_tokens:
      if self.special_tokens and text_or_special_token in self.special_tokens:
        pretokens.append((text_or_special_token.encode("utf-8"), )) # acting as special_token
      
      else: 
        matches = re.finditer(PAT, text_or_special_token) # acting as text
        
        for match in matches:
          pretokens.append(tuple(bytes([b]) for b in match.group(0).encode("utf-8")))
        
    return pretokens
  
  def encode(self, text: str) -> list[int]:
    # pretokens is a list rather than a dictionary
    pretokens = self.pretokenize(text)
    
    token_id_seq = []
    
    for pretoken in tqdm(pretokens,
                         desc="Encoding",
                         unit="pretoken",
                         total=len(pretokens)):
      if (self.special_tokens 
          and self.special_token_bytes
          and len(pretoken) == 1
          and pretoken[0] in self.special_token_bytes):
        token_id_seq.append(self.reverse_vocab[pretoken[0]])
        continue
      
      cur_pretoken = pretoken
      
      while True:
        best_pair = None
        best_rank = float("inf")
        
        for pair in zip(cur_pretoken[:-1], cur_pretoken[1:]):
          rank = self.merge_ranks.get(pair)
          if rank is not None and rank < best_rank:
            best_pair = pair
            best_rank = rank
            
        if best_pair is None:
          break
        
        merge = best_pair
        new_symbol = merge[0] + merge[1]

        n = len(cur_pretoken)
        new_pretoken = []
        i = 0
        while i < n:
          if i + 1 < n and cur_pretoken[i] == merge[0] and cur_pretoken[i+1] == merge[1]:
            new_pretoken.append(new_symbol)
            i += 2
          else:
            new_pretoken.append(cur_pretoken[i])
            i += 1
            
        cur_pretoken = tuple(new_pretoken)
      
      for token in cur_pretoken:
        token_id_seq.append(self.reverse_vocab[token])
        
    return token_id_seq
  
  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    for text in iterable:
      yield from self.encode(text)
  
  def decode(self, ids: list[int]) -> str:
    byte_seq = b"".join(self.vocab[id] for id in ids)
    return byte_seq.decode("utf-8", errors="replace")
    
  
def main():
  ROOT = Path(__file__).resolve().parents[1]
  train_input_path = ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
  valid_input_path = ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
  
  train_output_path = ROOT / "tokenizer_output" / "TinyStories-TokenIDs-train.bin"
  valid_output_path = ROOT / "tokenizer_output" / "TinyStories-TokenIDs-valid.bin"
  
  vocab_path = ROOT / "tokenizer_output" / "tinystories_vocab.json"
  merges_path = ROOT / "tokenizer_output" / "tinystories_merge.txt"
  
  special_tokens = ["<|endoftext|>"]
  
  tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
  
  with valid_input_path.open("r", encoding="utf-8") as f:
    text_valid = f.read()
    
  # This is assuming vocab_size < 2^16
  # Which is fine since vocab size is 10000 for TinyStories and 32000 for OWT
  token_ids = tokenizer.encode(text_valid)
  np_token_ids = np.array(token_ids, dtype=np.uint16)
  np_token_ids.tofile(valid_output_path)
  
  with train_input_path.open("r", encoding="utf-8") as f:
    text_train = f.read()  

  token_ids = tokenizer.encode(text_train)
  np_token_ids = np.array(token_ids, dtype=np.uint16)
  np_token_ids.tofile(train_output_path)
    
  
if __name__ == "__main__":
  main()