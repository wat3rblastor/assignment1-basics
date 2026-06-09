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
    
    for pretoken in pretokens:
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
  

def find_chunk_boundaries(
  file: BinaryIO,
  chunk_size: int,
  split_special_token: bytes,
) -> list[int]:
  """
  Chunk the file into parts that can be counted independently.
  May return fewer chunks if the boundaries end up overlapping.
  """
  assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

  # Get total file size in bytes
  file.seek(0, os.SEEK_END)
  file_size = file.tell()
  file.seek(0)

  desired_num_chunks = max(1, (file_size + chunk_size - 1) // chunk_size)

  # Initial guesses for chunk boundary locations, uniformly spaced
  # Chunks start on previous index, don't include last index
  chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
  chunk_boundaries[-1] = file_size

  mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

  for bi in range(1, len(chunk_boundaries) - 1):
      initial_position = chunk_boundaries[bi]
      file.seek(initial_position)  # Start at boundary guess
      while True:
          mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

          # If EOF, this boundary should be at the end of the file
          if mini_chunk == b"":
              chunk_boundaries[bi] = file_size
              break

          # Find the special token in the mini chunk
          found_at = mini_chunk.find(split_special_token)
          if found_at != -1:
              chunk_boundaries[bi] = initial_position + found_at
              break
          initial_position += mini_chunk_size

  # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
  return sorted(set(chunk_boundaries))


def encode_file_to_bin(
  tokenizer: Tokenizer,
  input_path: Path,
  output_path: Path,
):
  total_bytes = input_path.stat().st_size
  output_path.parent.mkdir(parents=True, exist_ok=True)
  
  chunk_size = 2 ** 20
  with input_path.open("rb") as f:
    chunk_boundaries = find_chunk_boundaries(f, chunk_size, b"<|endoftext|>")
    
  with input_path.open("rb") as f, output_path.open("wb") as out:
    with tqdm(total=total_bytes, desc=f"Encoding", unit="B", unit_scale=True) as pbar:
      for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk = chunk_bytes.decode("utf-8")
        
        token_ids = tokenizer.encode(chunk)
        np_token_ids = np.array(token_ids, dtype=np.uint16)
        np_token_ids.tofile(out)
        
        pbar.update(len(chunk_bytes))
      
  
def main():
  ROOT = Path(__file__).resolve().parents[1]
  train_input_path = ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
  valid_input_path = ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
  
  train_output_path = ROOT / "data" / "TinyStories-TokenIDs-train.bin"
  valid_output_path = ROOT / "data" / "TinyStories-TokenIDs-valid.bin"
  
  vocab_path = ROOT / "tokenizer_output" / "tinystories_vocab.json"
  merges_path = ROOT / "tokenizer_output" / "tinystories_merge.txt"
  
  special_tokens = ["<|endoftext|>"]
  
  tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

  # This is assuming vocab_size < 2^16
  encode_file_to_bin(tokenizer, valid_input_path, valid_output_path)
  encode_file_to_bin(tokenizer, train_input_path, train_output_path)
    
  
if __name__ == "__main__":
  main()