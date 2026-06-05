import os
import regex as re

from typing import BinaryIO
from collections import defaultdict


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
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

    chunk_size = file_size // desired_num_chunks

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
  

def call_find_chunk_boundaries(input_path: str | os.PathLike) -> list[int]:
  with open(input_path, "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
  
  return boundaries
    

def initialize_vocabulary(special_tokens: list[str]) -> dict[int, bytes]:
  vocabulary = {}
  
  for i in range(len(special_tokens)):
    vocabulary[i] = special_tokens[i].encode("utf-8")
    
  num_special_tokens = len(special_tokens)
  
  for i in range(256):
    vocabulary[i + num_special_tokens] = bytes([i])
    
  return vocabulary


def pretokenize(input_path: str | os.PathLike, chunk_boundaries: list[int], special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
  pretokens = defaultdict(int)
  
  PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  
  # Serial implementation
  with open(input_path, "rb") as f:
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
      f.seek(start)
      raw_chunk = f.read(end - start).decode("utf-8", errors="ignore") # this is a string
      
      # Remove special tokens
      if special_tokens:
        special_pattern = "|".join([re.escape(special_token) for special_token in special_tokens])
        chunks = re.split(special_pattern, raw_chunk)
      else:
        chunks = [raw_chunk]
      
      for chunk in chunks:
        matches = re.finditer(PAT, chunk)
        
        for match in matches:
          # I HAVE TO CHANGE PRETOKENS TO BYTES
          pretokens[tuple(bytes([b]) for b in match.group(0).encode("utf-8"))] += 1
          
  return pretokens


def get_stats(pretokens: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], int]:
  pairs = defaultdict(int)
  
  for pretoken, freq in pretokens.items():
    for i in range(len(pretoken) - 1):
      pairs[pretoken[i], pretoken[i+1]] += freq
      
  return pairs


# Naive merge (for now)
def merge(pretokens: dict[tuple[bytes, ...], int], vocabulary: dict[int, bytes], init_vocab_size: int, vocab_size: int
) -> list[tuple[bytes, bytes]]:  
  merges = []
  
  for cur_vocab_idx in range(init_vocab_size, vocab_size):
    # Get most frequent pair
    pairs = get_stats(pretokens)    
    max_pair = max(pairs, key=lambda p: (pairs[p], p))
    
    # Add pair to merges
    merges.append(max_pair)
    
    # Add to vocabulary
    new_symbol = max_pair[0] + max_pair[1]
    vocabulary[cur_vocab_idx] = new_symbol
    
    # Merge pair for pretokens
    new_pretokens = defaultdict(int)
    for pretoken, freq in pretokens.items():
      new_pretoken = []
      n = len(pretoken)
      
      i = 0
      while i < n:
        if i < n - 1 and pretoken[i] == max_pair[0] and pretoken[i+1] == max_pair[1]:
          new_pretoken.append(new_symbol)
          i += 2
        else:
          new_pretoken.append(pretoken[i])
          i += 1
          
      new_pretokens[tuple(new_pretoken)] += freq
      
    pretokens = new_pretokens

  return merges


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges."""
    
    chunk_boundaries = call_find_chunk_boundaries(input_path)
    
    vocabulary = initialize_vocabulary(special_tokens)
    init_vocab_size = len(vocabulary)  

    pretokens = pretokenize(input_path, chunk_boundaries, special_tokens)
  
    merges = merge(pretokens, vocabulary, init_vocab_size, vocab_size)
    
    return vocabulary, merges