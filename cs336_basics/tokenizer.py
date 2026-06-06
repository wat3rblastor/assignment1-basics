import os
import regex as re
import json
import cProfile

from pathlib import Path
from multiprocessing import Pool
from typing import BinaryIO
from collections import defaultdict
from tqdm import tqdm


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


def pretokenize_chunk(args) -> dict[tuple[bytes, ...], int]:
  input_path, start, end, special_tokens = args
  
  pretokens = defaultdict(int)
  
  PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  
  # Serial implementation
  with open(input_path, "rb") as f:
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
        pretokens[tuple(bytes([b]) for b in match.group(0).encode("utf-8"))] += 1
          
  return pretokens


def pretokenize(input_path: str | os.PathLike, chunk_boundaries: list[int], special_tokens: list[str], num_processes: int
) -> dict[tuple[bytes, ...], int]:
  pretokens = defaultdict(int)
  
  args = [
    (input_path, start, end, special_tokens)
   for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
  ]
  
  with Pool(num_processes) as pool:
    results = tqdm(
      pool.imap(pretokenize_chunk, args),
      total=len(args),
      desc="Pretokenizing",
    )
    
  for local_pretokens in results:
    for pretoken, freq in local_pretokens.items():
      pretokens[pretoken] += freq

  return pretokens


def get_stats(pretokens: dict[tuple[bytes, ...], int]
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
  pair_counts = defaultdict(int)
  pair_to_pretokens = defaultdict(set)
  
  for pretoken, freq in pretokens.items():
    for i in range(len(pretoken) - 1):
      pair = (pretoken[i], pretoken[i+1])
      
      pair_counts[pair] += freq
      pair_to_pretokens[pair].add(tuple(pretoken))
      
  return pair_counts, pair_to_pretokens


def merge(pretokens: dict[tuple[bytes, ...], int], vocabulary: dict[int, bytes], init_vocab_size: int, vocab_size: int
) -> list[tuple[bytes, bytes]]:  
  merges = []
  pair_counts, pair_to_pretokens = get_stats(pretokens)
  
  for cur_vocab_idx in tqdm(range(init_vocab_size, vocab_size), desc="Merging"):
    pretoken_to_new_pretoken = {}
    
    # Get most frequent pair  
    max_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
    
    # Add pair to merges
    merges.append(max_pair)
    
    # Add to vocabulary
    new_symbol = max_pair[0] + max_pair[1]
    vocabulary[cur_vocab_idx] = new_symbol
    
    # Update pretokens, pair_counts, pair_to_pretokens
    for pretoken in list(pair_to_pretokens[max_pair]):
      if pretoken not in pretokens:
        continue
      
      n = len(pretoken)

      # Create new_pretoken
      new_pretoken_lst = []      
      i = 0
      while i < n:
        if i < n - 1 and pretoken[i] == max_pair[0] and pretoken[i+1] == max_pair[1]:
          new_pretoken_lst.append(new_symbol)
          i += 2
        else:
          new_pretoken_lst.append(pretoken[i])
          i += 1
          
      new_pretoken = tuple(new_pretoken_lst)
      
      # Update pretokens
      freq = pretokens[pretoken]
      pretoken_to_new_pretoken[pretoken] = new_pretoken
      
      # Update pair_counts
      for pair in zip(pretoken[:-1], pretoken[1:]):
        if pair in pair_counts:
          pair_counts[pair] -= freq
          if pair_counts[pair] == 0:
            del pair_counts[pair]
        
      for pair in zip(new_pretoken[:-1], new_pretoken[1:]):
        pair_counts[pair] += freq
      
      # Update pair_to_pretokens
      # This will have stale entries, just will have to deal with it
      for pair in zip(new_pretoken[:-1], new_pretoken[1:]):
        pair_to_pretokens[pair].add(new_pretoken)
          
    pair_to_pretokens.pop(max_pair, None)
    
    # Update pretokens
    for pretoken, new_pretoken in pretoken_to_new_pretoken.items():
      pretokens[new_pretoken] += pretokens[pretoken]
      del pretokens[pretoken]

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

    num_processes = max(((os.cpu_count() or 1) - 1) // 2, 1)
    pretokens = pretokenize(input_path, chunk_boundaries, special_tokens, num_processes)
  
    merges = merge(pretokens, vocabulary, init_vocab_size, vocab_size)
    
    return vocabulary, merges
  
  
def main():
  ROOT = Path(__file__).resolve().parents[1]

  input_path = ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
  vocab_path = ROOT / "tinystories_vocab.json"
  merges_path = ROOT / "tinystories_merge.txt"

  special_tokens = ["<|endoftext|>"]
  vocabulary, merges = run_train_bpe(input_path, 10000, special_tokens)

  
  with vocab_path.open("w", encoding="utf-8") as f:
    json.dump({k: v.hex() for k, v in vocabulary.items()}, f, indent=2)
    
  with merges_path.open("w", encoding="utf-8") as f:
    for left, right in merges:
      f.write(f"{left} {right}\n")
      
  
if __name__ == "__main__":
  cProfile.run("main()", "tokenizer.prof")