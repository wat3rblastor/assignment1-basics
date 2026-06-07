import regex as re
import time

from pathlib import Path

# Because of this relative import, you have to run this file from the root directory
from cs336_basics.tokenizer import Tokenizer


def part_a():
  ROOT = Path(__file__).resolve().parents[1]
  input_path = ROOT / "data" / "TinyStories-10.txt"
  vocab_path = ROOT / "tokenizer_output" / "tinystories_vocab.json"
  merges_path = ROOT / "tokenizer_output" / "tinystories_merge.txt"

  special_tokens = ["<|endoftext|>"]
  
  tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
  
  num_bytes = []
  num_token_ids = []
  
  with open(input_path, "rb") as f:
    data = f.read()
    text = data.decode("utf-8")
    
    special_pattern = "|".join([re.escape(special_token) for special_token in special_tokens])
    chunks = re.split(f"({special_pattern})", text)

    for chunk in chunks:
      token_ids = tokenizer.encode(chunk)
      
      num_bytes.append(len(chunk.encode("utf-8")))
      num_token_ids.append(len(token_ids))
      
  tokenizer_compression_ratio = []
      
  for num_byte, num_token_id in zip(num_bytes, num_token_ids):
    if num_token_id != 0:
      tokenizer_compression_ratio.append(num_byte / num_token_id)
      
  print("TinyStories:", tokenizer_compression_ratio)
  print(f"Average Compression Ratio: {sum(num_bytes) / sum(num_token_ids)}")
  
def part_b():
  ROOT = Path(__file__).resolve().parents[1]
  input_path = ROOT / "data" / "owt-10.txt"
  vocab_path = ROOT / "tokenizer_output" / "tinystories_vocab.json"
  merges_path = ROOT / "tokenizer_output" / "tinystories_merge.txt"

  special_tokens = ["<|endoftext|>"]
  
  tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
  
  num_bytes = []
  num_token_ids = []
  
  with open(input_path, "rb") as f:
    data = f.read()
    text = data.decode("utf-8")
    
    special_pattern = "|".join([re.escape(special_token) for special_token in special_tokens])
    chunks = re.split(f"({special_pattern})", text)

    for chunk in chunks:
      token_ids = tokenizer.encode(chunk)
      
      num_bytes.append(len(chunk.encode("utf-8")))
      num_token_ids.append(len(token_ids))
      
  tokenizer_compression_ratio = []
      
  for num_byte, num_token_id in zip(num_bytes, num_token_ids):
    if num_token_id != 0:
      tokenizer_compression_ratio.append(num_byte / num_token_id)
      
  print("OWT:", tokenizer_compression_ratio)
  print(f"Average OWT Compression Ratio: {sum(num_bytes) / sum(num_token_ids)}")
  
def part_c():
  ROOT = Path(__file__).resolve().parents[1]
  input_path = ROOT / "data" / "TinyStories-10.txt"
  vocab_path = ROOT / "tokenizer_output" / "tinystories_vocab.json"
  merges_path = ROOT / "tokenizer_output" / "tinystories_merge.txt"

  special_tokens = ["<|endoftext|>"]
  
  tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
  
  num_bytes = []
  
  with open(input_path, "rb") as f:
    data = f.read()
    text = data.decode("utf-8")
    
    special_pattern = "|".join([re.escape(special_token) for special_token in special_tokens])
    chunks = re.split(f"({special_pattern})", text)

  start_time = time.perf_counter()
  for chunk in chunks:
    token_ids = tokenizer.encode(chunk)
    num_bytes.append(len(chunk.encode("utf-8")))
  end_time = time.perf_counter()
    
  total_bytes = sum(num_bytes)
  time_elapsed = end_time - start_time
  
  print(f"Throughput: {total_bytes / time_elapsed: .2f} bytes/second")
  
  
def main():
  part_a()
  part_b()
  part_c()
  
if __name__ == "__main__":
  main()