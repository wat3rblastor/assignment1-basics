

# First, let's implemented without assuming chunking

def find_chunk_boundaries(input_path: str) -> list[int]:
  raise NotImplementedError


def initialize_vocabulary() -> dict[int, bytes]:
  raise NotImplementedError


def pretokenize(input_path: str, chunk_boundaries: list[int]
) -> dict[tuple[bytes, ...], int]:
  raise NotImplementedError


def merge(pretokens: dict[tuple[bytes, ...], int]) -> tuple[bytes, bytes]:
  raise NotImplementedError


def run_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges."""
    
    chunk_boundaries = find_chunk_boundaries(input_path)
    
    vocabulary = initialize_vocabulary()
    cur_size = len(vocabulary)
    
    merges = []
    
    pretokens = pretokenize(input_path, chunk_boundaries)
    
    for _ in range(vocab_size - cur_size):
      merges.append(merge(pretokens))
    
    return vocabulary, merges
  
  
  