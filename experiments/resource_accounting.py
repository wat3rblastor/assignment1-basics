def calculate_flops(
  vocab_size,
  batch_size,
  context_length,
  num_layers,
  d_model,
  num_heads,
  d_ff
):
  attention_flops = 2 * batch_size * context_length * context_length * 2 * d_model
  multihead_attention_flops = 2 * batch_size * context_length * (2 * d_model * d_model + 2 * d_model * d_model + context_length * 2 * d_model)
  ffn_flops = 6 * batch_size * context_length * d_model * d_ff
  transformer_block_flops = multihead_attention_flops + ffn_flops
  final_linear_flops = 2 * batch_size * context_length * d_model * vocab_size
  transformer_lm_flops = num_layers * transformer_block_flops + final_linear_flops
  
  print(f"Attention FLOPs: {attention_flops}")
  print(f"MultiHeadAttention FLOPs: {multihead_attention_flops}")
  print(f"FFN FLOPs: {ffn_flops}")
  print(f"Transformer Block FLOPs: {transformer_block_flops}")
  print(f"Final Linear FLOPs: {final_linear_flops}")
  print(f"Transformer LM FLOPs: {transformer_lm_flops}")
  
  print(f"Attention Percentage in MultiHeadAttention: {attention_flops / multihead_attention_flops}")
  print(f"MultiHeadAttention Percentage in Transformer Block: {multihead_attention_flops / transformer_block_flops}")
  print(f"FFN Percentage in Transformer Block: {ffn_flops / transformer_block_flops}")
  print(f"Transformer Block Percentage in LM: {num_layers * transformer_block_flops / transformer_lm_flops}")


def calculate_gpt2_xl_flops():
  vocab_size = 50257
  batch_size = 1
  context_length = 1024
  num_layers = 48
  d_model = 1600
  num_heads = 25
  d_ff = 4288
  print("Calculating GPT2-XL FLOPs")
  calculate_flops(vocab_size, batch_size, context_length, num_layers, d_model, num_heads, d_ff)
  
  
def calculate_gpt2_small_flops():
  vocab_size = 50257
  batch_size = 1
  context_length = 1024
  num_layers = 12
  d_model = 768
  num_heads = 12
  d_ff = 2048
  print("Calculating GPT2-small FLOPs")
  calculate_flops(vocab_size, batch_size, context_length, num_layers, d_model, num_heads, d_ff)
  
  
def calculate_gpt2_medium_flops():
  vocab_size = 50257
  batch_size = 1
  context_length = 1024
  num_layers = 24
  d_model = 1024
  num_heads = 16
  d_ff = 2688
  print("Calculating GPT2-medium FLOPs")
  calculate_flops(vocab_size, batch_size, context_length, num_layers, d_model, num_heads, d_ff)
  
  
def calculate_gpt2_large_flops():
  vocab_size = 50257
  batch_size = 1
  context_length = 1024
  num_layers = 36
  d_model = 1280
  num_heads = 20
  d_ff = 3392
  print("Calculating GPT2-medium FLOPs")
  calculate_flops(vocab_size, batch_size, context_length, num_layers, d_model, num_heads, d_ff)
  

def calculate_gpt2_xl_increased_context_flops():
  vocab_size = 50257
  batch_size = 1
  context_length = 16384
  num_layers = 48
  d_model = 1600
  num_heads = 25
  d_ff = 4288
  print("Calculating GPT2-XL Increased Context FLOPs")
  calculate_flops(vocab_size, batch_size, context_length, num_layers, d_model, num_heads, d_ff)

  
def main():
  calculate_gpt2_small_flops()
  print()
  calculate_gpt2_medium_flops()
  print()
  calculate_gpt2_large_flops()
  print()
  calculate_gpt2_xl_flops()
  print()
  calculate_gpt2_xl_increased_context_flops()
  
  
if __name__ == "__main__":
  main()