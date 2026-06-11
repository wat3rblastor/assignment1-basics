### Question unicode1
(a) It returns '\x00'.

(b) The character's string representation is more for programmers while its printed representation is for everyone.

(c) For the string representation, it displays as '\x00'. For its printed representation, it does not appear, as chr(0) references the null character.

### Question unicode2
(a) UTF-8 encoding represents unicode characters ranging from 1-4 bytes, with the most common characters being represented as 1 byte. UTF-16 encoding represents unicode characters ranging from 2-4 bytes. UTF-32 characters represent UTF-32 characters as 4 bytes. It is preferable to training our tokenizer on UTF-8 encoded bytes because on average, it will be shorter in length. This is because the more common characters are more likely to appear in the training text, and since more common unicode characters are only represented with 1 byte, the UTF-8 encoded byte representation will be much shorter than the UTF-16 or UTF-32 representation.

(b) This function is incorrect because characters may be represented by multiple bytes in UTF-8 encoding, meaning if you decode assuming each byte maps cleanly to one character, you will encounter an error. An example input string for which decode_utf8_bytes_to_str_wrong produces incorrect output is "π". 

(b) bytes([192, 128]) does not decode to any Unicode character(s). This is because not every 2 byte sequence maps to a valid Unicode character.

### Question train_bpe_tinystories
(a) It took around 2 minutes and 33.4 GB of memory to train. This is just from a quick glance, but the longest possible token is possibly 0x206772616e646461756768746572. I don't know what this represents, so I don't know if this makes sense. Remember, we did byte-pair encoding, not character-pair encoding.

(b) The pretokenization takes the most time. 

### Question tokenizer_experiments
(a) The tokenizer's compression ratio for the TinyStories tokenizer on the first 10 documents of TinyStories is 4.163 (bytes / token). I did not train the tokenizer for OpenWebText because I didn't want to train it for 12 hours.

(b) If I tokenize my OWT sample with the TinyStories tokenizer, I get a compression ratio of 3.199 (bytes / token). Conceptually, this tokenizer did not merge the most frequent byte pairs in the OWT text. Therefore, the most frequent bytes are likely represented with more tokens, and therefore we have a worse off compression ratio.

(c) The throughput of my tokenizer is around 1103240.34 bytes/second. It would take (825 * 10^9) / 1103240.34 = 747797.1663 seconds to tokenize the entire Pile dataset, or around 8.66 days. :O

### Question transformer_accounting
(a)
#### Linear Layer
Contains vocab_size * d_model parameters, or 50,257 * 1,600 = 80,411,200 parameters.

#### Embedding Layer
Contains vocab_size * d_model parameters, or 80,411,200 parameters.

#### RMSNorm
Contains d_model parameters, or 1,600 parameters.

#### MultiHead Attention
Contains self.hd_k * d_model + self.hd_k * d_model + self.hd_v * d_model + d_model * self.hd_v parameters. Since self.hd_v = self.hd_k = d_model, contains 4 * d_model^2 = 10,240,000 parameters.

#### FFN
Contains 3 * d_ff * d_model parameters, or 3 * 4288 * 1,600 = 20,582,400 parameters.

#### Transformer Block
Contains 2 RMSNorm modules, one MultiHeadAttention Module, one FFN module. Therefore, contains 2 * 1,600 + 10,240,000 + 20,582,400 = 30,8256,00 parameters.

#### TransformerLM
Contains one embedding module, 48 transformer modules, one RMS module, and one linear module. Therefore, contains 80,411,200 + 48 * 30,8256,00 + 1,600 + 80,411,200 = 1,640,452,800 parameters

Assuming single-precision floating point, it will take 6.56 GB to load the entire model.

(b) Listed below are the number of FLOPs required for each component. We will only consider FLOPs from matmuls.

#### Embedding Layer
0 FLOPs, no matmuls

#### RMSNorm
0 FLOPs, no matmuls

#### Attention
Q.shape = (batch_size, num_heads, seq_len, d_k)
K.shape = (batch_size, num_heads, seq_len, d_k)
V.shape = (batch_size, num_heads, seq_len, d_v)

Q @ K.transpose(-2, -1) = 2 * batch_size * num_heads * seq_len * d_k * seq_len FLOPs

value_weights.shape = (batch_size, num_heads, seq_len, seq_len)
value_weights @ V = 2 * batch_size * num_heads * seq_len * seq_len * d_v FLOPs

Total FLOPs = 2 * batch_size * num_heads * seq_len^2 * (d_k + d_v)

Total FLOPs = 2 * 1 * 1024 * 1024 * 2 * 1600 = 6.71 GLOPs

#### MultiHead Attention
x.shape = (batch_size, seq_len, d_model)
self.W_Q.shape = (self.hd_k, d_model)
self.W_K.shape = (self.hd_k, d_model)
self.W_V.shape = (self.hd_v, d_model)
self.W_O.shape = (d_model, self.hd_v)

x @ self.W_Q.T = 2 * batch_size * seq_len * d_model * self.hd_k FLOPs
x @ self.W_K.T = 2 * batch_size * seq_len * d_model * self.hd_k FLOPs
x @ self.W_V.T = 2 * batch_size * seq_len * d_model * self.hd_v FLOPs

Q.shape = (batch_size, num_heads, seq_len, d_k)
K.shape = (batch_size, num_heads, seq_len, d_k)
V.shape = (batch_size, num_heads, seq_len, d_v)

Notice self.hd_k = num_heads * d_k. Similarly, self.hd_v = num_heads * d_v

attention_vals = scaled_dot_product_attention(Q, K, V, mask) = 2 * batch_size * num_heads * seq_len^2 * (d_k + d_v) FLOPs

After the reshaping, attention_vals.shape = (batch_size, seq_len, self.hd_v)
attention_vals @ self.W_O.T = 2 * batch_size * seq_len * self.hd_v * d_model FLOPs

Total FLOPs = 2 * batch_size * seq_len * (2 * d_model * self.hd_k + 2 * d_model * self.hd_v + seq_len * num_heads * (d_k + d_v))

Total FLOPs = 2 * 1 * 1024 * (2 * 1600 * 1600 + 2 * 1600 * 1600 + 1024 * 2 * 1600) = 27.68 GFLOPs

#### FFN
x.shape = (batch_size, seq_len, d_model)
self.w1.shape = (d_ff, d_model)
self.w2.shape = (d_model, d_ff)
self.w3.shape = (d_ff, d_model)

x @ self.w1.T = 2 * batch_size * seq_len * d_model * d_ff FLOPs
x @ self.w3.T = 2 * batch_size * seq_len * d_model * d_ff FLOPs

silu_x.shape = (batch_size, seq_len, d_ff)
w3_x.shape = (batch_size, seq_len, d_ff)

(silu_x * w3_x) @ self.w2.T = 2 * batch_size * seq_len * d_ff * d_model FLOPs

Total FLOPs = 6 * batch_size * seq_len * d_model * d_ff

Total FLOPs = 6 * 1 * 1024 * 1600 * 4288 = 42.15 GFLOPs

#### TransformerBlock
x.shape = (batch_size, seq_len, d_model)

Total FLOPs = 2 * batch_size * seq_len * (2 * d_model * self.hd_k + 2 * d_model * self.hd_v + seq_len * num_heads * (d_k + d_v)) 
            + 6 * batch_size * seq_len * d_model * d_ff 

Since self.hd_k = self.hd_v = d_model and d_k = d_v = d_model / num_heads

Total FLOPs = 2 * batch_size * seq_len * (4 * d_model * d_model + 2 * seq_len * d_model)
            + 6 * batch_size * seq_len * d_model * d_ff

Total FLOPs = 2 * 1 * 1024 * (4 * 1600 * 1600 + 2 * 1024 * 1600) + 6 * 1 * 1024 * 1600 * 4288 = 69.8 GFLOPs

#### TransformerLM
TransformerBlocks = num_layers * (TransformerBlockTotalFLOPs) FLOPs

x = (batch_size, seq_len, d_model)
self.linear.w.shape = (vocab_size, d_model)

x = self.linear(x) = 2 * batch_size * seq_len * d_model * vocab_size FLOPs

TotalFLOPs = num_layers * (TransformerBlockTotalFLOPs) + 2 * batch_size * seq_len * d_model * vocab_size FLOPs

= num_layers * (2 * batch_size * seq_len * (4 * d_model * d_model + 2 * seq_len * d_model) + 6 * batch_size * seq_len * d_model * d_ff) + 2 * batch_size * seq_len * d_model * vocab_size FLOPs

= 2 * batch_size * seq_len * (num_layers * (4 * d_model^2 + 2 * seq_len * d_model + 3 * d_model * d_ff) + d_model * vocab_size) FLOPs

In total (assuming batch_size=1), it will take 2 * 1 * 1,025 * (48 * (4 * 1,600^2 + 2 * 1,024 * 1,600 + 3 * 1,600 * 4,288) + 1,600 * 50,257) = 3.52 TFLOPs.

(c) The Transformer blocks use up the most FLOPs, using 3.35 TFLOPs. Within the Transformer block, the FFN uses up the most FLOPs, using up 60% of the FLOPs, while Attention uses up the remaining 40% of FLOPs.

(d) I got tired of doing the calculations manually and created a script (which I candidly should've done for parts b and c...). Run the experiments/resource_accounting.py to get the FLOP counts.

As the model size increases, the scaled_dot_product_attention takes up a lower percentage of the FLOPs in the MultiHeadAttention module, the FFN takes up a higher percentage of the FLOPs in the TransformerBlock, and the Transformer takes up a higher pecentage of the FLOPs in the LM.

(e) When you increase the context length, the total FLOPs become 133,577,729,638,400 = 133.5 TFLOPs. The attention percentage in MultiHeadAttention increases by a lot, the FNN percentage percentage in the Transformer Block decreases, and the Transfoerm Block Percentage increases. This shows that when you increase context length, you mainly pay the cost through attention.