# Resource counting 

vocab_size: 50,257
context_length: 1,024
num_layers: 48
d_model: 1,600
num_heads: 25
d_ff: 4,288 (the nearest multiple of 64 to 8
3 × 1, 600)
d_k = 64

Token embedding = vocab_size x d_model = 50257 * 1600 = 80411200

Per transformer = (x num_layers )

RMS norm = 2 x d_model = 3200
Attention (QKVO) = d_model x d_model = 4 * d_model^2 = 10240000 
SwigLu W1,W2, W3 = 3 x d_model * d_ff = 20582400

per_trans = 48 x (3200 + 10240000 + 20582400 ) = 1479628800

RMSNorm = d_model = 1600
LM Head = vocab_size * d_model = 80411200


total_param = 80411200 + 1479628800 + 1600 + 80411200 = 1640452800

total_memory = 1640452800 * 4 = 6561811200 / 1e9 = 6.56 GB


(b) Feed Forward FLOPS 

Per transfomer 
- Q, K, V projection = 3 * (seq * d_model) * (d_model* d_model) = 6 * (seq * d_model * d_model) = 15,728,640,000
- Attention score = (seq * d_k) * (d_k * seq) * num_heads = 2 * (seq * d_k * seq) * 25 = 3,355,443,200
- Attention x V = (seq * seq) * (seq x d_v) * num_heads = 25 * 2 * (seq * seq * d_v) =  3,355,443,200
- Output projection = (seq * d_model) * (d_model * d_model) = 2 * (seq * d_model * d_model) = 5,242,880,000
- SwiGLU FFN W1 W3  = 2 *(seq * d_model) * (d_model * d_ff)  = 4 * (seq * d_ff * d_model) +   2 * (seq * d_ff * d_model) = 42,152,755,200
-

total_layers = 48 * (69,835,161,600) = 3,352,087,756,800

LM Head = 2 * (seq * d_model * vocab_size) = 164,682,137,600

total = 3,516,769,894,400 = 3.52 T Flops

(c) Which part dominates
QKV proj: 15.73B
Attn scores: 3.36B
Attn × V: 3.36B
Output proj: 5.24B
SwiGLU FFN: 42.15B ← biggest
per block: 69.84B → ×48 = 3.352T
LM head: 164.7B

SwiGLU FFN (60%) > Attention Projection (30%) > attention score (10%) 

SwiGLU is 60.35% of total block.


# small 
================================================================
(a) PARAMETERS & MEMORY
================================================================
  Token embedding  (V*d)        = 38,597,376
  Per block:
    RMSNorm   (2*d_model)       = 1,536
    Attn QKVO (4*d_model^2)     = 2,359,296
    SwiGLU    (3*d_model*d_ff)  = 4,718,592
    -> per block                = 7,079,424
  All 12 blocks                 = 84,953,088
  Final RMSNorm  (d_model)      = 768
  LM head        (V*d)          = 38,597,376
  --------------------------------------------
  TOTAL PARAMETERS              = 162,148,608  (162.15 M)
  MEMORY (fp32)                 = 648,594,432 bytes  (0.65 GB)

================================================================
(b) FORWARD-PASS FLOPs   (seq = 1024)
================================================================
  Per block:
    Q,K,V projection       =      3,623,878,656   (3.62 B)
    Attention scores QK^T  =      1,610,612,736   (1.61 B)
    Attention (scores x V) =      1,610,612,736   (1.61 B)
    Output projection      =      1,207,959,552   (1.21 B)
    SwiGLU FFN             =      9,663,676,416   (9.66 B)
    -> per block           =     17,716,740,096   (17.72 B)
  All 12 blocks            =    212,600,881,152   (212.60 B)
  LM head                  =     79,047,426,048   (79.05 B)
  --------------------------------------------
  TOTAL FLOPs              =    291,648,307,200  (291.65 B)

================================================================
(c) WHAT DOMINATES A BLOCK
================================================================
  Q,K,V projection     3.62 B   20.45%
  Attention scores     1.61 B    9.09%
  Attention x V        1.61 B    9.09%
  Output proj          1.21 B    6.82%
  SwiGLU FFN           9.66 B   54.55%  <-- biggest

  Grouped:
    SwiGLU FFN              54.55%
    Attention projections   27.27%
    Attention scores        18.18%

  => SwiGLU FFN is 54.55% of every block, so it dominates.


  # medium 
================================================================
(a) PARAMETERS & MEMORY
================================================================
  Token embedding  (V*d)        = 51,463,168
  Per block:
    RMSNorm   (2*d_model)       = 2,048
    Attn QKVO (4*d_model^2)     = 4,194,304
    SwiGLU    (3*d_model*d_ff)  = 8,454,144
    -> per block                = 12,650,496
  All 24 blocks                 = 303,611,904
  Final RMSNorm  (d_model)      = 1,024
  LM head        (V*d)          = 51,463,168
  --------------------------------------------
  TOTAL PARAMETERS              = 406,539,264  (406.54 M)
  MEMORY (fp32)                 = 1,626,157,056 bytes  (1.63 GB)

================================================================
(b) FORWARD-PASS FLOPs   (seq = 1024)
================================================================
  Per block:
    Q,K,V projection       =      6,442,450,944   (6.44 B)
    Attention scores QK^T  =      2,147,483,648   (2.15 B)
    Attention (scores x V) =      2,147,483,648   (2.15 B)
    Output projection      =      2,147,483,648   (2.15 B)
    SwiGLU FFN             =     17,314,086,912   (17.31 B)
    -> per block           =     30,198,988,800   (30.20 B)
  All 24 blocks            =    724,775,731,200   (724.78 B)
  LM head                  =    105,396,568,064   (105.40 B)
  --------------------------------------------
  TOTAL FLOPs              =    830,172,299,264  (830.17 B)

================================================================
(c) WHAT DOMINATES A BLOCK
================================================================
  Q,K,V projection     6.44 B   21.33%
  Attention scores     2.15 B    7.11%
  Attention x V        2.15 B    7.11%
  Output proj          2.15 B    7.11%
  SwiGLU FFN          17.31 B   57.33%  <-- biggest

  Grouped:
    SwiGLU FFN              57.33%
    Attention projections   28.44%
    Attention scores        14.22%

  => SwiGLU FFN is 57.33% of every block, so it dominates.
# large
================================================================
(a) PARAMETERS & MEMORY
================================================================
  Token embedding  (V*d)        = 64,328,960
  Per block:
    RMSNorm   (2*d_model)       = 2,560
    Attn QKVO (4*d_model^2)     = 6,553,600
    SwiGLU    (3*d_model*d_ff)  = 13,025,280
    -> per block                = 19,581,440
  All 36 blocks                 = 704,931,840
  Final RMSNorm  (d_model)      = 1,280
  LM head        (V*d)          = 64,328,960
  --------------------------------------------
  TOTAL PARAMETERS              = 833,591,040  (833.59 M)
  MEMORY (fp32)                 = 3,334,364,160 bytes  (3.33 GB)

================================================================
(b) FORWARD-PASS FLOPs   (seq = 1024)
================================================================
  Per block:
    Q,K,V projection       =     10,066,329,600   (10.07 B)
    Attention scores QK^T  =      2,684,354,560   (2.68 B)
    Attention (scores x V) =      2,684,354,560   (2.68 B)
    Output projection      =      3,355,443,200   (3.36 B)
    SwiGLU FFN             =     26,675,773,440   (26.68 B)
    -> per block           =     45,466,255,360   (45.47 B)
  All 36 blocks            =  1,636,785,192,960   (1.64 T)
  LM head                  =    131,745,710,080   (131.75 B)
  --------------------------------------------
  TOTAL FLOPs              =  1,768,530,903,040  (1.77 T)

================================================================
(c) WHAT DOMINATES A BLOCK
================================================================
  Q,K,V projection    10.07 B   22.14%
  Attention scores     2.68 B    5.90%
  Attention x V        2.68 B    5.90%
  Output proj          3.36 B    7.38%
  SwiGLU FFN          26.68 B   58.67%  <-- biggest

  Grouped:
    SwiGLU FFN              58.67%
    Attention projections   29.52%
    Attention scores        11.81%
 => SwiGLU FFN is 58.67% of every block, so it dominates.
 ## x-large 
 ================================================================
(a) PARAMETERS & MEMORY
================================================================
  Token embedding  (V*d)        = 80,411,200
  Per block:
    RMSNorm   (2*d_model)       = 3,200
    Attn QKVO (4*d_model^2)     = 10,240,000
    SwiGLU    (3*d_model*d_ff)  = 20,582,400
    -> per block                = 30,825,600
  All 48 blocks                 = 1,479,628,800
  Final RMSNorm  (d_model)      = 1,600
  LM head        (V*d)          = 80,411,200
  --------------------------------------------
  TOTAL PARAMETERS              = 1,640,452,800  (1.64 B)
  MEMORY (fp32)                 = 6,561,811,200 bytes  (6.56 GB)

================================================================
(b) FORWARD-PASS FLOPs   (seq = 16348)
================================================================
  Per block:
    Q,K,V projection       =    251,105,280,000   (251.11 B)
    Attention scores QK^T  =    855,222,732,800   (855.22 B)
    Attention (scores x V) =    855,222,732,800   (855.22 B)
    Output projection      =     83,701,760,000   (83.70 B)
    SwiGLU FFN             =    672,962,150,400   (672.96 B)
    -> per block           =  2,718,214,656,000   (2.72 T)
  All 48 blocks            = 130,474,303,488,000   (130.47 T)
  LM head                  =  2,629,124,595,200   (2.63 T)
  --------------------------------------------
  TOTAL FLOPs              = 133,103,428,083,200  (133.10 T)

================================================================
(c) WHAT DOMINATES A BLOCK
================================================================
  Q,K,V projection   251.11 B    9.24%
  Attention scores   855.22 B   31.46%  <-- biggest
  Attention x V      855.22 B   31.46%
  Output proj         83.70 B    3.08%
  SwiGLU FFN         672.96 B   24.76%

  Grouped:
    SwiGLU FFN              24.76%
    Attention projections   12.32%
    Attention scores        62.93%

  => SwiGLU FFN is 24.76% of every block, so it dominates.

  # XL GPT 

  ================================================================
(a) PARAMETERS & MEMORY
================================================================
  Token embedding  (V*d)        = 80,411,200
  Per block:
    RMSNorm   (2*d_model)       = 3,200
    Attn QKVO (4*d_model^2)     = 10,240,000
    SwiGLU    (3*d_model*d_ff)  = 20,582,400
    -> per block                = 30,825,600
  All 48 blocks                 = 1,479,628,800
  Final RMSNorm  (d_model)      = 1,600
  LM head        (V*d)          = 80,411,200
  --------------------------------------------
  TOTAL PARAMETERS              = 1,640,452,800  (1.64 B)
  MEMORY (fp32)                 = 6,561,811,200 bytes  (6.56 GB)

================================================================
(b) FORWARD-PASS FLOPs   (seq = 16384)
================================================================
  Per block:
    Q,K,V projection       =    251,658,240,000   (251.66 B)
    Attention scores QK^T  =    858,993,459,200   (858.99 B)
    Attention (scores x V) =    858,993,459,200   (858.99 B)
    Output projection      =     83,886,080,000   (83.89 B)
    SwiGLU FFN             =    674,444,083,200   (674.44 B)
    -> per block           =  2,727,975,321,600   (2.73 T)
  All 48 blocks            = 130,942,815,436,800   (130.94 T)
  LM head                  =  2,634,914,201,600   (2.63 T)
  --------------------------------------------
  TOTAL FLOPs              = 133,577,729,638,400  (133.58 T)

================================================================
(c) WHAT DOMINATES A BLOCK
================================================================
  Q,K,V projection   251.66 B    9.23%
  Attention scores   858.99 B   31.49%  <-- biggest
  Attention x V      858.99 B   31.49%
  Output proj         83.89 B    3.08%
  SwiGLU FFN         674.44 B   24.72%

  Grouped:
    SwiGLU FFN              24.72%
    Attention projections   12.30%
    Attention scores        62.98%

  => SwiGLU FFN is 24.72% of every block, so it dominates.