# 🧠 Step 2 — LTX-2 Foundation Model: Every Gear Inside the Machine

> **Series:** JUST-DUB-IT Deep Dive | **File:** 02 of 10
> **Primary Code:** `packages/ltx-core/src/ltx_core/model/transformer/`
> **All files read:** `model.py`, `transformer.py`, `attention.py`, `adaln.py`, `rope.py`,
> `feed_forward.py`, `modality.py`, `transformer_args.py`, `text_projection.py`,
> `timestep_embedding.py`, `gelu_approx.py`, `utils.py`

---

## 🎯 What You Will Learn in This Step

By the end of this step you will understand:

- What LTX-2 is and how it fits into the overall system
- The **Asymmetric Dual-Stream Architecture** — why two parallel transformers with different sizes
- **Timestep Embeddings** — how the model knows how noisy the current data is (with math)
- **AdaLN-Single** — adaptive normalisation and why gates matter
- **RoPE** — rotary position embeddings from scratch, including 3D video vs 1D audio
- **Attention** — Q-K-V mechanics, multi-head attention, masking, all three types used here
- **Bidirectional Cross-Attention** — the synchronisation bridge between audio and video
- **Feed-Forward Networks** — the non-linear "thinking" layer
- **Text Projection** — how Gemma output bridges dimension gaps
- **Complete data flow** traced from raw input to predicted velocity

---

## 🪐 2.0 — LTX-2 at a Glance

**LTX-2** is a **19-billion-parameter** audio-video foundation model built by **Lightricks** (January
2026). It generates synchronised video + audio simultaneously from a text prompt.

| Property | Value |
|----------|-------|
| Total parameters | ~19 billion |
| Video stream share | ~14 billion |
| Audio stream share | ~5 billion |
| Architecture | Diffusion Transformer (DiT) |
| Training paradigm | Flow Matching (Rectified Flow) |
| Max output duration | ~20 seconds |
| Text encoder | Gemma 3 (Google, 12B, quantised) |

The model is structured as an **asymmetric dual-stream** DiT. "Dual-stream" means video tokens
and audio tokens each travel through their own sequence of operations. "Asymmetric" means the
two streams are **unequal in size** — the video stream is twice as wide as the audio stream,
because video contains far more information.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        LTX-2 Overall Architecture                      │
│                                                                        │
│  Text Prompt ───▶ [Gemma 3 Text Encoder] ─────────────────────┐       │
│                                                                 │       │
│  Noisy Video ───▶ [Video VAE Encoder] ─▶ Video Latents (128D)  │       │
│                                                │                │       │
│  Noisy Audio ───▶ [Audio VAE Encoder] ─▶ Audio Latents (128D)  │       │
│                                                │                │       │
│                                         ┌──────▼────────┐       │       │
│                                         │  LTXModel     │◀──────┘       │
│                                         │  Transformer  │               │
│                                         │  (48 blocks)  │               │
│                                         └──────┬────────┘               │
│                                                │                        │
│  Clean Video ◀── [Video VAE Decoder] ◀── Video Velocities               │
│  Clean Audio ◀── [Audio VAE Decoder] ◀── Audio Velocities               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ 2.1 — The `LTXModel` Class: Top-Level Configuration

**File:** `model.py`, class `LTXModel` (line 32)

The constructor accepts all architectural hyperparameters:

```python
class LTXModel(torch.nn.Module):
    def __init__(self, *,
        # ─── MODEL TYPE ──────────────────────────────────────────────
        model_type: LTXModelType = LTXModelType.AudioVideo,
        #   Options: AudioVideo | VideoOnly | AudioOnly
        #   JustDubit always uses AudioVideo

        # ─── VIDEO STREAM ─────────────────────────────────────────────
        num_attention_heads: int = 32,        # 32 heads in video attention
        attention_head_dim: int = 128,        # 128 dims per head
        # ▲▲▲ inner_dim = 32 × 128 = 4,096 ▲▲▲

        in_channels: int = 128,               # Input latent channels
        out_channels: int = 128,              # Output latent channels (= velocity dims)
        num_layers: int = 48,                 # 48 transformer blocks stacked
        cross_attention_dim: int = 4096,      # Text context dimension for video
        caption_channels: int = 3840,         # Gemma output dimension
        positional_embedding_theta: float = 10000.0,   # RoPE base frequency
        positional_embedding_max_pos: list = [20, 2048, 2048],  # [T, H, W] maxes

        # ─── AUDIO STREAM ─────────────────────────────────────────────
        audio_num_attention_heads: int = 32,  # Same 32 heads
        audio_attention_head_dim: int = 64,   # BUT only 64 dims per head!
        # ▲▲▲ audio_inner_dim = 32 × 64 = 2,048 ▲▲▲ (HALF of video!)

        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list = [20],  # [T] only

        # ─── CROSS-MODAL CONTROLS ─────────────────────────────────────
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
    ):
```

### Why the Asymmetry?

```
VIDEO inner dimension:  32 × 128 = 4,096   ◀── "Wide river"
AUDIO inner dimension:  32 × 64  = 2,048   ◀── "Narrow river"
                                     ▲
                               HALF the size!
```

**Intuition:** Consider what each modality actually contains:

- A single video frame at 720p has `1280 × 720 × 3 = 2,764,800 values`
- A single second of CD-quality audio has `44,100 values`
- **The ratio is roughly 60:1 in raw data density**

After VAE compression, this ratio reduces dramatically, but video still contains far more
information per time step than audio. Allocating twice the model capacity to video is the
appropriate engineering trade-off.

---

## 🕐 2.2 — Timestep Embedding: Teaching the Model "How Noisy Am I?"

**File:** `timestep_embedding.py`

### The Problem

The model starts with pure noise and removes it gradually over ~30 denoising steps. At step 1,
the data is nearly 100% noise. At step 30, it is nearly 100% clean. The model must behave very
differently at each step:

- **Step 1 (very noisy):** Make coarse, global decisions. "This is a face. Put it on the left."
- **Step 30 (nearly clean):** Make tiny refinements. "This pixel should be slightly darker."

Without knowing the current noise level, the model would attempt the same operation at every
step and produce garbage.

### Solution: Sinusoidal Embedding

Function: `get_timestep_embedding()` (line 6):

```python
def get_timestep_embedding(timesteps, embedding_dim=256, max_period=10000):
    half_dim = embedding_dim // 2                               # 128

    # Create 128 exponentially-spaced frequencies
    # freq[i] = exp(-log(10000) × i / 127)  for i in [0, 127]
    # freq[0]  = 1.0      (fastest oscillation)
    # freq[63] = 0.01
    # freq[127] = 0.0001  (slowest oscillation)
    exponent = -math.log(max_period) * torch.arange(0, half_dim) / (half_dim - 1)
    emb = torch.exp(exponent)                                   # shape: [128]

    # Multiply timestep by each frequency
    emb = timesteps[:, None].float() * emb[None, :]    # [batch, 128]

    # Apply both sin and cos → unique fingerprint for every timestep
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [batch, 256]
    return emb
```

### Worked Example

Let `embedding_dim = 8` (for brevity) and `timestep σ = 0.30`:

```
Step 1: Half dimension = 4 frequencies:
    freq = [1.000, 0.100, 0.010, 0.001]   (exponential decrease)

Step 2: Multiply by σ = 0.30:
    angles = [0.300, 0.030, 0.003, 0.0003]

Step 3: Apply sin and cos:
    sin_part = [sin(0.300), sin(0.030), sin(0.003), sin(0.0003)]
             = [  0.2955,     0.0300,     0.0030,     0.0003  ]

    cos_part = [cos(0.300), cos(0.030), cos(0.003), cos(0.0003)]
             = [  0.9553,     0.9996,     1.0000,     1.0000  ]

Final 8D embedding = [0.2955, 0.0300, 0.0030, 0.0003, 0.9553, 0.9996, 1.0000, 1.0000]
                      ←────── sin half ──────────────────────────────────────────────→
                                                        ←──── cos half ───────────────→
```

**Why sin AND cos?** Because two different timesteps produce distinguishable, orthogonal patterns.
The low-frequency dimensions (near index 127) change slowly → they encode "are we early or late?"
The high-frequency dimensions (near index 0) change rapidly → they encode fine temporal detail.

### The Full Embedding Pipeline

The raw 256D sinusoidal vector is not fed directly to the transformer. It passes through a full
processing pipeline:

```
σ (scalar noise level, e.g. 0.30)
    │
    ▼  ─────────────────────────────────────────────────
    sin/cos encoding                              [256D]
    ─────────────────────────────────────────────────────
    │
    ▼  ─────────────────────────────────────────────────
    TimestepEmbedding
      Linear(256 → 4096)
      SiLU activation       ← smooth gating (not hard clip like ReLU)
      Linear(4096 → 4096)
    ─────────────────────────────────────────────────────  [4096D]
    │
    ▼  ─────────────────────────────────────────────────
    AdaLayerNormSingle
      SiLU
      Linear(4096 → 4096 × 6 = 24,576)
    ─────────────────────────────────────────────────────
    │
    Split into 6 chunks (each 4096D)
    │
    ├──▶ shift_msa   ← shift before self-attention
    ├──▶ scale_msa   ← scale before self-attention
    ├──▶ gate_msa    ← gate after  self-attention
    ├──▶ shift_mlp   ← shift before feed-forward
    ├──▶ scale_mlp   ← scale before feed-forward
    └──▶ gate_mlp    ← gate after  feed-forward
```

**The `AdaLayerNormSingle` linear layer** (from `adaln.py` line 28):
```python
self.linear = torch.nn.Linear(embedding_dim, 6 * embedding_dim)
# Input:  4096D vector (processed timestep embedding)
# Output: 24,576D vector → split into 6 × 4096D chunks
```

For the audio stream, there is a **separate** `audio_adaln_single` that produces 6 × 2048D chunks,
because the audio inner dimension is 2048, not 4096.

---

## 📐 2.3 — AdaLN-Single: Why Scale, Shift, AND Gate?

**File:** `adaln.py` | Used in: `transformer.py` (lines 164-170, 249-255)

The three modulation operations are applied at each sub-layer:

```
Formula applied BEFORE self-attention (transformer.py lines 168-170):

norm_vx = RMSNorm(vx)
norm_vx = norm_vx × (1 + scale_msa) + shift_msa    ← MODULATE
vx      = vx + attention(norm_vx, pe=...) × gate_msa  ← GATE
```

Let us understand each operation's role:

### Scale (Multiplicative Modulation)
```
Without scale: norm_vx = RMSNorm(vx)              ← all timesteps treated equally
With scale:    norm_vx = RMSNorm(vx) × (1 + s)    ← amplify or suppress specific directions
                                            ↑
                         if s = 0   → unchanged
                         if s = 1   → doubled
                         if s = -1  → zeroed (suppressed)
```

At high noise (early steps), scale might suppress irrelevant directions. At low noise (final
steps), scale might amplify subtle refinement directions.

### Shift (Additive Modulation)
```
norm_vx = norm_vx + shift
```

A global constant added to every token. This biases the input in a specific direction of the
hidden space, depending on the current denoising step. Think of it as "prior belief correction."

### Gate (Residual Strength Control)
```
vx = vx + attention_output × gate
```

The gate controls how much of the attention output is actually added back to the residual stream.

- `gate → 0` : "Ignore this layer's output; keep what you had"
- `gate → 1` : "Accept the full attention output"
- `gate → 2` : "Double the influence of this layer"

In the early layers of training, gates are near zero. The model gradually "opens" gates as it
learns that those layers provide useful information. This makes training stable — the identity
function (gate=0) is the default, and improvement is gradual.

### Why All Three Together?

The combination gives the model **full affine control** over each layer's behaviour per timestep.
This is the neural network equivalent of saying:

> "At timestep σ=0.9, pre-condition the self-attention input by stretching it in direction X and
> shifting it toward Y, then use only 30% of what the attention computes (gate=0.3). At σ=0.1,
> do the opposite."

---

## 🌀 2.4 — RoPE: Rotary Position Embeddings

**File:** `rope.py` | **Open in your editor:** `rope.py` is currently open on your screen

### The Fundamental Problem

Standard attention computes:
```
score(token_i, token_j) = Q_i · K_j  / √d
```

This score is **permutation-invariant**: swapping the order of tokens does not change the scores.
The model has no inherent sense of which token came first, second, or twenty-seventh.

In natural language, position matters enormously — "The cat ate the fish" means something
completely different from "The fish ate the cat." We need the model to know where each token is.

### Old Approach: Absolute Position Embeddings (APE)

```python
token_embedding = embedding_table[word] + position_table[position_id]
```

Problem: The model can only handle positions up to `len(position_table)`. Tokens at new,
unseen positions have no embedding. The model cannot generalise beyond training length.

### New Approach: Rotary Position Embeddings (RoPE)

Instead of adding a fixed vector, RoPE **rotates** the Q and K vectors by an angle that depends
on the token's position. The crucial mathematical insight:

```
dot_product( Rotate(Q, θ_i), Rotate(K, θ_j) )  ∝  f(θ_i - θ_j)
```

The dot product becomes a function of the **relative position** (i − j), NOT the absolute positions
i or j. This means:

1. Two tokens at positions 3 and 5 produce the same relative-position signal as tokens at positions
   103 and 105 — the model automatically generalises to longer sequences.
2. Positional information is baked into Q and K **without** modifying the token embeddings V,
   preserving the expressiveness of value vectors.

### The Mathematics

For simplicity, consider a 2D vector `[a, b]` at position `p` with frequency `θ`:

```
Rotation matrix R(pθ) = | cos(pθ)  -sin(pθ) |
                        | sin(pθ)   cos(pθ) |

R(pθ) × [a, b]ᵀ = [a·cos(pθ) - b·sin(pθ),   a·sin(pθ) + b·cos(pθ)]ᵀ
```

For actual heads of dimension `d`, the rotation is applied to **pairs** of dimensions independently.
This is the "interleaved" style used in LTX-2 (`LTXRopeType.INTERLEAVED`):

```python
# rope.py lines 29-39
def apply_interleaved_rotary_emb(input_tensor, cos_freqs, sin_freqs):
    # Reshape to pairs: [a₁, b₁, a₂, b₂, ...] → [[a₁,b₁], [a₂,b₂], ...]
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)   # t1=[a₁,a₂,...], t2=[b₁,b₂,...]

    # Create rotated counterpart: [-b₁, a₁, -b₂, a₂, ...]
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

    # Apply rotation: v·cos(θ) + v_rot·sin(θ)
    # = [a·cos - b·sin, b·cos + a·sin] = R(θ)×[a,b]ᵀ ✓
    return input_tensor * cos_freqs + input_tensor_rot * sin_freqs
```

### Traced Numerical Example

```
Input token at position p=5, dimension=4: v = [3.0, 4.0, 1.0, 2.0]
Frequencies θ₁=0.10, θ₂=0.01 (different per pair)
Angles: pθ₁ = 0.5, pθ₂ = 0.05

Pair 1: [3.0, 4.0]
  cos(0.5) = 0.8776,  sin(0.5) = 0.4794
  rotated  = [3.0×0.8776 - 4.0×0.4794, 3.0×0.4794 + 4.0×0.8776]
           = [2.632 - 1.918, 1.438 + 3.510]
           = [0.714, 4.948]

Pair 2: [1.0, 2.0]
  cos(0.05) = 0.9988, sin(0.05) = 0.0500
  rotated   = [1.0×0.9988 - 2.0×0.0500, 1.0×0.0500 + 2.0×0.9988]
            = [0.999 - 0.100, 0.050 + 1.998]
            = [0.899, 2.048]

Final output: [0.714, 4.948, 0.899, 2.048]  ← now position-encoded!
```

Another token at position p=5 will produce the same rotation → their dot product only depends
on the relative offset between their positions.

### 3D RoPE for Video vs 1D RoPE for Audio

This is one of the most elegant aspects of LTX-2. The same RoPE mechanism scales to different
dimensionalities:

**Video tokens have 3D positions (time, height, width):**
```python
# From model.py:
positional_embedding_max_pos = [20, 2048, 2048]   # [max_frames, max_H, max_W]

# A video token at frame 10, row 512, column 768 has position:
positions = [10/20, 512/2048, 768/2048] = [0.50, 0.25, 0.375]  # normalised to [0,1]
```

The 4096 inner dimensions are allocated:
```
├── First N dims  → encode temporal position (where in time?)
├── Next N dims   → encode vertical position (where vertically in frame?)
└── Last N dims   → encode horizontal position (where horizontally?)
```

**Audio tokens have 1D positions (time only):**
```python
# From model.py:
audio_positional_embedding_max_pos = [20]   # [max_time_in_seconds] only

# An audio token at t=5s has position:
positions = [5/20] = [0.25]  # only one number!
```

**Cross-attention uses 1D temporal RoPE only:**

When video attends to audio (or vice versa), the only shared dimension is **time**. So the
cross-attention PE uses only the temporal component of the video position:

```python
# transformer_args.py lines 195-202:
cross_pe = _prepare_positional_embeddings(
    positions=modality.positions[:, 0:1, :],  # ← ONLY index 0 (time), drop H and W
    inner_dim=self.audio_cross_attention_dim, # ← Use audio dimension (2048)
    max_pos=[self.cross_pe_max_pos],          # ← Single 1D max
)
```

This ensures that when video token at time t=5s attends to audio tokens, it is naturally biased
to attend more to audio tokens near t=5s than those at t=20s.

### Frequency Grid Generation

The frequencies θ are not arbitrary. They are generated on an exponential scale (lines 69-111):

```python
# generate_freq_grid_pytorch() - line 90:
indices = theta ** (
    torch.linspace(
        math.log(start, theta),    # log_θ(1) = 0
        math.log(end, theta),      # log_θ(θ) = 1
        inner_dim // n_elem,       # n = dim / (2 × n_dims)
    )
)
indices = indices * math.pi / 2   # scale to [0, π/2]
```

With `theta=10000`:
```
indices ≈ [1.0, 1.4, 2.0, 2.8, ..., 7071, 10000] (exponential spacing)
```

Multiplied by fractional positions in [−1, +1], the angles sweep from very small (slow rotation =
coarse position) to very large (fast rotation = fine position). This gives the model a
**multi-scale** sense of position.

### `use_middle_indices_grid` — Patch-Level Position

Each token in the LTX-2 latent space represents not a single pixel but a **patch** of pixels —
because the VAE compresses multiple pixels into one token. Therefore, each token covers a
**range** `[start, end]` of positions, not a single point.

```python
# generate_freqs() lines 129-133:
if use_middle_indices_grid:
    indices_grid_start = indices_grid[..., 0]
    indices_grid_end   = indices_grid[..., 1]
    indices_grid = (indices_grid_start + indices_grid_end) / 2.0  # Use middle!
```

When `use_middle_indices_grid=True`, RoPE uses the **centre** of each patch's spatial extent.
This is the correct position to represent the patch — the centre is the best single-point
summary of a patch.

---

## 🔍 2.5 — Attention: The Complete Mechanics

**File:** `attention.py`

### Motivation: "Everyone Talks to Everyone"

In a transformer, every token needs to gather information from all other relevant tokens. The
attention mechanism implements this "everyone talks to everyone" idea efficiently.

### Standard Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax( Q × Kᵀ / √d_k + mask ) × V
```

Let us expand each piece:

**Q (Queries):** "What am I looking for?"
```
Query for token i = to_q(hidden_state_i)
```

**K (Keys):** "What do I offer to be searched for?"
```
Key for token j = to_k(hidden_state_j)
```

**V (Values):** "If you attend to me, here is what you get."
```
Value for token j = to_v(hidden_state_j)
```

**dot product Q × Kᵀ:** Relevance score — "How useful is token j to token i?"
```
score[i,j] = Q_i · K_j   (dot product → scalar)
```

**÷ √d_k:** Stability scaling to prevent exploding gradients.
```
If d_k is large (say 128), dot products grow like √128 on average.
Dividing by √128 ≈ 11.3 re-scales to unit variance.
```

**+ mask:** Control which token pairs are allowed to communicate.
```
mask[i,j] = 0     → allowed  (no change to score)
mask[i,j] = -∞   → blocked   (after softmax, attention weight → 0)
```

**softmax:** Convert raw scores to probabilities (sum to 1 over j for each i).
```
weight[i,j] = exp(score[i,j]) / Σⱼ exp(score[i,j])
```

**× V:** Weighted combination of value vectors.
```
output_i = Σⱼ weight[i,j] × V_j
```

### Multi-Head Attention

Instead of one big attention operation, the embedding dimension is split into 32 **heads**, each
of which performs attention independently:

```
Inner dim = 4096 → split into 32 heads × 128 dims/head

head_k computes attention using its own slice of Q, K, V
→ 32 different "perspectives" on the same data
→ outputs concatenated back to 4096D
→ projected to final output: Linear(4096 → 4096)
```

In code (`attention.py`, `PytorchAttention.__call__` line 29):

```python
# Reshape to multi-head format
q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
# From: [batch, tokens, heads×d_head]
# To:   [batch, heads, tokens, d_head]

out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
# Efficient fused kernel: computes Q·Kᵀ/√d + mask → softmax → ×V

out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
# Reshape back: [batch, tokens, heads×d_head]
```

### QK Normalization

Before attention, both Q and K are RMS-normalised:

```python
self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

q = self.q_norm(self.to_q(x))
k = self.k_norm(self.to_k(context))
```

From `utils.py`:
```python
def rms_norm(x, eps=1e-6):
    return x / sqrt(mean(x²) + ε)   # Divides by RMS magnitude
```

**Why normalise Q and K?** Without normalisation, in deep networks the dot products `Q · K` can
reach extreme magnitudes, causing softmax to produce near-one-hot attention (one token gets
nearly all attention, others get nothing). This makes gradients vanish and training unstable.
After RMSNorm, magnitudes are controlled → distributed, smooth attention patterns.

### The Three Attention Types in Each Block

```
┌──────────────────────────────────────────────────────────────────────┐
│ Type 1: Self-Attention (attn1 / audio_attn1)                        │
│                                                                      │
│  Q = video tokens    K = video tokens    V = video tokens            │
│  context = None  → context = x  (looks at itself!)                  │
│                                                                      │
│  "Each video token gathers context from every other video token"     │
│  "Captures spatial layout, temporal motion, global structure"        │
└──────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────┐
│ Type 2: Text Cross-Attention (attn2 / audio_attn2)                  │
│                                                                      │
│  Q = video/audio tokens    K = text tokens    V = text tokens        │
│                                                                      │
│  "Each video/audio token reads from the text prompt"                 │
│  "Interprets instructions: language, speaker identity, content"      │
└──────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────┐
│ Type 3: AV Cross-Attention (audio_to_video_attn / video_to_audio_attn)│
│                                                                      │
│  Sub-op A:  Q = video tokens    K,V = audio tokens                  │
│  "Video asks audio: what sound is happening now? How should I move   │
│   my lips to match you?"                                             │
│                                                                      │
│  Sub-op B:  Q = audio tokens    K,V = video tokens                  │
│  "Audio asks video: what visual event is happening? Should I make    │
│   a loud impact sound or a soft ambient hum?"                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🌉 2.6 — Bidirectional AV Cross-Attention: The Sync Bridge

**File:** `transformer.py`, lines 186–247

This is the most architecturally unique component of LTX-2. It runs in **every** single one of
the 48 transformer blocks. By the time data has passed through all 48 blocks, the video and audio
representations have exchanged information 48 × 2 = 96 times in each direction.

### Full Code Trace with Annotations

```python
# ─────── STEP 1: Prepare normalised versions of both streams ─────────
# (computed once, used in both sub-operations)
vx_norm3 = rms_norm(vx, eps=self.norm_eps)   # [batch, 500, 4096]
ax_norm3 = rms_norm(ax, eps=self.norm_eps)   # [batch, 80,  2048]

# ─────── STEP 2: Compute cross-attention AdaLN values ─────────────────
# These come from SEPARATE AdaLN modules from the self-attention ones!
# Audio-side: 5 values = 4 scale/shift + 1 gate
(
    scale_audio_for_a2v,  # How to scale audio keys/vals before a2v attention
    shift_audio_for_a2v,
    scale_audio_for_v2a,  # How to scale audio queries before v2a attention
    shift_audio_for_v2a,
    gate_v2a,             # How much of audio-to-video output to keep in audio
) = get_av_ca_ada_values(self.scale_shift_table_a2v_ca_audio, ...)

# Video-side: 5 values = 4 scale/shift + 1 gate
(
    scale_video_for_a2v,
    shift_video_for_a2v,
    scale_video_for_v2a,
    shift_video_for_v2a,
    gate_a2v,             # How much of audio-to-video output to keep in video
) = get_av_ca_ada_values(self.scale_shift_table_a2v_ca_video, ...)

# ─────── STEP 3: Audio → Video (A2V) attention ─────────────────────────
# "Video queries Audio: what are you doing? I want to sync my lips."
if run_a2v:
    # Independently modulate both sides before the attention:
    vx_scaled = vx_norm3 * (1 + scale_video_for_a2v) + shift_video_for_a2v
    ax_scaled = ax_norm3 * (1 + scale_audio_for_a2v) + shift_audio_for_a2v

    vx = vx + (
        self.audio_to_video_attn(
            vx_scaled,                            # Q = video tokens ask questions
            context=ax_scaled,                    # K,V = audio tokens answer
            pe=video.cross_positional_embeddings, # 1D temporal RoPE for Q
            k_pe=audio.cross_positional_embeddings, # 1D temporal RoPE for K
            mask=video.cross_attention_mask,      # ← JustDubit injects mask here!
        )
        * gate_a2v    # Learnable gate (0 → ignore audio, 1 → fully incorporate)
    )

# ─────── STEP 4: Video → Audio (V2A) attention ─────────────────────────
# "Audio queries Video: what visual event are you showing? Match my sound."
if run_v2a:
    ax_scaled = ax_norm3 * (1 + scale_audio_for_v2a) + shift_audio_for_v2a
    vx_scaled = vx_norm3 * (1 + scale_video_for_v2a) + shift_video_for_v2a

    ax = ax + (
        self.video_to_audio_attn(
            ax_scaled,                            # Q = audio tokens ask
            context=vx_scaled,                    # K,V = video tokens answer
            pe=audio.cross_positional_embeddings, # 1D temporal RoPE for audio Q
            k_pe=video.cross_positional_embeddings, # 1D temporal RoPE for video K
            mask=audio.cross_attention_mask,
        )
        * gate_v2a
    )
```

### Why Two Separate Gate Tables?

The `scale_shift_table_a2v_ca_audio` and `scale_shift_table_a2v_ca_video` (lines 101-102) are the
learnable parameter tables for cross-attention:

```python
self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))
```

Audio and video need to be modulated independently before they communicate: they have different
scales, different magnitudes, different statistical distributions at any given timestep. Separate
learnable tables let the model discover the proper pre-conditioning for each modality.

### JustDubit's Cross-Attention Mask

In standard LTX-2 generation: `cross_attention_mask = None` → every video token can attend to
every audio token freely (full cross-attention).

In JustDubit: `cross_attention_mask` is an **explicit masking pattern** that prevents the
conditioning frames (source video) from cross-attending to target audio, and vice versa:

```
                    AUDIO TOKENS
                  Source | Target
                 ════════════════
    V   Source  │  Block│  Allow │
    I           │───────┼────────│
    D   Target  │  Allow│  Allow │
    E           └────────────────┘
    O
```

This is the "modality-isolated cross-attention" innovation. Source video tokens do not need to
synchronise with target audio (they are already dubbed — they are conditioning, not output).
Only target video tokens need to exchange info with target audio tokens (the ones being generated).
This prevents information from leaking across what should be independent conditioning regions.

---

## 🍔 2.7 — Feed-Forward Network: The "Thinking" Layer

**Files:** `feed_forward.py`, `gelu_approx.py`

After attention gathers information from other tokens, the FFN **independently processes** each
token's representation to extract complex, non-linear features.

```python
# feed_forward.py
class FeedForward(torch.nn.Module):
    def __init__(self, dim, dim_out, mult=4):
        super().__init__()
        inner_dim = dim * mult      # Expand 4×: 4096 → 16,384
        self.net = torch.nn.Sequential(
            GELUApprox(dim, inner_dim),    # Linear(4096 → 16384) + GELU
            torch.nn.Identity(),            # No-op (preserves sequential API)
            torch.nn.Linear(inner_dim, dim_out)  # Compress back: 16384 → 4096
        )

# gelu_approx.py
class GELUApprox(torch.nn.Module):
    def forward(self, x):
        return F.gelu(self.proj(x), approximate="tanh")  # Linear + GELU
```

### Dimension Expansion (for Video, dim=4096):
```
Input:   [batch, 500, 4096]   ← 500 video tokens × 4096 dims each
Expand:  [batch, 500, 16384]  ← 4× expansion via Linear + GELU
Compress:[batch, 500, 4096]   ← back to 4096 via Linear
```

### Why Expand Then Compress?

This is the "bottleneck-then-project" paradigm. In the 16,384-dim intermediate space, the model
has room to represent complex combinations of features. In the lower 4096D, the model must be
selective — only the most useful combinations survive the compression.

Think of it like: "spread out all your thoughts in a big notebook (16,384D), then summarise the
key insights concisely (4096D)."

### What is GELU?

**GELU** = Gaussian Error Linear Unit. Formula:
```
GELU(x) = x × Φ(x)   where Φ is the cumulative normal distribution

Approximation used (much faster to compute):
GELU(x) ≈ 0.5x × (1 + tanh(√(2/π) × (x + 0.044715x³)))
```

Comparison with other activations:
```
       │   value │ gradient │  characteristic
───────┼─────────┼──────────┼──────────────────────────────────
ReLU   │ max(0,x)│ 0 or 1   │ hard zero for negative x (dead neurons)
GELU   │ smooth  │ smooth   │ partial pass for slightly negative x
Sigmoid│ 0 to 1  │ small    │ saturates at extremes (vanishing gradient)
SiLU   │ x·σ(x) │ smooth   │ similar to GELU, slightly different shape
```

GELU is preferred in modern transformers because it is differentiable everywhere (no dead
neuron problem) and has shown empirically better performance in large-scale language and vision
models.

### The FFN Is Applied Independently Per Token

A critical property: the FFN applies the same function to **each token independently** (no cross-
token interaction). Only the attention mechanism has cross-token interaction. This is the design
principle of transformers:

- **Attention:** "Gather information from other tokens"
- **FFN:** "Process what I gathered, in my own private space"

---

## 📝 2.8 — Text Projection: Bridging Gemma to the Transformer

**File:** `text_projection.py`

Gemma 3 produces text embeddings of dimension **3840**. The video transformer needs **4096** and
the audio transformer needs **2048**. The `PixArtAlphaTextProjection` maps between these:

```python
class PixArtAlphaTextProjection(torch.nn.Module):
    def __init__(self, in_features=3840, hidden_size=4096):
        self.linear_1 = nn.Linear(3840, 4096)        # ← Project up
        self.act_1    = nn.GELU(approximate="tanh")   # ← Non-linear
        self.linear_2 = nn.Linear(4096, 4096)         # ← Refine

    def forward(self, caption):         # caption: [batch, seq, 3840]
        h = self.linear_1(caption)      # → [batch, seq, 4096]
        h = self.act_1(h)               # → non-linear transformation
        h = self.linear_2(h)            # → [batch, seq, 4096]
        return h
```

There are **two separate instances** — one for each stream:

```python
# model.py lines 133-163:
self.caption_projection = PixArtAlphaTextProjection(
    in_features=3840, hidden_size=4096        # Gemma → Video dim
)
self.audio_caption_projection = PixArtAlphaTextProjection(
    in_features=3840, hidden_size=2048        # Gemma → Audio dim
)
```

The same Gemma embeddings are fed to both, but each projection learns to extract **different
aspects** of the text relevant to its modality. The video projection might learn to emphasise
words describing visual events; the audio projection might learn to emphasise phonetics and
speaking style.

---

## 🧩 2.9 — The `Modality` and `TransformerArgs` Data Structures

**Files:** `modality.py`, `transformer_args.py`

### `Modality` — The Input Container

```python
@dataclass(frozen=True)
class Modality:
    enabled: bool             # Is this modality active in this call?
    latent: torch.Tensor      # Raw latent tokens.    Shape: [B, T, 128]
    timesteps: torch.Tensor   # Noise level per token. Shape: [B, T]
    positions: torch.Tensor   # Spatial positions.     Shape: [B, 3, T] video / [B,1,T] audio
    context: torch.Tensor     # Text embeddings.       Shape: [B, L, 3840]
    context_mask: torch.Tensor  # Which text tokens are real vs padding. Shape: [B, L]
    cross_attention_mask: torch.Tensor | None  # A↔V mask (JustDubit fills this!)
```

The `frozen=True` makes instances immutable (like a named tuple). When you need to update a
field, you use `replace()`:

```python
# transformer.py line 257:
return replace(video, x=vx), replace(audio, x=ax)
# Creates new Modality with same fields but x replaced
```

### `TransformerArgs` — The Preprocessed Container

`TransformerArgs` is what the transformer blocks actually see after preprocessing:

```python
@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor                     # [B, T, 4096]  Patchified and projected
    context: torch.Tensor               # [B, L, 4096]  Text, projected to inner dim
    context_mask: torch.Tensor          # Formatted attention mask (−∞ for padding)
    timesteps: torch.Tensor             # [B, 1, 24576] → 6 modulation values × 4096
    embedded_timestep: torch.Tensor     # [B, 1, 4096]  Raw timestep embedding
    positional_embeddings: tuple        # (cos_freqs, sin_freqs) for self-attention
    cross_positional_embeddings: tuple  # (cos_freqs, sin_freqs) for cross-attention
    cross_scale_shift_timestep: tensor  # AV cross-attn scale/shift conditioning
    cross_gate_timestep: tensor         # AV cross-attn gate conditioning
    cross_attention_mask: tensor | None # Pass-through of av cross-attn mask
    enabled: bool
```

### The `TransformerArgsPreprocessor.prepare()` Pipeline

```python
# transformer_args.py lines 120-148
def prepare(self, modality: Modality) -> TransformerArgs:

    # ── Step 1: Patchify ─────────────────────────────────────────────
    x = self.patchify_proj(modality.latent)
    #   [B, T, 128] × Linear(128 → 4096) → [B, T, 4096]

    # ── Step 2: Timestep embedding ───────────────────────────────────
    timestep, embedded_timestep = self._prepare_timestep(
        modality.timesteps, x.shape[0], modality.latent.dtype
    )
    # σ → sinusoidal → MLP → AdaLN → [B, 1, 6×4096]

    # ── Step 3: Text projection ──────────────────────────────────────
    context, attention_mask = self._prepare_context(
        modality.context, x, modality.context_mask
    )
    # [B, L, 3840] → caption_projection → [B, L, 4096]

    # ── Step 4: Attention mask formatting ────────────────────────────
    attention_mask = self._prepare_attention_mask(attention_mask, x.dtype)
    # Boolean mask → floating-point: 0 or -inf

    # ── Step 5: Positional embeddings ────────────────────────────────
    pe = self._prepare_positional_embeddings(
        positions=modality.positions,   # [B, 3, T] for video
        inner_dim=self.inner_dim,       # 4096
        max_pos=self.max_pos,           # [20, 2048, 2048]
        ...
    )
    # 3D positions → RoPE (cos, sin) tensors

    return TransformerArgs(x=x, context=context, timesteps=timestep, ...)
```

---

## 🌊 2.10 — Complete Forward Pass: Data Tracing End-to-End

Let us trace a concrete example: **one denoising step** of a 4-second dubbed video (30fps, 512×512).

### Input Sizes (estimated after VAE compression)

```
Video latent: compressed by ~8× in time, 8× in space
  Raw:     30fps × 4s × 512 × 512 × 3 = 94,371,840 values
  Latents: 15 frames × 64 × 64 = 61,440 tokens × 128 channels

Audio latent: compressed by ~160× 
  Raw:     48000Hz × 4s = 192,000 samples
  Latents: 1200 tokens × 128 channels
```

### Token Flow Through the Model

```
INPUT TO FORWARD():
  video = Modality(latent=[1, 61440, 128], timesteps=[1, 61440], ...)
  audio = Modality(latent=[1, 1200, 128], ...)

PREPROCESSING:
  video.x = patchify_proj(video.latent)   → [1, 61440, 4096]
  audio.x = audio_patchify_proj(audio.latent) → [1, 1200, 2048]

  video.timesteps → sinusoidal → MLP → [1, 1, 4096 × 6]
  audio.timesteps → sinusoidal → MLP → [1, 1, 2048 × 6]

  video.context → caption_projection → [1, 256, 4096]
  audio.context → audio_caption_projection → [1, 256, 2048]

  video.positions → 3D RoPE → (cos [1, 32, 61440, 128], sin [1, 32, 61440, 128])
  audio.positions → 1D RoPE → (cos [1, 32, 1200, 64], sin [1, 32, 1200, 64])

FOR EACH OF 48 BLOCKS:

  ① Video Self-Attn:
     Q=K=V: [1, 61440, 4096]
     Computes: 61440 × 61440 attention matrix!  ← expensive!
     Output: [1, 61440, 4096]

  ② Video Text Cross-Attn:
     Q: [1, 61440, 4096]  K,V: [1, 256, 4096]
     Attention matrix: 61440 × 256  ← much cheaper
     Output: [1, 61440, 4096]

  ③ Audio Self-Attn:
     Q=K=V: [1, 1200, 2048]
     Attention matrix: 1200 × 1200
     Output: [1, 1200, 2048]

  ④ Audio Text Cross-Attn:
     Q: [1, 1200, 2048]  K,V: [1, 256, 2048]
     Output: [1, 1200, 2048]

  ⑤ AV Cross-Attn (A→V):
     Q: [1, 61440, 4096]  K,V: [1, 1200, 2048]   ← different dims!
     Output: [1, 61440, 4096]  (video stream updated)

  ⑥ AV Cross-Attn (V→A):
     Q: [1, 1200, 2048]  K,V: [1, 61440, 4096]   ← different dims!
     Output: [1, 1200, 2048]  (audio stream updated)

  ⑦ Video FFN:
     [1, 61440, 4096] → [1, 61440, 16384] → [1, 61440, 4096]

  ⑧ Audio FFN:
     [1, 1200, 2048] → [1, 1200, 8192] → [1, 1200, 2048]

POST-PROCESSING:
  video: LayerNorm → scale/shift → proj_out(4096 → 128) → [1, 61440, 128]
  audio: LayerNorm → scale/shift → audio_proj_out(2048 → 128) → [1, 1200, 128]

OUTPUT:
  video_velocity = [1, 61440, 128]  ← direction to move video latent toward clean
  audio_velocity = [1, 1200, 128]   ← direction to move audio latent toward clean
```

### How velocity turns to denoised output

From `utils.py`:
```python
def to_denoised(sample, velocity, sigma):
    return sample - velocity * sigma
    # x_clean ≈ x_noisy - velocity × σ
```

**Intuition:** The velocity is the direction from the current noisy sample toward the clean sample.
Subtracting `velocity × σ` takes one step along the straight-line path from noise to clean.
This is the core of **Flow Matching** (covered in Step 5).

---

## 📊 2.11 — Architecture Summary Card

```
╔══════════════════════════════════════════════════════════════════════╗
║                        LTX-2 MODEL SUMMARY                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  Total Params:       ~19 Billion                                      ║
║  Transformer Blocks: 48                                               ║
║  Attention Backend:  PyTorch SDPA / XFormers / FlashAttention3        ║
╠═══════════════════════════════╦══════════════════════════════════════╣
║  VIDEO STREAM                 ║  AUDIO STREAM                        ║
╠═══════════════════════════════╬══════════════════════════════════════╣
║  Inner dim:   4,096 (32×128) ║  Inner dim:   2,048 (32×64)         ║
║  Patchify in:  128 → 4096    ║  Patchify in:  128 → 2048           ║
║  Text ctx dim: 4,096          ║  Text ctx dim: 2,048                ║
║  Self-attn:    ✅ (attn1)    ║  Self-attn:    ✅ (audio_attn1)    ║
║  Text cross:   ✅ (attn2)    ║  Text cross:   ✅ (audio_attn2)    ║
║  FFN inner:    16,384 (4×)   ║  FFN inner:    8,192 (4×)          ║
║  Position enc: 3D RoPE        ║  Position enc: 1D RoPE (time only)  ║
║  Max dims:  [20, 2048, 2048] ║  Max dims:    [20]                   ║
╠═══════════════════════════════╩══════════════════════════════════════╣
║  CROSS-MODAL (per block, both directions):                           ║
║    A→V attention: video Q  |  audio K,V  |  1D temporal RoPE        ║
║    V→A attention: audio Q  |  video K,V  |  1D temporal RoPE        ║
║    Both use: independent AdaLN scale/shift per side + learnable gate ║
╠══════════════════════════════════════════════════════════════════════╣
║  Text Encoder:    Gemma 3 (12B, quantised), output dim = 3840        ║
║  Text Projection: PixArtAlphaTextProjection (3840→4096, 3840→2048)   ║
║  Precision:       bfloat16                                           ║
║  Normalisation:   RMSNorm everywhere (no LayerNorm in main blocks)   ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## ✅ 2.12 — Quick Knowledge Check

1. **Why does the audio stream have inner_dim=2048 while video has 4096?**
   → Video contains vastly more spatial information per token; asymmetric sizing allocates
   compute proportional to information density.

2. **What are the three values produced by AdaLN for each sub-layer?**
   → `shift` (additive bias), `scale` (multiplicative factor), `gate` (residual strength).
   The gate is the most impactful: it controls whether the sub-layer's output is used at all.

3. **What mathematical operation does RoPE apply?**
   → It rotates Q and K vectors in 2D pairs by an angle proportional to position × frequency.
   The dot product Q·K then becomes a function of relative position only.

4. **Why does the AV cross-attention use only 1D (temporal) RoPE instead of 3D?**
   → Audio has no spatial structure (height/width). The only shared dimension between a video
   token and an audio token is their location in time. 1D temporal RoPE encodes this correctly.

5. **Why expand the FFN to 4× then compress back?**
   → The higher-dimensional intermediate space allows the model to compute complex non-linear
   combinations (attention alone is linear). The compression forces selectivity, retaining only
   useful transformations.

6. **What does `mask=video.cross_attention_mask` do in the AV attention?** (JustDubit specific)
   → It blocks specific video–audio token pairs from attending to each other. JustDubit uses it
   to prevent source video tokens from cross-attending to target audio tokens (modality isolation).

---

*← Previous: [Step 01: The Big Picture](./Step_01_The_Big_Picture.md)*
*→ Next: [Step 03: VAEs — How Video & Audio Get Compressed](./Step_03_VAEs_Compression.md)*
