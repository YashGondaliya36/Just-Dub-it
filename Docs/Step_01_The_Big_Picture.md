# 📖 Step 1 — The Big Picture: What, Why, and How JUST-DUB-IT Works

> **Series:** JUST-DUB-IT Deep Dive | **File:** 01 of 10
> **Code Source:** `f:/Just-Dub/just-dub-it/` | **Paper:** arXiv:2601.22143

---

## 🎯 What You Will Learn in This Step

- What video dubbing is and why it is extremely hard
- Why old multi-stage approaches fail
- The core innovation of JUST-DUB-IT (single unified model)
- Complete repository structure — every folder, every file, why it exists
- The relationship between the three packages
- Your full learning roadmap for all 10 steps

---

## 🎬 1.1 — What is Video Dubbing?

Imagine you have a video of a person speaking **English**. You want to produce a new version of that
video where:

```
INPUT  ──→  OUTPUT
─────────────────────────────────────────────────────────
Person speaking English        Person speaking French
   + English audio         →      + French audio
   + English lip movements         + French lip movements
   + same background               + SAME background
   + same pose/clothing            + SAME pose/clothing
   + same lighting                 + SAME lighting
   + natural background sounds     + SYNCHRONIZED ambient sounds
```

This is **video dubbing**. It is what happens when Hollywood releases a movie in 20 languages —
except now an AI does it automatically.

On the surface it sounds simple. In practice, it is one of the hardest problems in computer vision
and audio processing combined, because every small mistake is instantly visible and audible to a
human observer: a slightly wrong lip shape, a voice that sounds different, a background sound that
cuts at the wrong moment — all of these break the illusion immediately.

---

## 🚧 1.2 — Why is This So Hard? (The Problem in Depth)

### 1.2.1 — The Multi-Modal Coupling Problem

Video and audio are **not independent**. They are coupled in time with millisecond precision:

```
Time →   0.0s     0.1s     0.2s     0.3s     0.4s
         ─────────────────────────────────────────
Video:   [lips:O] [lips:U] [lips:M] [lips:rest] ...
Audio:    "Oh—"    "—ouu—"  "—mm"    (silence)  ...
```

If your audio says "Bonjour" but your lips move as if saying "Hello", the viewer immediately
notices. The synchronisation must be **exact**, not approximate.

### 1.2.2 — The Old Multi-Stage Pipeline (And Why It Breaks)

Before JUST-DUB-IT, the standard approach was to chain specialised modules together:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Legacy Multi-Stage Pipeline                       │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  Separate │   │ Translate│   │ Generate │   │  Paste   │        │
│  │  voice   │──▶│  text    │──▶│  new TTS │──▶│  new     │        │
│  │  from BG │   │  to FR   │   │  voice   │   │  lips    │        │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
│       ↑                                             ↑               │
│  Source Separation fails          Face stitching produces          │
│  when speech overlaps             unnatural seams and              │
│  with environment sounds          fails with head motion           │
└─────────────────────────────────────────────────────────────────────┘
```

**Every handoff between modules introduces errors:**

| Module Failure Point | Why It Fails | Real-World Impact |
|---------------------|--------------|-------------------|
| **Source Separation** | Cannot perfectly separate speech from background (echo, overlap, laughter) | BG audio is degraded, artifacts appear |
| **Text-to-Speech** | Generic TTS does not preserve speaker identity | The person sounds like a robot, not themselves |
| **Face Lipsyncing** | Assumes frontal face, clean image; fails with motion, occlusion, side angles | Lips look fake, especially when person turns head |
| **Audio Re-mixing** | French is often longer than English → silence gaps appear | The video feels unnatural, timing is off |
| **No Cross-Talk** | Each module is ignorant of the others | Errors cascade; fixing one module creates new problems downstream |

**The deepest problem:** treating audio and video as **separate signals** that get edited
independently and then merged. Nature does not work this way. When you speak, your voice and
your mouth motion are produced by the same physical system simultaneously. Separating and
re-merging them always loses that synchronisation.

---

## 💡 1.3 — The JUST-DUB-IT Innovation: One Model, Everything Simultaneously

JUST-DUB-IT's breakthrough is conceptually simple:

> **Use a single AI model that generates BOTH video and audio at the same time.**

```
┌─────────────────────────────────────────────────────────────────────┐
│                   JUST-DUB-IT Approach                              │
│                                                                     │
│                   ┌───────────────────────┐                         │
│   SOURCE VIDEO ──▶│                       │──▶ DUBBED VIDEO         │
│   (English)       │    LTX-2 Foundation   │    (French lips)        │
│                   │    Model (19B params)  │                         │
│   TEXT PROMPT ──▶│    + JustDubit LoRA   │──▶ DUBBED AUDIO         │
│   (French text)   │    (tiny adapter)     │    (French voice)        │
│                   └───────────────────────┘                         │
│                                                                     │
│   ✅ Video and audio generated TOGETHER — always in sync            │
│   ✅ Background sounds naturally adapt                              │
│   ✅ Speaker identity preserved                                     │
│   ✅ Robust to complex motion, side angles, occlusions              │
└─────────────────────────────────────────────────────────────────────┘
```

The AI model (called **LTX-2**, built by Lightricks) is a **19-billion-parameter foundation model**
that already knows how to jointly generate video + audio. JUST-DUB-IT adds a tiny **LoRA adapter**
(Low-Rank Adaptation — roughly 0.1% of the total parameters) that teaches the model:

> *"When I give you a source video + a new text dialogue, re-generate only the face and voice to
> match the new dialogue, but keep everything else the same."*

Because video and audio are generated **jointly in a single forward pass**, they are always
perfectly synchronised — not as an afterthought, but by construction.

---

## 📦 1.4 — Repository Structure: Every File Explained

The repository is a **monorepo** (one Git repository containing several installable Python packages)
managed by **`uv`**, a fast modern Python package manager.

```
just-dub-it/
│
├── pyproject.toml          ← ROOT CONFIG: workspace definition, dev tools (ruff, pytest)
├── uv.lock                 ← LOCKED DEPS: exact versions of every dependency
├── README.md               ← Project overview
├── JUST-DUB-IT_Complete_Architecture_Explained.docx  ← Reference doc
│
└── packages/               ← THREE sub-packages, each independently installable
    │
    ├── ltx-core/           ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │   │                   THE BRAIN — Neural network architectures (raw PyTorch)
    │   ├── pyproject.toml  ← Package definition: name="ltx-core", no CLI entry points
    │   └── src/ltx_core/
    │       ├── model/
    │       │   ├── transformer/      ← The 19B DiT transformer (video + audio streams)
    │       │   │   ├── model.py      ← LTXModel class: the whole transformer
    │       │   │   ├── transformer.py← BasicAVTransformerBlock: one layer
    │       │   │   ├── attention.py  ← Attention: Q-K-V computation
    │       │   │   ├── adaln.py      ← AdaLayerNormSingle: timestep conditioning
    │       │   │   ├── rope.py       ← Rotary Position Embeddings (3D+1D)
    │       │   │   ├── feed_forward.py ← FeedForward: MLP sublayer
    │       │   │   ├── modality.py   ← Modality dataclass: input data container
    │       │   │   ├── transformer_args.py ← Preprocessing pipeline
    │       │   │   ├── text_projection.py  ← Project Gemma output to right dim
    │       │   │   └── timestep_embedding.py ← Convert σ to embedding vector
    │       │   ├── video_vae/        ← 3D VAE: compress/decompress video
    │       │   ├── audio_vae/        ← 1D VAE: compress/decompress audio
    │       │   ├── upsampler/        ← Spatial 2× upscaler (Stage 2)
    │       │   └── common/           ← Shared model utilities
    │       ├── conditioning/         ← How inputs (images, video, audio) condition generation
    │       ├── components/           ← Building blocks: scheduler, noiser, guider, diffusion step
    │       ├── text_encoders/gemma/  ← Wrapper around Gemma 3 text encoder
    │       ├── guidance/             ← CFG (Classifier-Free Guidance) logic
    │       ├── loader/               ← Load checkpoints, apply LoRA weights
    │       ├── tools.py              ← AudioLatentTools, VideoLatentTools helpers
    │       ├── types.py              ← Shared data types: LatentState, AudioLatentShape
    │       └── utils.py              ← rms_norm, to_velocity, to_denoised
    │
    ├── ltx-pipelines/      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │   │                   THE DIRECTOR — Orchestrates end-to-end inference
    │   ├── pyproject.toml
    │   └── src/ltx_pipelines/
    │       ├── pipeline_justdubit.py  ← ⭐ MAIN FILE: JustDubitPipeline class
    │       ├── pipeline_utils.py      ← Shared encode/decode/denoise helpers
    │       ├── model_ledger.py        ← Smart model loader (manages GPU memory)
    │       ├── media_io.py            ← Read/write MP4 video and audio files
    │       ├── ic_lora.py             ← In-Context LoRA implementation
    │       ├── distilled.py           ← Stage 2 distilled refinement
    │       ├── constants.py           ← AUDIO_SAMPLE_RATE, sigma values, etc.
    │       └── utils.py               ← Pipeline helper utilities
    │
    └── ltx-trainer/        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        │                   THE COACH — All the tools to train your own LoRA
        ├── AGENTS.md        ← AI agent instructions for automated training
        ├── CLAUDE.md        ← Claude AI context notes
        ├── pyproject.toml
        ├── configs/         ← YAML training configs (justdubit.yaml etc.)
        ├── docs/            ← Additional training documentation
        ├── scripts/
        │   ├── train.py             ← Main training entry point
        │   └── process_dataset.py  ← Preprocess videos into latents
        └── src/ltx_trainer/
            ├── trainer.py               ← Main training loop (46,968 bytes — largest file!)
            ├── training_strategies/
            │   ├── justdubit.py         ← ⭐ JustDubit-specific training logic
            │   ├── text_to_video.py     ← Text-to-video strategy (for reference)
            │   ├── video_to_video.py    ← Video-to-video strategy
            │   └── base_strategy.py     ← Abstract base class all strategies inherit
            ├── datasets.py              ← Data loading and batching
            ├── validation_sampler.py    ← Generate samples during training (53KB!)
            ├── config.py                ← Parse and validate training YAML config
            ├── model_loader.py          ← Load model for training
            ├── captioning.py            ← Auto-caption videos for dataset building
            └── timestep_samplers.py     ← Sample noise levels during training
```

---

## 🔗 1.5 — How the Three Packages Relate to Each Other

```
                    ┌───────────────────────────────┐
                    │          ltx-core              │
                    │    "The Building Material"     │
                    │                                │
                    │  Defines raw neural networks:  │
                    │  - LTXModel (transformer)      │
                    │  - VideoVAE (compression)      │
                    │  - AudioVAE (compression)      │
                    │  - Text encoder wrappers       │
                    │  - Data types & math utils     │
                    └───────────┬───────────┬────────┘
                                │           │
                   depends on   │           │  depends on
                                │           │
               ┌────────────────▼─┐     ┌───▼──────────────────┐
               │  ltx-pipelines   │     │    ltx-trainer        │
               │  "The Director"  │     │    "The Coach"        │
               │                  │     │                       │
               │  Orchestrates    │     │  Teaches the brain    │
               │  inference:      │     │  new skills:          │
               │  1. Load models  │     │  1. Load dataset      │
               │  2. Encode input │     │  2. Add noise         │
               │  3. Denoise 30x  │     │  3. Run forward pass  │
               │  4. Decode       │     │  4. Compute loss      │
               │  5. Save MP4     │     │  5. Update LoRA       │
               └──────────────────┘     └───────────────────────┘
                       ↑                          ↑
                       │                          │
               YOU use this to                YOU use this to
               dub a video                    train a new LoRA
```

**Dependency rule:** `ltx-pipelines` and `ltx-trainer` both import from `ltx-core`. They **never**
import from each other. `ltx-core` has no dependencies on the other two.

---

## 🔑 1.6 — The `pyproject.toml` — Understanding the Workspace

From the root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["packages/*"]           # All three packages are workspace members

[tool.uv.sources]
ltx-core = { workspace = true }    # Use local code, not PyPI version
ltx-pipelines = { workspace = true }

[dependency-groups]
dev = [
    "pre-commit>=4.3.0",           # Run code checks before git commit
    "ruff>=0.14.3",                # Lightning-fast Python linter + formatter
    "pytest~=9.0",                 # Testing framework
]
```

**`uv`** is like `pip` + `conda` + `virtualenv` combined, written in Rust — extremely fast. When
you run `uv sync --frozen`, it reads `uv.lock` and installs **exactly** those versions. The
`--frozen` flag means "don't update anything, just use what's locked."

---

## 📐 1.7 — Intuition: Why a Single Model Beats a Pipeline

Think of it this way. Imagine teaching someone to translate a speech:

**Approach A (Old Pipeline):**
1. Person A writes down what was said
2. Person B translates the text
3. Person C reads the translation in a voice booth
4. Person D animates the face using the audio

Every handoff loses information. Person C doesn't know the speaker's original voice. Person D
doesn't know the emotion in Person C's delivery. The result feels assembled, not natural.

**Approach B (JUST-DUB-IT):**
One person who simultaneously hears the original speech, understands the translation, and
produces the dubbed version — because in their mind, all modalities are processed together.
They naturally synchronise everything because they experience it holistically.

This is precisely what LTX-2 achieves architecturally: both audio and video latents pass through
**the same transformer** simultaneously, exchanging information at every single layer via
bidirectional cross-attention. The model cannot make an audio decision without the video context,
and vice versa.

---

## 🗺️ 1.8 — Your Full Learning Roadmap

| Step | Topic | Key Files | Concepts |
|------|-------|-----------|----------|
| **01** | The Big Picture ← *You are here* | `README.md`, `pyproject.toml` | Problem, innovation, repo structure |
| **02** | LTX-2 Foundation Model | `model/transformer/model.py`, `transformer.py`, `attention.py`, `adaln.py`, `rope.py` | DiT, dual-stream, AdaLN, RoPE, attention, FFN |
| **03** | VAEs — Compression | `model/video_vae/`, `model/audio_vae/` | 3D spatial-temporal compression, mel spectrograms |
| **04** | Text Encoding with Gemma | `text_encoders/gemma/` | How text becomes dual context embeddings |
| **05** | Flow Matching & Denoising | `components/schedulers.py`, `components/noisers.py` | Rectified flow, noise schedules, Euler steps |
| **06** | The JustDubit Pipeline | `pipeline_justdubit.py`, `pipeline_utils.py` | Two-stage generation, CFG guidance, cross-attn mask |
| **07** | IC-LoRA — The Key Innovation | `ic_lora.py`, training LoRA files | In-Context LoRA, modality-isolated cross-attention |
| **08** | Training Data Pipeline | `training_strategies/justdubit.py` | Language switching, inpainting, latent-aware masking, lip augmentation |
| **09** | The Trainer | `trainer.py`, `datasets.py`, `config.py` | Training loop, masked loss, gradient checkpointing |
| **10** | End-to-End Flow | All files | Run it yourself, trace a complete example |

---

## ✅ 1.9 — Quick Knowledge Check

After reading this step, you should be able to answer:

1. **Why does separating audio and video processing cause problems in dubbing?**
   → Because they are coupled in time with millisecond precision; editing one without awareness
   of the other always breaks synchronisation.

2. **What does `uv sync --frozen` do?**
   → Installs exactly the dependency versions listed in `uv.lock`, without updating anything.

3. **Which package would you look at if you want to understand how the neural network attention
   layers work?**
   → `ltx-core`, specifically `src/ltx_core/model/transformer/`

4. **Which package manages loading models and running inference?**
   → `ltx-pipelines`, specifically `pipeline_justdubit.py`

5. **Can `ltx-pipelines` import code from `ltx-trainer`?**
   → No. Only `ltx-core` is shared. `ltx-pipelines` and `ltx-trainer` are independent consumers
   of `ltx-core` and do not depend on each other.

---

*Next → [Step 02: LTX-2 Foundation Model — Every Gear Inside the Machine](./Step_02_LTX2_Foundation_Model.md)*
