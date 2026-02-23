# 📚 JUST-DUB-IT Deep Dive Documentation

> Complete, code-backed technical documentation for the JUST-DUB-IT: Video Dubbing via
> Joint Audio-Visual Diffusion project.
>
> Each step builds on the previous. Read them in order for best results.

---

## 📖 Step Index

| # | Title | Key Concepts | Status |
|---|-------|-------------|--------|
| [01](./Step_01_The_Big_Picture.md) | **The Big Picture** | Problem, innovation, repo structure, package relationships | ✅ Done |
| [02](./Step_02_LTX2_Foundation_Model.md) | **LTX-2 Foundation Model** | DiT, dual-stream, AdaLN, RoPE, attention types, FFN, velocity model | ✅ Done |
| 03 | **VAEs — Compression** | 3D Video VAE, mel-spectrogram Audio VAE, latent spaces | 🔜 Coming |
| 04 | **Text Encoding with Gemma** | Gemma 3 architecture, dual context embeddings, caption projection | 🔜 Coming |
| 05 | **Flow Matching & Denoising** | Rectified flow, noise schedules, Euler ODE solver, sigma values | 🔜 Coming |
| 06 | **The JustDubit Pipeline** | Two-stage inference, CFG guidance, conditioning, cross-attn mask | 🔜 Coming |
| 07 | **IC-LoRA: The Key Innovation** | In-Context LoRA, modality-isolated attention, LoRA math | 🔜 Coming |
| 08 | **Training Data Pipeline** | Language switching, inpainting, latent masking, lip augmentation | 🔜 Coming |
| 09 | **The Trainer** | Training loop, masked loss, gradient checkpointing, dataset format | 🔜 Coming |
| 10 | **End-to-End Flow** | Complete traced example from input video to dubbed MP4 | 🔜 Coming |

---

## 🗂️ Primary Code Files Referenced

```
packages/
├── ltx-core/src/ltx_core/
│   ├── model/transformer/
│   │   ├── model.py              ← Step 02: LTXModel, LTXModelType
│   │   ├── transformer.py        ← Step 02: BasicAVTransformerBlock
│   │   ├── attention.py          ← Step 02: Attention (Q-K-V, multi-head)
│   │   ├── adaln.py              ← Step 02: AdaLayerNormSingle
│   │   ├── rope.py               ← Step 02: 3D/1D RoPE, apply_rotary_emb
│   │   ├── feed_forward.py       ← Step 02: FeedForward, GELUApprox
│   │   ├── modality.py           ← Step 02: Modality dataclass
│   │   ├── transformer_args.py   ← Step 02: Preprocessing pipeline
│   │   ├── text_projection.py    ← Step 02: PixArtAlphaTextProjection
│   │   └── timestep_embedding.py ← Step 02: sinusoidal timestep encoding
│   ├── model/video_vae/          ← Step 03
│   ├── model/audio_vae/          ← Step 03
│   ├── text_encoders/gemma/      ← Step 04
│   └── components/               ← Step 05
├── ltx-pipelines/src/ltx_pipelines/
│   ├── pipeline_justdubit.py     ← Step 06
│   └── ic_lora.py                ← Step 07
└── ltx-trainer/src/ltx_trainer/
    ├── training_strategies/justdubit.py  ← Step 08
    └── trainer.py                        ← Step 09
```

---

## 🧠 Key Concepts Quick Reference

| Concept | Where Used | Short Explanation |
|---------|-----------|-------------------|
| **DiT** | Step 02 | Diffusion Transformer — uses transformer architecture for diffusion generation |
| **Dual-Stream** | Step 02 | Video and audio each have their own transformer path, with cross-attention bridges |
| **Asymmetric** | Step 02 | Video (4096D) is twice as wide as audio (2048D) |
| **AdaLN-Single** | Step 02 | Scale/shift/gate each layer based on the current noise level (timestep) |
| **RoPE** | Step 02 | Rotary Position Embeddings — encode position via vector rotation |
| **3D RoPE** | Step 02 | Video uses time+height+width; audio uses time only |
| **Bidirectional Cross-Attention** | Step 02 | A→V and V→A attention in every block — the sync bridge |
| **GELU** | Step 02 | Smooth activation function used in FFN |
| **Flow Matching** | Step 05 | Training/inference paradigm: predict velocity along straight path from noise to data |
| **VAE** | Step 03 | Variational AutoEncoder — compress video/audio into compact latent tokens |
| **LoRA** | Step 07 | Low-Rank Adaptation — tiny (~0.1%) trainable adapter on top of frozen model |
| **IC-LoRA** | Step 07 | In-Context LoRA — the specific adapter JustDubit uses for dubbing |
| **CFG** | Step 06 | Classifier-Free Guidance — enhance adherence to conditioning |

---

*Generated: 2026-02-24 | Code version: just-dub-it main branch*
