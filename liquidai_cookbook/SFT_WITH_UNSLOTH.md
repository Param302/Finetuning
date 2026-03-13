# 🧠 Supervised Fine-Tuning (SFT) & Edge Deployment: A Systems Engineering Approach

**TL;DR:** I successfully engineered an end-to-end Supervised Fine-Tuning (SFT) pipeline on a 1.2 Billion parameter Liquid Foundation Model (LFM2.5-1.2B) using a highly constrained local GPU (6GB RTX 3050). By leveraging 4-bit quantization, Unsloth’s custom CUDA kernels, and Low-Rank Adaptation (LoRA), I trained an 11-million parameter "side-brain" on 9,500 conversational rows with a massive **16,384 token context window**. I bypassed VRAM bottlenecks using gradient accumulation, continuously evaluated against a 500-row holdout set to prevent overfitting, and exported the final merged model to a GGUF format for offline, air-gapped inference via `llama.cpp`.

Here is the master cheatsheet, blueprint, and technical breakdown of my workflow.

---

## 🛠️ 1. Hardware Optimization & Model Loading

**The Challenge:** Loading a 1.2B parameter model in standard 16-bit precision requires ~2.4GB just for weights, and context window memory scales quadratically. My RTX 3050 only has 6GB of VRAM.
**The Solution:** 4-bit Quantization + Unsloth.

* **Technique:** I used `load_in_4bit=True` to mathematically compress the model weights, slashing VRAM usage by over 70% with negligible intelligence degradation.
* **Unsloth:** Bypassed standard PyTorch overhead. Unsloth's optimized kernels and gradient checkpointing use 30% less VRAM, allowing me to push the context window much further than normally possible.

## 🪛 2. Model Surgery: Parameter-Efficient Fine-Tuning (PEFT)

**The Challenge:** Updating all 1.2 billion parameters (Full SFT) would crash the GPU instantly and require massive compute.
**The Solution:** Low-Rank Adaptation (LoRA).

* **What I Did:** I froze the entire base model to make it read-only.
* **Target Modules:** I surgically injected empty adapter matrices into the Attention (`q_proj`, `k_proj`, `v_proj`, `out_proj`) and MLP (`w1`, `w2`, `w3`) layers.
* **The Math (`r=16`):** By setting the rank to 16, I decomposed the weight updates. Instead of training 1.2 billion parameters, I trained a hyper-focused **11.1 million parameters** (less than 1% of the model).
* **Hardware Hacks:** Set `lora_dropout=0` and `bias="none"` to allow Unsloth to fuse operations at the hardware level, drastically speeding up the training loop.

## 📊 3. Data Engineering & The 16k Context Window

**The Challenge:** Models don't read raw text; they read structured token sequences. Furthermore, I needed to ensure the model actually learned, rather than just memorized.

* **The Split:** I took a 10,000-row dataset and strictly partitioned it: **9,500 rows for training** and **500 rows for evaluation**.
* **Chat Templating:** Mapped the raw JSON into the strict `<|im_start|>user...` conversational syntax required by the Liquid model.
* **Response Masking (`train_on_responses_only`):** *Crucial step.* I explicitly instructed the loss function to ignore the user's prompt (masked with `-100`). The model is only mathematically penalized for getting the *answer* wrong, not the question.

## ⚙️ 4. The Training Engine (Surviving the VRAM Limit)

**The Challenge:** Pushing a **16,384 token context window** on a 6GB GPU is practically impossible with standard batching.
**The Solution:** Micro-batching and Gradient Accumulation.

Here is the exact production-grade configuration I used for the `SFTTrainer`:

* **`per_device_train_batch_size = 1`:** I forced the GPU to process only *one* 16k-token conversation at a time to avoid a CUDA Out of Memory (OOM) crash.
* **`gradient_accumulation_steps = 8`:** To prevent erratic learning from a batch size of 1, I accumulated the gradients over 8 steps in memory before actually updating the weights. This effectively faked a batch size of 8 without blowing up VRAM.
* **`num_train_epochs = 1`:** Replaced the hard-coded `max_steps` kill-switch to ensure the model saw every single one of the 9,500 training rows.
* **`lr_scheduler_type = "cosine"`:** Smoothly curved the learning rate down to zero at the end of the epoch, allowing the model to perfectly settle into the optimal weights.
* **Telemetry (`eval_steps=50`):** Hooked the training loop up to Weights & Biases (`wandb`). Every 50 steps, the model was tested against the 500 hidden evaluation rows to ensure the Validation Loss was dropping alongside the Training Loss (verifying true generalization, not overfitting).

## 🚀 5. The Escape Hatch: Edge Deployment

**The Challenge:** A model trapped in a cloud Jupyter notebook is just a prototype. It needs to run locally.

* **GGUF Export:** I utilized Unsloth's native `save_pretrained_gguf` function.
* **The Merge:** The script automatically took the frozen base model, mathematically merged my newly trained 11-million parameter LoRA weights into the base architecture, and quantized the entire package into a single `.gguf` file.
* **Offline Inference:** I can now drop this file into `llama.cpp` or LM Studio and run a private, highly-specialized, air-gapped AI system entirely on CPU or edge hardware.

---

### 💡 Core Pro-Tips & Learnings

1. **SFT vs. LoRA:** SFT is the *objective* (teaching by example). LoRA is the *mathematical tool* that makes SFT physically possible on consumer hardware.
2. **Context Scales Quadratically:** You cannot double the context window without exponentially increasing memory pressure. If you increase `max_seq_length`, you *must* aggressively drop `batch_size` and increase `gradient_accumulation` to compensate.
3. **The "Tutorial Illusion":** Running `max_steps=60` on 10k rows means the model only ever saw 480 rows. To actually train, you must use epochs and process the full dataset.
4. **Data Quality is the Moat:** The math (LoRA, Unsloth, Optimizers) is a solved commodity. The true differentiator of an AI engineer is the ability to curate, clean, and format the 10,000 rows of highly specific data that goes *into* this pipeline.