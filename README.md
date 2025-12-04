# SpeechIntentEval
**Benchmark and evaluation demo for indirect, emotional, hedged, and socially-loaded speech.**

LLMs rarely fail because they “don’t know the answer.”
They fail because they **misunderstand human intent**.

Humans communicate with:
- **hints** (“It’s freezing in here.”)
- **emotional states** (“I’ve been exhausted lately.”)
- **hedged commitments** (“I *should* finish that paper.”)
- **sarcasm** (“Sure, whatever.”)

Traditional evaluations measure:
- toxicity,
- factuality,
- correctness.

They *do not* measure:
> **Did the model understand what the human actually meant?**

SpeechIntentEval focuses on **pragmatic understanding**.

---

#  Why this matters

Most current safety systems do one of two things:
- **Over-refusal**: “I cannot assist with that.”
- **Literal misunderstanding**: “The current temperature is 18°C.”

Humans expect:
- **helpful action**,
- **empathy**, or
- **contextual reasoning**.

This project evaluates those behaviors.

---

# What SpeechIntentEval evaluates

### ✔️ Speech act recognition
Is the utterance:
- Indirect request?
- Emotional complaint?
- Hedged planning?
- Sarcasm?
- Polite refusal?

### ✔️ Interpretation
Does the system explain why?

### ✔️ Response alignment
Does the model’s reply:
- understand the implied task?
- acknowledge emotion?
- respond appropriately?

---

# Demo (Highly recommended)
**Live Hugging Face Space**  
https://huggingface.co/spaces/kriti0608/SpeechIntentEval

Paste:
- utterance
- model reply
- interpret

Simple + powerful.
This is the “aha” moment.

---

# Features

- **Intent inference with explanation**
- **Verdict on response quality**
- **Human-aligned categories**
- Fully client-side and easy to extend
- No API keys required for v1

---

# Quickstart

Clone:

```bash
git clone https://github.com/kritibehl/SpeechIntentEval
cd SpeechIntentEval
