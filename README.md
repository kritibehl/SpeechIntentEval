# SpeechIntentEval — Speech Intent Classification & Evaluation

Live Demo: https://huggingface.co/spaces/kriti0608/SpeechIntentEval

Mini-project to simulate a Siri-style speech intent pipeline:
1. Load short wake-word / command audio clips (e.g., “stop”, “go”, “on”, “off”).
2. Extract features from raw audio.
3. Train a simple intent classifier.
4. Evaluate accuracy + confusion matrix over the labeled samples.

---

## Quickstart

```bash
git clone https://github.com/kritibehl/SpeechIntentEval.git
cd SpeechIntentEval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
