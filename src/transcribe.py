from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

class SpeechTranscriber:
    def __init__(self, model_name="openai/whisper-base"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

    def transcribe(self, audio, sr):
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs.input_features)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
