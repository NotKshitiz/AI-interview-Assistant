import time
import random
import gradio as gr
import whisper
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer, pipeline

# Simple human fillers
FILLERS = [
    "hmm…",
    "okay…",
    "right…",
    "got it…",
    "yeah…",
    "mm-hmm…",
    "I see…",
    "go on…",
    "alright…",
    "interesting…",
]

MODEL_ID = "t5-small"

class LightweightAI:
    def __init__(self):
        self.metrics = {"load_time": 0, "last_latency": 0, "fillers": 0}

        #Load Whisper (STT)
        print("Loading Whisper Tiny (CPU)...")
        self.stt_model = whisper.load_model("tiny")

        #Load ONNX T5-small
        print("Loading ONNX summarizer (CPU)...")
        start_load = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            export=True,
            provider="CPUExecutionProvider"
        )

        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer
        )

        self.metrics["load_time"] = round(time.time() - start_load, 2)
        print(f"Ready in {self.metrics['load_time']}s")
        print("Providers:", self.model.providers)

    # STT
    def transcribe_audio(self, audio_path):
        if audio_path is None:
            return ""
        result = self.stt_model.transcribe(audio_path)
        return result["text"].strip()

    # Filler
    def insert_filler(self, recent_text=""):
        self.metrics["fillers"] += 1

        t = recent_text.lower()
        if len(t.split()) < 5:
            return "go on…"
        if "because" in t or "so" in t:
            return "okay…"
        if "i think" in t:
            return "hmm…"

        return random.choice(FILLERS)

    # ONNX summarization
    def summarize_text(self, text):
        if not text or len(text.split()) < 3:
            return "Waiting for meaningful speech…"

        start = time.time()

        out = self.summarizer(
            "summarize: " + text,
            max_length=25,
            min_length=5,
            num_beams=1
        )

        self.metrics["last_latency"] = int((time.time() - start) * 1000)
        return out[0]["summary_text"]


# Instantiate engine
ai = LightweightAI()

def run_demo(audio_file):
    speech_text = ai.transcribe_audio(audio_file)

    if not speech_text:
        filler = ai.insert_filler("")
        return "No speech detected.", f"Interviewer: {filler}", 0, ai.metrics["fillers"]

    summary = ai.summarize_text(speech_text)
    filler = ai.insert_filler(speech_text)

    status = f'Transcribed:\n"{speech_text}"\n\nInterviewer: "{filler}"'

    return summary, status, ai.metrics["last_latency"], ai.metrics["fillers"]


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("#Project")
    with gr.Row():
        with gr.Column(scale=2):
            input_audio = gr.Audio(
                label="Speak",
                type="filepath",
                sources=["microphone"]
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            summary_box = gr.Textbox(label="Summary", interactive=False)
            status_box = gr.Textbox(label="Conversation", interactive=False, lines=8)

    with gr.Row():
        latency_stat = gr.Number(label="Inference Latency (ms)", value=0, interactive=False)
        filler_stat = gr.Number(label="Fillers Used", value=0, interactive=False)
        gr.Markdown(f"**Model Load Time:** {ai.metrics['load_time']}s")

    submit_btn.click(run_demo, input_audio, [summary_box, status_box, latency_stat, filler_stat])

demo.launch(share=True, debug=True)