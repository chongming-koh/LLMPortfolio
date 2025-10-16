# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 00:41:14 2025

@author: Koh Chong Ming
"""

import os
import time
import whisper
import gradio as gr
from IPython.display import Markdown, display, update_display
from openai import OpenAI

#Find the key file

os.chdir("C:\\Users\\vital\\PythonStuff\\keys")
cwd = os.getcwd() 

with open("nebius_api_key", "r") as file:
    nebius_api_key = file.read().strip()

os.environ["NEBIUS_API_KEY"] = nebius_api_key

# Nebius uses the same OpenAI() class, but with additional details
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

# Define models
llm_models = {
    "Llama 3.1 8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen 2.5 72B": "Qwen/Qwen2.5-72B-Instruct",
    "DeepSeek V3 0324": "deepseek-ai/DeepSeek-V3-0324",
    "OpenAI GPT OSS 20B": "openai/gpt-oss-20b",
    "Hermes 4 70B": "NousResearch/Hermes-4-70B"
}

def transcribe_audio(audio_file):
    if audio_file is None:
        return "Please upload an audio file.", "⚠️ No file uploaded."

    print("🟢 Starting transcription...")
    audio_model = whisper.load_model("tiny")
    result = audio_model.transcribe(audio_file)
    transcription = result["text"]
    print("✅ Transcription completed.")

    return transcription, "✅ Transcription completed!"

#Good code. Streaming Version
def summarize_text(transcription, selected_model):
    if not transcription:
        yield "⚠️ No transcription found. Please transcribe first."
        return

    print("🟢 Streaming summary generation...")
    print(f"Model selected: {selected_model}")

    system_prompt = (
        "You are an experienced meeting secretary responsible for producing clear, concise, and well-structured minutes.\n"
        "You identify and summarize minutes of meetings from transcripts, with a summary, key discussion points and insights,"
        "takeaways, unresolved questions or follow-ups, in markdown format with no code blocks."
    )

    prompt = f"""Below is a meeting transcript. Write minutes in markdown with no code blocks including:
    - **Meeting Summary:** (2–3 sentences overview)
    - **Key Discussion Points:** (bulleted list)
    - **Decisions Made:** (bulleted list)
    - **Action Items:** (who / what / when)
    - **Next Steps / Follow-Ups:** (if any)
    - **My Insights:** (Provide your interesting insights of the transcript)

    Transcript:
    {transcription}
    """

    try:
        # ✅ Stream tokens from Nebius client
        stream = nebius_client.chat.completions.create(
            model=llm_models[selected_model],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            max_tokens=1024,
            temperature=0.7,
        )

        full_text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_text += token
                yield full_text  # ✅ sends incremental updates to Gradio Textbox

        print("\n✅ Done streaming.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"❌ Error: {str(e)}"
        
# Define Gradio UI
def build_ui():
    with gr.Blocks(title="🎧 Audio Transcription & LLM Summarizer") as demo:
        gr.Markdown(
            "# 🎧 Audio Transcription & LLM Summarizer\n"
            "Upload your MP3 file, transcribe it, and summarize using your preferred LLM."
        )

        # Two-column layout: Transcription (left) | Summarizer (right)
        with gr.Row():
            # 🎙️ Left column – Transcription
            with gr.Column(scale=1):
                gr.Markdown("## 🎙️ Transcription")
                audio_input = gr.Audio(label="Upload MP3", type="filepath")
                transcribe_btn = gr.Button("Transcribe Audio")
                result_box = gr.Textbox(
                    label="Transcribed Text",
                    placeholder="Transcribed text will appear here.",
                    lines=10
                )
                status_box = gr.Markdown("Waiting for transcription...")

                transcribe_btn.click(
                    fn=transcribe_audio,
                    inputs=audio_input,
                    outputs=[result_box, status_box]
                )

            # 🧠 Right column – Summarizer
            with gr.Column(scale=1):
                gr.Markdown("## 🧠 Summarize with LLM")
                model_dropdown = gr.Dropdown(
                    choices=list(llm_models.keys()),
                    label="Select LLM Model",
                    value="Llama 3.1 8B"
                )
                summarize_btn = gr.Button("Summarize Transcription")
                summary_output = gr.Markdown(label="Summary Output")

                summarize_btn.click(
                    fn=summarize_text,
                    inputs=[result_box, model_dropdown],
                    outputs=summary_output
                )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(inbrowser=True, debug=True)