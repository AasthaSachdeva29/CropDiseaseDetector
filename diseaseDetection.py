import os
import mimetypes
from pathlib import Path

import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv

# Load GOOGLE_API_KEY from a .env file placed next to this script
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyBnr7YPdpSjFW_2EVQGvV_0qrYPYMddBi0AQ.Ab8RN6J3q_49DcfyBUMvhcKrmuS_vgIUM0PqBtGtOt8b2FeXhQ"))

# generation_config = {
#     "temperature": 0.5,
#     "top_p": 1.0,
#     "top_k": 32,
#     "max_output_tokens": 4096,
# }

safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

generation_config = {
    "temperature": 0.5,
    "top_p": 1.0,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": f"HARM_CATEGORY_{c}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for c in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

input_prompt = """You are a highly skilled plant pathologist specializing in rice crops.
Analyze the provided image and respond in this structure:

1. Crop Healthy or Not: one word.
2. Disease Identification: disease name in one word.
3. Detailed Findings: 3 bullet points on affected plant parts, symptoms, and likely causes.
4. Name of Fungicides: commonly used fungicides that can cure this disease.
5. Next Steps: bullet points covering treatment, prevention, and pesticide/fungicide names.
6. Recommendations: advice for maintaining plant health and preventing spread.

Disclaimer: This analysis is based on plant pathology principles and should not replace
professional agricultural advice. Consult qualified agricultural experts before applying any treatment.
"""


def read_image_data(file_path):
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return {"mime_type": mime, "data": path.read_bytes()}


def generate_response(prompt, image_path):
    response = model.generate_content([prompt, read_image_data(image_path)])
    return response.text


def process_uploaded_files(files):
    if not files:
        return None, "Please upload an image."
    first = files[0]
    # Works with both newer Gradio (file path string) and older (object with .name)
    file_path = first if isinstance(first, str) else first.name
    try:
        return file_path, generate_response(input_prompt, file_path)
    except Exception as e:
        return file_path, f"Error: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# 🌾 Rice Crop Disease Detector")
    upload_button = gr.UploadButton(
        "Click to upload an image",
        file_types=[".jpg", ".jpeg", ".png", ".webp"],
        file_count="multiple",
    )
    image_output = gr.Image(label="Uploaded image")
    file_output = gr.Textbox(label="Analysis", lines=20)

    upload_button.upload(
        process_uploaded_files,
        upload_button,
        [image_output, file_output],
    )

# launch() must be outside the `with` block
demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))