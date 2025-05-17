import os
import sys
import torch
# Add the path to Video-LLaVA
sys.path.append(os.path.join(os.path.dirname(__file__), 'Video-LLaVA'))
from transformers import pipeline
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
import gradio as gr
import time

def main(path):
    try:
        # Setup
        disable_torch_init()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        # Define models and parameters
        model_path = 'LanguageBind/Video-LLaVA-7B'
        cache_dir = 'cache_dir'
        model_name = get_model_name_from_path(model_path)

        tokenizer, model, processor, _ = load_pretrained_model(
            model_path, None, model_name, device=device, cache_dir=cache_dir
        )
        video_processor = processor['video']
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        # Ensure unique file handling to prevent caching issues
        unique_id = str(int(time.time()))
        temp_audio_path = f"audio_{unique_id}.mp3"

        # Process video
        video_tensor = video_processor(path, return_tensors='pt')['pixel_values']
        if isinstance(video_tensor, list):
            tensor = [vid.to(model.device, dtype=torch.float16) for vid in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)

        # Generate video summary with Video-LLaVA
        input_prompt = (
            "Imagine you are an expert video summarizer. Provide a rich, detailed, and engaging summary of the events in this video."
        )
        conv.append_message(conv.roles[0], input_prompt)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        videollava_output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        # Extract audio and transcribe using Whisper
        clip = VideoFileClip(path)
        clip.audio.write_audiofile(temp_audio_path)

        whisper_model = whisper.load_model("base", device=device)
        transcription_result = whisper_model.transcribe(temp_audio_path)

        # Summarize combined text with meaningful integration of audio transcription
        summarizer = pipeline("summarization", model="Falconsai/text_summarization", device=0 if torch.cuda.is_available() else -1)
        combined_text = (
            f"Video Context: {videollava_output}\nAudio Context: {transcription_result['text']}\n" +
            "Provide a cohesive summary that integrates both the video and audio insights."
        )

        summary_output = summarizer(
            combined_text, max_length=300, min_length=150, do_sample=True
        )

        # Format outputs
        video_summary = f"**Video Summary:**\n{videollava_output}"
        audio_transcription = f"**Audio Transcription:**\n{transcription_result['text']}"
        combined_summary = f"**Final Summary:**\n{summary_output[0]['summary_text']}"

        # Clean up temporary audio file
        os.remove(temp_audio_path)

        return video_summary, audio_transcription, combined_summary

    except Exception as e:
        return f"Error processing video: {str(e)}", "", ""



# Gradio interface
demo = gr.Interface(
    fn=main,
    inputs=[gr.Video(label="Upload a Video")],
    outputs=[
        gr.Textbox(label="Video-LLaVA Output"),
        gr.Textbox(label="Audio Transcription"),
        gr.Textbox(label="Final Summarization")
    ],
    title="Enhanced Video Summarizer with Audio Insights",
    description="Upload a video to get a detailed and engaging summary using advanced AI models, including insights from the audio content."
)

demo.launch(share=True)
