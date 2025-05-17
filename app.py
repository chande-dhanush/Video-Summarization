from flask import Flask, request, render_template, jsonify
import os
import sys
import torch
import google.generativeai as genai
import random
# Add the path to Video-LLaVA
sys.path.append(os.path.join(os.path.dirname(__file__), 'Video-LLaVA'))
from transformers import pipeline
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyC7nhvSQSzSdIxIBMdbBlGI53cqRbfP6tw"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Global variables to store the latest video data
latest_video_summary = ""
latest_audio_transcription = ""
latest_final_summary = ""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Save uploaded file
            video_file = request.files["video"]
            video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
            video_file.save(video_path)

            # Process video and generate summary
            summary_video, summary_text, transcription = process_video(video_path)
            
            # Store the latest data for chat access
            global latest_video_summary, latest_audio_transcription, latest_final_summary
            latest_video_summary = summary_video
            latest_audio_transcription = transcription
            latest_final_summary = summary_text

            # Return results
            return jsonify({
                "Video Summary": summary_video,
                "Audio Transcription": transcription,
                "Final Summary": summary_text
            })
        except Exception as e:
            # Clean up any temporary files
            for temp_file in [video_path, "audio.mp3"]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            
            # Free GPU memory in case of error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Return error message
            return jsonify({
                "error": f"An error occurred during processing: {str(e)}"
            }), 500
    
    return render_template("index.html")  # Serve HTML interface

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get the question from the request
        data = request.json
        question = data.get("question", "")
        
        if not question:
            return jsonify({"answer": "Please provide a question about the video content."}), 400
        
        # If no video has been processed yet
        if not latest_video_summary and not latest_audio_transcription and not latest_final_summary:
            return jsonify({"answer": "Please upload and process a video first."}), 400
        
        # Use Gemini model to answer the question
        answer = get_gemini_response(question)
        
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

def get_gemini_response(question):
    """
    Use Gemini API to provide a contextual answer based on the video content
    """
    try:
        # Combine all video content as context
        context = f"""
Video Analysis: {latest_video_summary}

Audio Transcription: {latest_audio_transcription}

Final Summary: {latest_final_summary}
        """
        
        # Create a prompt that instructs Gemini to answer based only on the video content
        prompt = f"""You are an AI assistant that answers questions about video content. 
You only have information about a specific video that has been processed.

Here is what we know about the video:
{context}

Please answer the following question ONLY based on the information provided above.
If the answer cannot be determined from the information provided, DO NOT simply say "I don't see information about that in the video content." 
Instead, provide a creative and varied response that explains the limitation in a friendly way. For example:
- "I've searched through the video details, but there's no mention of that specific information."
- "The video content doesn't appear to show or discuss that particular aspect."
- "That's an interesting question, but from what I can see in the video information, this isn't addressed."
- "While I can tell you about other aspects of the video, that particular detail isn't included in what I can access."
- "I don't have enough information from this video to answer that specific question, but I can tell you about [something else from the video]."

Be conversational, helpful and creative in your responses while making it clear when information isn't available.

Question: {question}
Answer:"""

        # Generate response from Gemini
        response = model.generate_content(prompt)
        
        # Extract and return the answer text
        answer = response.text.strip()
        
        # Limit response length if needed
        max_length = 500
        if len(answer) > max_length:
            answer = answer[:max_length] + "..."
            
        return answer
    
    except Exception as e:
        print(f"Error with Gemini API: {str(e)}")
        # Fallback to simple keyword matching if Gemini fails
        return find_relevant_answer(question, f"{latest_video_summary} {latest_audio_transcription} {latest_final_summary}")

def find_relevant_answer(question, content):
    """
    Find a relevant answer to the question from the content.
    This is a simple implementation and could be improved with actual NLP.
    Used as a fallback if Gemini API fails.
    """
    # Convert to lowercase for case-insensitive matching
    question_lower = question.lower()
    
    # Extract keywords from the question (simple approach)
    # Exclude common stop words
    stop_words = ["what", "when", "where", "who", "why", "how", "does", "did", "is", "are", 
                 "was", "were", "will", "would", "could", "should", "can", "about", "the", 
                 "a", "an", "in", "on", "at", "to", "for", "with", "by", "of"]
    
    keywords = [word for word in question_lower.split() if word not in stop_words and len(word) > 3]
    
    # If no meaningful keywords found, return a generic response
    if not keywords:
        if "summary" in question_lower or "about" in question_lower:
            return latest_final_summary
        
        # Varied responses for vague questions
        vague_responses = [
            "I need a bit more context to answer that. Could you be more specific about what you'd like to know from the video?",
            "That's a bit general for me to answer based just on this video. Could you ask something more specific?",
            "I'm not sure exactly what you're looking for. Could you rephrase your question with more details?",
            "I'd be happy to help, but I need a more specific question about the video content.",
            "Your question is a bit broad - could you narrow it down to a specific aspect of the video?"
        ]
        return random.choice(vague_responses)
    
    # Split content into sentences for more targeted responses
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    
    # Score each sentence based on keyword matches
    sentence_scores = []
    for sentence in sentences:
        score = 0
        sentence_lower = sentence.lower()
        for keyword in keywords:
            if keyword in sentence_lower:
                score += 1
        if score > 0:
            sentence_scores.append((sentence, score))
    
    # Return the highest scoring sentence(s)
    if sentence_scores:
        # Sort by score (highest first)
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top sentence with a period
        top_sentence = sentence_scores[0][0]
        if not top_sentence.endswith('.'):
            top_sentence += '.'
        return top_sentence
    
    # Fallback responses when no matching information is found
    fallback_responses = [
        f"The video content doesn't seem to mention anything about {' or '.join(keywords)}.",
        f"I searched through the video information, but couldn't find anything related to {' or '.join(keywords[:2] if len(keywords) > 2 else keywords)}.",
        f"That's an interesting question about {' and '.join(keywords[:2] if len(keywords) > 2 else keywords)}, but it's not covered in this video content.",
        f"While the video shows people in a classroom setting, I don't see specific information about {' or '.join(keywords[:2] if len(keywords) > 2 else keywords)}.",
        f"The video doesn't appear to address that particular topic. Would you like to know about something else from the video?",
        f"I don't see any details about {keywords[0] if keywords else 'that'} in what I can access from the video content."
    ]
    
    # Special cases
    if "summarize" in question_lower or "summary" in question_lower:
        return latest_final_summary
    if "transcript" in question_lower:
        return "The full transcript is available in the Audio Transcription tab."
    
    return random.choice(fallback_responses)

def process_video(path):
    try:
        # Clear CUDA cache before starting
        torch.cuda.empty_cache()
        
        # Set memory limits for CUDA to prevent OOM errors
        if torch.cuda.is_available():
            # Reserve only a portion of GPU memory
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        disable_torch_init()
        inp = 'Please provide a detailed summary of the video, focusing on the main events as they unfold. Identify the key actions, decisions, and turning points in the narrative. Explain the sequence of events clearly and concisely, highlighting the cause-and-effect relationships between them. Include the beginning, middle, and end of the story or process presented in the video.'
        model_path = 'LanguageBind/Video-LLaVA-7B'
        cache_dir = 'cache_dir'
        
        # Use CPU for some operations to reduce GPU memory usage
        cpu_device = 'cpu'
        gpu_device = 'cuda'
        
        model_name = get_model_name_from_path(model_path)
        
        # Load model with memory optimization
        tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, device=gpu_device, cache_dir=cache_dir)
        video_processor = processor['video']
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles

        # Process video on CPU first
        video_tensor = video_processor(path, return_tensors='pt')['pixel_values']
        
        # Move to GPU only when needed
        if type(video_tensor) is list:
            tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)

        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Process tokens on CPU first then move to GPU
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(gpu_device)
        
        # Limit max tokens to prevent memory issues
        max_new_tokens = min(512, 1024)  # Reduce from 1024 to prevent memory issues
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # Free up memory before generation
        torch.cuda.empty_cache()
        
        with torch.inference_mode():
            # Use reduced max_new_tokens to prevent memory issues
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=max_new_tokens,  # Use the reduced value
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            
            # Free up GPU memory immediately after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        # Extract audio and transcribe - do this on CPU to save GPU memory
        with VideoFileClip(path) as clip:
            audio_path = "audio.mp3"
            clip.audio.write_audiofile(audio_path)

        # Free GPU memory before loading whisper model
        torch.cuda.empty_cache()
        
        # Load whisper model on CPU to save GPU memory
        with torch.device('cpu'):
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
        
        # Free GPU memory before loading summarizer
        torch.cuda.empty_cache()
        
        # Use CPU for summarization to save GPU memory
        device_map = "cpu"
        summarizer = pipeline("summarization", model="Falconsai/text_summarization", device_map=device_map)

        # Combine outputs and limit length to prevent token sequence length errors
        ARTICLE = outputs + result['text']
        
        # BART model has a max length of 1024 tokens, so we need to truncate if longer
        # Split into chunks if needed and process separately
        max_chunk_length = 1000  # Slightly less than 1024 to be safe
        
        if len(ARTICLE) > max_chunk_length:
            # Process in chunks
            chunks = [ARTICLE[i:i+max_chunk_length] for i in range(0, len(ARTICLE), max_chunk_length)]
            summaries = []
            
            for chunk in chunks:
                if len(chunk.strip()) > 100:  # Only process substantial chunks
                    try:
                        chunk_summary = summarizer(chunk, max_length=100, min_length=10, do_sample=True)
                        summaries.append(chunk_summary[0]['summary_text'])
                    except Exception as e:
                        print(f"Error summarizing chunk: {e}")
                        continue
            
            # Combine summaries if we have multiple
            if summaries:
                combined_summary = " ".join(summaries)
                # Final summarization pass on the combined summaries if needed
                if len(combined_summary) > max_chunk_length:
                    out = [{'summary_text': combined_summary[:max_chunk_length]}]
                else:
                    try:
                        out = summarizer(combined_summary, max_length=500, min_length=100, do_sample=True)
                    except Exception as e:
                        print(f"Error in final summarization: {e}")
                        out = [{'summary_text': combined_summary[:200]}]
            else:
                # Fallback if all summarization attempts failed
                out = [{'summary_text': ARTICLE[:200] + "..."}]
        else:
            # Process normally if within token limits
            try:
                out = summarizer(ARTICLE, max_length=500, min_length=100, do_sample=True)
            except Exception as e:
                print(f"Error in summarization: {e}")
                # Fallback to a simple truncation
                out = [{'summary_text': ARTICLE[:200] + "..."}]

        # Format the transcription and summary for better display
        formatted_transcription = result['text'].strip()
        formatted_summary = out[0]['summary_text'].strip()
        
        # Ensure the video summary is properly formatted
        formatted_video_summary = outputs.strip()
        
        return formatted_video_summary, formatted_summary, formatted_transcription
    
    finally:
        # Free GPU memory
        torch.cuda.empty_cache()

        # Clean up temporary files
        if os.path.exists(path):
            os.remove(path)
        if os.path.exists("audio.mp3"):
            os.remove("audio.mp3")

if __name__ == "__main__":
    app.run(debug=True)