# Video-Summarization
We created a pipeline that takes a video as an input, passes it on to 2 different models for video details extraction and the other for Audio Detail extraction, both the outputs are merged using falcons ai - text summarizer, that's returned as the final output. We finally built an interactive UI using React and Cursor AI




### Steps to run
1. Clone the Video-LLaVa model https://github.com/PKU-YuanGroup/Video-LLaVA.git (it's open source)
2. Install all the requirements from the requirements file, using a virtual environment is recommended so that u don't mess up all your configs.
3. For Interactive ui, run app.py and for gradio ui, run gradio_runner.py
   
