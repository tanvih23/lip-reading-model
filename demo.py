import gradio as gr
from our_predict import predict_from_video
from preprocess import convert

def run_demo(video):
    if video is None:
        return "No video uploaded"
    
    # Gradio passes a temp file path — convert it first
    converted = convert(video)
    result = predict_from_video(converted)
    
    if not result or result == "":
        return "Could not predict — make sure the video shows a clear front-facing view"
    
    return f"Predicted: {result}"

interface = gr.Interface(
    fn=run_demo,
    inputs=gr.Video(format="mp4"),  # tell Gradio to convert to mp4 on upload
    outputs=gr.Textbox(label="Predicted Text"),
    title="Lip Reading Demo",
    description="Upload a video to predict spoken words using LipNet."
)

interface.launch()