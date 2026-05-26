import gradio as gr

def predict_lip_reading(video):
    return "Prediction will appear here"

demo = gr.Interface(
    fn=predict_lip_reading,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Textbox(label="Predicted Text"),
    title="Lip Reading Demo",
    description="Upload a video to predict spoken words using LipNet."
)

demo.launch()