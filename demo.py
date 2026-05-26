import gradio as gr
from our_predict import predict_from_video

demo = gr.Interface(
    fn=predict_from_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Textbox(label="Predicted Text"),
    title="Lip Reading Demo",
    description="Upload a video to predict spoken words using LipNet."
)

demo.launch(share=True)

