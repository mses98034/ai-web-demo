import gradio as gr
from fake_model import predict_image
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text"
)

demo.launch()