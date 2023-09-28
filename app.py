import gradio as gr
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("summarization", model="ayoubkirouane/billsum-BART-base-cnn")

# Function to perform text summarization
def summarize_text(input_text):
    summary = pipe(input_text, max_length=1024, min_length=50, do_sample=False)[0]['summary_text']
    return summary

# Create a Gradio interface
iface = gr.Interface(
    fn=summarize_text,
    inputs="text",
    outputs="text",
    allow_flagging=False , 
    title="Text Summarization App Using billsum-BART-base-cnn Model ",
    description="Enter a piece of text, and this app will generate a summary using the 'billsum-BART-base-cnn' summarization model.",
)

# Launch the app
iface.launch(debug=True)