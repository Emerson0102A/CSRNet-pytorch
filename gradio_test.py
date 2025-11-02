import gradio as gr

def add(a, b):
    return float(a) + float(b)

demo = gr.Interface(
    fn=add,
    inputs=[gr.Number(label="a"), gr.Number(label="b")],
    outputs=gr.Number(label="sum"),
    title="Adder",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        show_error=True,
        share=True,
    )
