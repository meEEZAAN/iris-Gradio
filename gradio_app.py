import joblib
import numpy as np
import gradio as gr

# Load model
model = joblib.load("app/model.joblib")
class_names = np.array(["setosa", "versicolor", "virginica"])

def predict_and_explain(sepal_length, sepal_width, petal_length, petal_width):
    # Prepare features
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict class + probabilities
    probs = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]

    prob_dict = {name: float(p) for name, p in zip(class_names, probs)}

    # Image path — adjust path for Docker
    image_path = f"app/images/{pred_name}.jpg"

    explanation = (
        f"### Prediction: **{pred_name}**\n\n"
        f"- Sepal: {sepal_length:.1f} × {sepal_width:.1f} cm\n"
        f"- Petal: {petal_length:.1f} × {petal_width:.1f} cm\n\n"
        "Typically:\n"
        "- **Setosa** has short petals\n"
        "- **Virginica** has long and wide petals\n"
        "- **Versicolor** is in between\n"
    )

    return prob_dict, explanation, image_path

with gr.Blocks(title="Iris Classifier") as demo:
    gr.Markdown("# 🌸 Iris Classifier (Random Forest)")
    gr.Markdown(
        "Move the sliders or pick an example to see which Iris species the model predicts, "
        "along with the probability for each class."
    )

    with gr.Row():
        with gr.Column():
            sepal_length = gr.Slider(4.0, 8.0, value=5.1, step=0.1, label="Sepal length (cm)")
            sepal_width  = gr.Slider(2.0, 4.5, value=3.5, step=0.1, label="Sepal width (cm)")
            petal_length = gr.Slider(1.0, 7.0, value=1.4, step=0.1, label="Petal length (cm)")
            petal_width  = gr.Slider(0.1, 2.5, value=0.2, step=0.1, label="Petal width (cm)")

            predict_btn = gr.Button("🔮 Predict")

            gr.Examples(
                examples=[
                    [5.1, 3.5, 1.4, 0.2],
                    [6.0, 2.9, 4.5, 1.5],
                    [6.9, 3.1, 5.4, 2.1],
                ],
                inputs=[sepal_length, sepal_width, petal_length, petal_width],
                label="Try some typical Iris examples"
            )

        with gr.Column():
            probs_label = gr.Label(
                label="Class probabilities",
                num_top_classes=3,
            )
            explanation_md = gr.Markdown()

            # 👇 add this line
            image_output = gr.Image(label="Predicted Iris image", type="filepath")

    # Button click event
    predict_btn.click(
        fn=predict_and_explain,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=[probs_label, explanation_md, image_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
