import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import torch

def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return processor, model

def recognize_text(image, processor, model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def main():
    st.title("Handwritten Text Recognition with TrOCR")
    
    processor, model = load_model()
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    image_url = st.text_input("Or enter an image URL:", "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    elif image_url:
        try:
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return
    else:
        st.warning("Please upload an image or provide a URL.")
        return
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Recognize Text"):
        with st.spinner("Processing..."):
            text = recognize_text(image, processor, model)
        
        st.success("Recognized Text:")
        st.write(text)

if __name__ == "__main__":
    main()
