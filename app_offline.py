import streamlit as st
from PIL import Image
import pytesseract
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import pyttsx3

# Load models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="./models/distilbart")
    processor = BlipProcessor.from_pretrained("./models/blip")
    caption_model = BlipForConditionalGeneration.from_pretrained("./models/blip")
    return summarizer, processor, caption_model

summarizer, processor, caption_model = load_models()

# Braille mapping
def text_to_braille(text):
    braille_dict = {
        'a':'⠁','b':'⠃','c':'⠉','d':'⠙','e':'⠑','f':'⠋','g':'⠛','h':'⠓','i':'⠊','j':'⠚',
        'k':'⠅','l':'⠇','m':'⠍','n':'⠝','o':'⠕','p':'⠏','q':'⠟','r':'⠗','s':'⠎','t':'⠞',
        'u':'⠥','v':'⠧','w':'⠺','x':'⠭','y':'⠽','z':'⠵',' ':' ','1':'⠼⠁','2':'⠼⠃',
        '3':'⠼⠉','4':'⠼⠙','5':'⠼⠑','6':'⠼⠋','7':'⠼⠛','8':'⠼⠓','9':'⠼⠊','0':'⠼⠚',
        '.':'⠲',',':'⠂','?':'⠦','!':'⠖','-':'⠤',':':'⠒',';':'⠆','\'':'⠄','\"':'⠐⠦','(':'⠐⠣',')':'⠐⠜'
    }
    return ''.join(braille_dict.get(c.lower(), '⍰') for c in text)

# Offline text-to-speech
def generate_audio(text, filename="output.mp3"):
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()

# Simplify text
def simplify_text(text):
    # Basic simplification logic
    return " ".join([word if len(word) <= 10 else "[simplified]" for word in text.split()])

# Elaborate text
def elaborate_text(text):
    return text + " This means it can help students understand the topic better."

# UI
st.title("BrailleBooth – Offline GenAI App")
st.write("Upload textbook image and optional diagram. Get a simplified summary, image description, Braille output, and audio narration.")

text_image = st.file_uploader("📄 Upload Text Page Image", type=["jpg", "jpeg", "png"])
diagram_image = st.file_uploader("🖼️ Upload Diagram Image (Optional)", type=["jpg", "jpeg", "png"])

if st.button("🔁 Generate Accessible Output"):
    if text_image:
        # OCR
        image = Image.open(text_image)
        ocr_text = pytesseract.image_to_string(image)

        # Summarization
        summary = summarizer(ocr_text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']

        # Caption
        if diagram_image:
            diag = Image.open(diagram_image).convert("RGB")
            inputs = processor(diag, return_tensors="pt")
            out = caption_model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        else:
            caption = "No diagram provided."

        # Combine
        final_text = f"Summary: {summary}\n\nDiagram: {caption}"
        braille_text = text_to_braille(final_text)
        generate_audio(final_text)

        st.subheader("📝 Summary")
        st.write(summary)

        st.subheader("🖼️ Image Description")
        st.write(caption)

        st.subheader("⠿ Braille Output")
        st.code(braille_text)

        st.subheader("🔊 Audio")
        st.audio("output.mp3")

        # Feedback section
        st.subheader("💬 Feedback")
        if st.button("🧒 Explain in simpler words"):
            st.info(simplify_text(summary))
        if st.button("🔍 Elaborate further"):
            st.info(elaborate_text(summary))
    else:
        st.warning("Please upload a text image to begin.")
