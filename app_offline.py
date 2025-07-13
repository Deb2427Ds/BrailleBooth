import streamlit as st
from PIL import Image
import pytesseract
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import pyttsx3

# Load models from Hugging Face
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
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

# Simplify summary
def simplify_text(text):
    return "Here’s a simpler version: " + ' '.join([w for w in text.split() if len(w) <= 8])

# Elaborate summary
def elaborate_text(text):
    return text + " This information is important because it helps readers understand the topic more deeply."

# Streamlit UI
st.title("BrailleBooth – GenAI Accessibility App")
st.write("Upload a textbook or story image (and optional diagram) to generate accessible output.")

# Uploads
text_image = st.file_uploader("📄 Upload Text Page Image", type=["png", "jpg", "jpeg"])
diagram_image = st.file_uploader("🖼️ Upload Diagram Image (Optional)", type=["png", "jpg", "jpeg"])

if st.button("Generate Output"):
    if text_image:
        image = Image.open(text_image)
        raw_text = pytesseract.image_to_string(image)

        # Summarization
        summary = summarizer(raw_text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']

        # Image captioning
        if diagram_image:
            diag = Image.open(diagram_image).convert("RGB")
            inputs = processor(diag, return_tensors="pt")
            output = caption_model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)
        else:
            caption = "No diagram provided."

        # Combine and convert
        combined = f"Summary: {summary}\n\nDiagram: {caption}"
        braille = text_to_braille(combined)
        generate_audio(combined)

        # Display results
        st.subheader("📝 Summary")
        st.write(summary)

        st.subheader("🖼️ Image Caption")
        st.write(caption)

        st.subheader("⠿ Braille Version")
        st.code(braille)

        st.subheader("🔊 Audio")
        st.audio("output.mp3")

        # Feedback options
        st.subheader("💬 Feedback Options")
        if st.button("🧒 Explain in Simpler Words"):
            st.info(simplify_text(summary))
        if st.button("📚 Elaborate Further"):
            st.info(elaborate_text(summary))

    else:
        st.warning("Please upload a text image to begin.")
