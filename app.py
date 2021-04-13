from transformers import MarianTokenizer, MarianMTModel
from detoxify import Detoxify
import pandas as pd
import os
import streamlit as st 
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():

    st.header("Toxic comment detection")
    text = st.text_area("Enter your text here")
    if st.button("Judge It!"):
    # Initialize the tokenizer
        tokenizer = get_tokenizer()
        # Initialize the model
        model = get_model()

        # Tokenize text
        tokenized_text = tokenizer([text], return_tensors="pt")

        # Perform translation and decode the output
        translation = model.generate(**tokenized_text)
        translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]

        results = Detoxify('original').predict(translated_text)

        # Print translated text
        st.table(pd.DataFrame(results, index=[text]).round(5))

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_tokenizer(): 
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-sv-en")
    return tokenizer

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model(): 
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-sv-en")
    return model

if __name__ == "__main__":
    main()