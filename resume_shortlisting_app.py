import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import numpy as np
import torch
import streamlit as st
from tika import parser
import string
import nltk
import pandas as pd 

nltk.download('stopwords')

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
st.set_page_config(page_title="Smart Resume Finder")
st.title("Smart Resume Finder")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = np.mean(outputs.last_hidden_state.cpu().numpy(), axis=1)
    return embedding

def preprocess_text(text):
    text = text.lower()
    
    punctuation_to_remove = string.punctuation.replace('@', '').replace('.', '')
    
    translation_table = str.maketrans('', '', punctuation_to_remove)
    
    text = text.translate(translation_table)
    
    text = ' '.join(text.split())
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    
    lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in filtered_words]

    
    lemmatized_text = ' '.join(lemmatized_words)
    
    return lemmatized_text

def find_most_suited_resume(preprocessed_files, job_description_embedding, skills_embedding):
    highest_similarity = -1
    most_suited_resume_index = -1
    resume_similarities = []  # List to store resume index and average similarity

    # Calculate embeddings and similarities
    for idx, resume_text in enumerate(preprocessed_files):
        # Get embedding for the current resume
        resume_embedding = get_embedding(resume_text)
        
        similarity_description = cosine_similarity(job_description_embedding, resume_embedding)[0][0]
        similarity_skills = cosine_similarity(skills_embedding, resume_embedding)[0][0]
        
        average_similarity = (similarity_description + similarity_skills) / 2
        
        resume_similarities.append((idx, average_similarity))
        
        if average_similarity > highest_similarity:
            highest_similarity = average_similarity
            most_suited_resume_index = idx
    
    return most_suited_resume_index, highest_similarity, resume_similarities

# Allow the user to upload multiple PDF files
st.sidebar.header("Resume Uploader")
uploaded_files = st.sidebar.file_uploader("Upload Resumes (PDF files)", type="pdf", accept_multiple_files=True)

# If files are uploaded
if uploaded_files:
    # Process each uploaded file
    list_files = []
    for uploaded_file in uploaded_files:
        raw = parser.from_buffer(uploaded_file.getvalue())
        
        list_files.append(raw['content'])

    preprocessed_files = []

    # Process each extracted text in list_files
    for text in list_files:
        preprocessed_text = preprocess_text(text)
        
        preprocessed_files.append(preprocessed_text)

job_Description = st.text_area("Enter Job Description:",height=200)
skills = st.text_area("Enter Skills:",height=200)

if st.button("Match Resumes") and preprocessed_files:
    with st.spinner("Matching resumes..."):
        job_description_embedding = get_embedding(job_Description)
        skills_embedding = get_embedding(skills)

        most_suited_resume_index, highest_similarity, resume_similarities = find_most_suited_resume(preprocessed_files, job_description_embedding, skills_embedding)

        # Display the most suited resume information
        if most_suited_resume_index != -1:
            st.header("Most Suited Resume")
            st.markdown(f"Resume Index: **{most_suited_resume_index}**")
            st.markdown(f"Cosine Similarity: **{highest_similarity:.2f}**")
            st.write("Details:")
            st.text(list_files[most_suited_resume_index])
        else:
            st.warning("No matching resumes found.")

        # Display all resume indices with their cosine similarity
        st.header("Resume Matches")
        if resume_similarities:

            matches_df = pd.DataFrame(resume_similarities, columns=["Resume Index", "Cosine Similarity"])
            matches_df["File Name"] = [uploaded_files[idx].name for idx, _ in resume_similarities]


            matches_df.sort_values(by="Cosine Similarity", ascending=False, inplace=True)
            styled_table = matches_df.style.format({"Cosine Similarity": "{:.2f}"}) \
                                        .background_gradient(cmap="Blues")  # Color gradient

            st.table(styled_table)

st.markdown(
    """
    <style>
    /* Customize header */
    .css-1h6yrub {
        color: #1f77b4; /* Set color */
    }

    /* Customize button */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
    }

    /* Customize table headers */
    .css-1stp9se th {
        background-color: #1f77b4;
        color: white;
    }
    /* Customize the font size of the text areas */
    textarea {
        font-size: 38px; /* Adjust the font size as needed */
    }
    /* Customize font size of text in text areas */
    div[data-testid="stTextArea"] textarea {
        font-size: 18px; /* Adjust the font size as needed */
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown("---")
st.markdown("Created by Tanisha Harde")