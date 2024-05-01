# Smart Resume Finder

Smart Resume Finder is a Streamlit application that helps you find the most suitable resume from a collection based on a given job description and set of skills. The application uses pre-trained language models to compute the similarity between the job description, skills, and the text of the resumes, allowing you to efficiently identify the best matches.
You can either run it by uploading resumes from your mobile phone or desktop files using streamlit app or resumes from a folder stored locally along with required job description and set of skills.

## Features

- **Upload Resumes**: Upload multiple PDF resumes to the application.
- **Job Description and Skills Input**: Enter the job description and the required skills for the position.
- **Resume Matching**: The application processes the uploaded resumes and matches them to the job description and skills using cosine similarity.
- **Best Match**: Displays the most suited resume based on the highest similarity.
- **List of Resumes with Similarity**: Provides a list of all resumes with their respective cosine similarity scores.

## Requirements

- Python 3.7 or higher
- Required Python packages:
    - [streamlit](https://pypi.org/project/streamlit/)
    - [transformers](https://pypi.org/project/transformers/)
    - [nltk](https://pypi.org/project/nltk/)
    - [torch](https://pypi.org/project/torch/)
    - [tika](https://pypi.org/project/tika/)
    - [sklearn](https://pypi.org/project/scikit-learn/)
    - [numpy](https://pypi.org/project/numpy/)

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/smart-resume-finder.git
    cd smart-resume-finder
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Unix-based systems
    venv\Scripts\activate    # For Windows
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application using the following command:
    ```bash
    streamlit run resume_shortlisting_app.py
    ```

2. The application will open in your browser at `http://localhost:8501`.

3. Follow the prompts to upload resumes, enter the job description and required skills, and find the most suitable resume.



## Acknowledgments

This project uses pre-trained language models from the [Hugging Face Transformers](https://github.com/huggingface/transformers) library and other open-source libraries such as NLTK, scikit-learn, and Apache Tika.
A folder of sample_resumes have been provided for the user to test the streamlit app. This was sourced from publicly available sample job resumes online.

Happy matching!

![image](https://github.com/tjh31/resumeshortlist/assets/64650488/7a8facc4-2404-4736-93ce-5f4cb72749a0)
![image](https://github.com/tjh31/resumeshortlist/assets/64650488/87a4e2a5-45fe-4d1a-9c66-d272751f98e0)
![image](https://github.com/tjh31/resumeshortlist/assets/64650488/0f1eb769-f648-4e59-be99-2a1afc7c25fd)

![image](https://github.com/tjh31/resumeshortlist/assets/64650488/5cb15499-2bea-4314-8a96-459b0e43171f)





