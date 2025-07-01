# AI-Detection-LLM
This is my project for the course Intro to Large Language Models. It takes written text and determines whether an AI or a human wrote the text. It uses three different trained ML models: SVM, Decision Tree, and AdaBoost, as well as three different deep learning models: CNN, RNN, and LSTM, all of which were trained with a dataset containing human and AI-written essays. The web app allows the user to make single predictions with a chosen model via typing in text or uploading a file (.txt, .pdf, or .docx), conduct batch processing by uploading a file containing multiple texts (.txt, .csv, or .docx), and compare model performance on a single prediction by typing in text or uploading a file (.txt, .pdf, or .docx). It also contains information about all of the models and a help page for assistance.

1. To use the app, you'll first need to clone this repository to your local computer. To do so, open Git Bash and navigate to the directory you want the repository to be in by typing "cd (insert path)".
2. Once you're in the desired directory, clone the repository by typing "git clone https://github.com/ennovak/AI-Detection-LLM.git"
3. Now, open the streamlit_ml_app folder in Visual Studio Code.
4. Open a new terminal and type "pip install -r requirements.txt".
5. Once all of the requirements are installed, you can run the app by typing "streamlit run app.py"!
