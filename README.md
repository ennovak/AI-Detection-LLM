# AI-Detection-LLM
This is my project for the course Intro to Large Language Models. It takes written text and determines whether an AI or a human wrote the text. It uses three different trained ML models: SVM, Decision Tree, and AdaBoost, all of which were trained with a dataset containing human and AI-written essays. The web app allows the user to manually type text for single predictions with a chosen model, compare results between all three models, or make predictions on uploaded files (.txt, .csv, .pdf, or .docx).

To run the app, first download the streamlit_ml_app folder and unzip it.
Then, you'll need to have Python installed. The steps are as follows:
  1. Go to python.org and download the latest Python version
  2. Run the installer and check "Add Python to PATH"
  3. Click "Install Now"
  4. Verify installation by opening Command Prompt and typing "python --version"
If you see your Python version, then you have successfully installed Python, and you can continue.
Then, you'll need to make sure you're in the right directory. Navigate to the streamlit_ml_app folder. You can find the path by right-clicking the folder and selecting "Copy as Path". Once you have the right path, type "cd YourPathHere" in your Command Prompt.
Once you're in the folder, you need to create a virtual environment. To do so, type "python -m venv mystreamlitapp1" into your Command Prompt. The name can be anything; it doesn't have to be mystreamlitapp1.
To activate the virtual environment, type "mystreamlitapp1\Scripts\activate" in your Command Prompt. Again, the name doesn't have to be mystreamlitapp1, but it must match the name you wrote in the previous step.
Now you need to install dependencies, so navigate to the inner folder by typing "cd streamlit_ml_app" in your Command Prompt. Once that's done, check to make sure you're in the right folder by typing "ls" in your Command Prompt. If you see the requirements.txt and app.py files, then you're in the right folder.
Now you can install the dependencies by typing "pip install -r requirements.txt" in your Command Prompt. This should install all needed dependencies and may take a few minutes to install.
Once that's done, open the streamlit_ml_app folder in VSCode. If there are any errors with importing libraries, you can install missing dependencies by typing "pip install (insert dependency name here)" in the terminal.
Now you're ready to run the app. You can run it by typing "streamlit run app.py" in the terminal.
