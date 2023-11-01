## ChatBot_PDF_Local-Palm

Chat with PDF(s) Chatbot
=====================================

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pustakgpt.streamlit.app)

This repository contains implementation of querying local PDF documents using open source LLMs or Google Palm API, using streamlit for web interface for ease of use, backend uses langchain for heavylifting.

Prerequisites
-------------

To run this application, you need to have the following installed:

* Python (version 3.9 or higher)
* pipenv (for managing the virtual environment)
* Ollama (for running LLMs locally): https://ollama.ai/
* Google Palm API key(if you want to use Palm model):
https://developers.generativeai.google/tutorials/setup

Installation
------------

1. Clone the repository to your local machine:
    
    ```shell
   https://github.com/shitan198u/ChatBot_PDF_Local-Palm.git
    ```
    
2. Navigate to the project directory:
    
    ```shell
    cd ChatBot_PDF_Local-Palm
    ```
3. Install pipenv if you haven't already:
    
    ```shell
   pip install pipenv
    ```

4. Create a virtual environment and install the dependencies:
    
    ```shell
    pipenv install
    ```
    
    This will automatically create a virtual environment and install the required dependencies specified in the `Pipfile`.
    
5. Activate the virtual environment:
    
    ```shell
    pipenv shell
    ```
 6. After installing Ollama:
    
    ```shell
    ollama pull mistral:instruct
    ```

Usage
-----
Run the app:

    streamlit run app.py

Contributing
------------

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please open an issue or a pull request.

License
-------

This project is licensed under the [MIT License](LICENSE).
