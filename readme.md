# GeminiRepoChat


https://github.com/user-attachments/assets/0b0534b8-e5e3-4db5-91e9-86b9303f7526



## Description

GeminiRepoChat is a powerful tool that allows users to interact with GitHub repositories using a chat interface powered by Google's Gemini AI. This application enables developers, code reviewers, and curious minds to explore repository contents, analyze code, and ask questions about specific files, all within an intuitive Streamlit-based interface.

## Features

- **Repository Exploration**: Fetch and browse contents of any public GitHub repository.
- **File Selection**: Choose specific files from the repository to analyze or ask questions about.
- **AI-Powered Chat**: Interact with the Gemini AI model to ask questions about selected files.
- **Code Analysis**: Get instant insights into code metrics, complexity, and potential code smells.
- **Syntax Highlighting**: View code with proper syntax highlighting for better readability.
- **Caching**: Efficient caching of repository contents to reduce API calls and improve performance.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- A GitHub account (for generating a personal access token)
- A Google Cloud account (for accessing the Gemini AI API)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/GeminiRepoChat.git
   cd GeminiRepoChat
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the project root directory with the following contents:
   ```
   GITHUB_TOKEN=your_github_personal_access_token
   GOOGLE_API_KEY=your_google_api_key
   ```

2. Replace `your_github_personal_access_token` with a GitHub personal access token. You can generate one [here](https://github.com/settings/tokens).

3. Replace `your_google_api_key` with your Google API key for accessing the Gemini AI model. You can obtain one from the [Google Cloud Console](https://console.cloud.google.com/).

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter a GitHub repository URL in the sidebar.

4. Click "Fetch Repository" to load the repository contents.

5. Select a file from the dropdown menu to analyze or ask questions about.

6. Use the chat interface to ask questions about the selected file or the repository in general.

