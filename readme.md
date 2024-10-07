# GitChat

## Overview

This Flask application allows users to fetch code from GitHub repositories and interact with the Gemini API to generate responses based on relevant code snippets. Users can filter the code by file extensions and content, making it easier to find the information they need.

## Features

- **Code Fetching**: Fetches and caches code from specified GitHub repositories.
- **Filtering Options**: Allows users to include or exclude specific file extensions and filter code content.
- **Relevant Code Extraction**: Uses sentence transformers to extract the most relevant code snippets based on user queries.
- **Gemini Chat Integration**: Interacts with the Gemini API to generate responses to user questions based on relevant code chunks.

## Requirements

- Python 3.7 or higher
- Flask
- Requests
- Google Generative AI
- Sentence Transformers
- NumPy

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/repository-name.git
   ```

2. Change into the project directory:

   ```bash
   cd repository-name
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:

   ```bash
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Fill in the required fields:
   - **Repository URL**: The URL of the GitHub repository you want to fetch code from.
   - **GitHub Token**: (Optional) A personal access token for private repositories.
   - **Include Extensions**: Comma-separated list of file extensions to include (e.g., `.py,.js`).
   - **Exclude Extensions**: Comma-separated list of file extensions to exclude (e.g., `.md`).
   - **Content Filter**: (Optional) A keyword to filter the content.
   - **Gemini Prompt**: Your question or prompt related to the code.
   - **Gemini API Key**: Your API key for the Gemini chat service.

4. Click the submit button to fetch the relevant code chunks and get a response from the Gemini API.

## Caching

Fetched repository content is cached to improve performance and reduce the number of API calls. Cached content is stored in the `./cache` directory.

