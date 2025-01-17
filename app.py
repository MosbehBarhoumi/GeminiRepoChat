import streamlit as st
import requests
import re
import base64
import os
import hashlib
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import torch
from dotenv import load_dotenv
import ast
from radon.raw import analyze
from radon.metrics import h_visit
from radon.visitors import ComplexityVisitor

# Load environment variables
load_dotenv()

# Gemini API configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# Determine the device to use for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize SentenceTransformer model
@st.cache_resource
def load_sentence_transformer():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to(device)
    return model

model_transformer = load_sentence_transformer()

# Initialize Gemini model
@st.cache_resource
def initialize_gemini_model():
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel('gemini-pro')

gemini_model = initialize_gemini_model()

# Repository content caching
class RepoCache:
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_cache_key(self, repo_url, include_extensions, exclude_extensions):
        key = f"{repo_url}_{include_extensions}_{exclude_extensions}"
        return hashlib.md5(key.encode()).hexdigest()

    def get_cached_content(self, cache_key):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return eval(f.read())  # Safely evaluate the stored list of tuples
        return None

    def set_cached_content(self, cache_key, content):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        with open(cache_file, 'w') as f:
            f.write(repr(content))  # Store the list of tuples as a string representation

repo_cache = RepoCache()

# Fetch repository contents from GitHub API
def fetch_repo_contents(repo_url, token=None, include_extensions=None, exclude_extensions=None):
    cache_key = repo_cache.get_cache_key(repo_url, include_extensions, exclude_extensions)

    cached_content = repo_cache.get_cached_content(cache_key)
    if cached_content:
        return cached_content

    match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
    if not match:
        raise ValueError("Invalid GitHub URL")

    owner, repo = match.groups()
    all_content = []

    include_extensions = set(ext.strip().lower() for ext in include_extensions.split(',')) if include_extensions else None
    exclude_extensions = set(ext.strip().lower() for ext in exclude_extensions.split(',')) if exclude_extensions else set()

    def process_contents(path=''):
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        headers = {'Authorization': f'token {token}'} if token else {}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch repository contents: {response.json().get('message', 'Unknown error')}")

        contents = response.json()

        if isinstance(contents, list):
            for item in contents:
                if item['type'] == 'dir':
                    process_contents(item['path'])
                elif item['type'] == 'file':
                    _, ext = os.path.splitext(item['name'])
                    ext = ext.lower()
                    if (include_extensions is None or ext in include_extensions) and ext not in exclude_extensions:
                        file_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{item['path']}"
                        file_response = requests.get(file_url, headers=headers)
                        if file_response.status_code == 200:
                            file_content = base64.b64decode(file_response.json()['content']).decode('utf-8', errors='ignore')
                            all_content.append((item['path'], file_content))
        elif isinstance(contents, dict) and contents['type'] == 'file':
            _, ext = os.path.splitext(path)
            ext = ext.lower()
            if (include_extensions is None or ext in include_extensions) and ext not in exclude_extensions:
                file_content = base64.b64decode(contents['content']).decode('utf-8', errors='ignore')
                all_content.append((path, file_content))

    process_contents()
    repo_cache.set_cached_content(cache_key, all_content)
    return all_content

# Generate a response using the Gemini AI model
def generate_response(model, context, question, file_name):
    prompt = f"""You are an AI assistant helping with a GitHub repository.
Current context:
- The user has selected the file: {file_name}
- Content of the file:

{context}

Please answer the following question about this file or the current context:
{question}

If the question is about the current file or context, answer it directly. If it's a general programming question, you can answer based on your knowledge."""

    response = model.generate_content(prompt)
    return response.text

def analyze_code(file_content, file_name):
    # Basic metrics
    loc = len(file_content.splitlines())
    
    # Use radon for more advanced metrics
    raw_metrics = analyze(file_content)
    
    # Count functions and classes
    try:
        tree = ast.parse(file_content)
        function_count = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        class_count = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    except SyntaxError:
        function_count = class_count = "N/A (Syntax Error)"
    
    # Calculate cyclomatic complexity
    try:
        cv = ComplexityVisitor.from_code(file_content)
        complexities = [item.complexity for item in cv.functions]
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0
    except:
        avg_complexity = "N/A"
    
    # Calculate Halstead metrics
    try:
        h_visit_result = h_visit(file_content)
        halstead_volume = round(h_visit_result.total.volume, 2)
        halstead_difficulty = round(h_visit_result.total.difficulty, 2)
    except:
        halstead_volume = halstead_difficulty = "N/A"
    
    # Identify potential code smells
    code_smells = []
    if loc > 300:
        code_smells.append("File might be too long (> 300 lines)")
    if isinstance(avg_complexity, (int, float)) and avg_complexity > 10:
        code_smells.append("Average function complexity is high (> 10)")
    
    # Instead of using raw_metrics.comment_lines, we'll calculate it manually
    comment_lines = sum(1 for line in file_content.splitlines() if line.strip().startswith('#'))
    if comment_lines / loc < 0.1:
        code_smells.append("Low comment ratio (< 10% of code)")
    
    return {
        "File Name": file_name,
        "Lines of Code": loc,
        "Function Count": function_count,
        "Class Count": class_count,
        "Average Cyclomatic Complexity": avg_complexity,
        "Comment Lines": comment_lines,
        "Logical Lines of Code": raw_metrics.sloc,
        "Halstead Volume": halstead_volume,
        "Halstead Difficulty": halstead_difficulty,
        "Potential Code Smells": code_smells
    }
# Main application logic
def main():
    st.title("GitHub Repository Chat with Gemini AI")

    # Sidebar for repository information
    st.sidebar.header("Repository Information")
    repo_url = st.sidebar.text_input("GitHub Repository URL", key="repo_url_input")
    include_extensions = st.sidebar.text_input("Include Extensions (comma-separated)", "py,md,txt", key="include_extensions_input")
    exclude_extensions = st.sidebar.text_input("Exclude Extensions (comma-separated)", "gitignore", key="exclude_extensions_input")

    # Fetch repository contents when the "Fetch Repository" button is clicked
    if st.sidebar.button("Fetch Repository", key="fetch_repo_button"):
        if repo_url:
            with st.spinner("Fetching repository contents..."):
                try:
                    st.session_state.repo_contents = fetch_repo_contents(repo_url, GITHUB_TOKEN, include_extensions, exclude_extensions)
                    st.session_state.selected_file = None  # Reset selected file
                    st.success("Repository contents fetched successfully!")
                except Exception as e:
                    st.error(f"Error fetching repository: {str(e)}")
        else:
            st.warning("Please enter a GitHub repository URL.")

    # Display file selection and chat interface
    if "repo_contents" in st.session_state:
        st.sidebar.header("Select a File to Ask Questions")
        file_names = [file[0] for file in st.session_state.repo_contents]
        st.session_state.selected_file = st.sidebar.selectbox("Choose a file:", options=file_names)

        # Display currently selected file
        st.write(f"Currently selected file: **{st.session_state.selected_file}**")

        # New code analysis section
        if st.session_state.selected_file:
            selected_file_content = next((file[1] for file in st.session_state.repo_contents if file[0] == st.session_state.selected_file), None)
            if selected_file_content:
                st.subheader("Code Analysis")
                try:
                    analysis_results = analyze_code(selected_file_content, st.session_state.selected_file)
                    
                    # Display basic metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Lines of Code", analysis_results["Lines of Code"])
                    col2.metric("Functions", analysis_results["Function Count"])
                    col3.metric("Classes", analysis_results["Class Count"])
                    
                    # Display complexity metrics
                    st.write("Complexity Metrics:")
                    st.write(f"- Average Cyclomatic Complexity: {analysis_results['Average Cyclomatic Complexity']}")
                    st.write(f"- Halstead Volume: {analysis_results['Halstead Volume']}")
                    st.write(f"- Halstead Difficulty: {analysis_results['Halstead Difficulty']}")
                    
                    # Display code smells
                    if analysis_results["Potential Code Smells"]:
                        st.write("Potential Code Smells:")
                        for smell in analysis_results["Potential Code Smells"]:
                            st.write(f"- {smell}")
                    else:
                        st.write("No significant code smells detected.")
                    
                    st.write("Note: These metrics are general indicators. Always consider the context of your specific project when interpreting them.")
                except SyntaxError:
                    st.warning("Unable to analyze this file. It may not be a valid Python file or may contain syntax errors.")
                except Exception as e:
                    st.error(f"An error occurred during code analysis: {str(e)}")

        # Initialize chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get user question and display it in the chat
        if question := st.chat_input("Ask a question about the selected file"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Get the selected file content
            selected_file_content = next((file[1] for file in st.session_state.repo_contents if file[0] == st.session_state.selected_file), None)

            if selected_file_content:
                with st.spinner("Generating response..."):
                    # Generate a response using the Gemini model
                    response = generate_response(gemini_model, selected_file_content, question, st.session_state.selected_file)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.chat_message("assistant").markdown(response)
            else:
                st.warning("No file selected or file content not found.")

    # Add a footer to the application
    st.sidebar.write("Chatbot powered by Gemini AI")

if __name__ == "__main__":
    main()