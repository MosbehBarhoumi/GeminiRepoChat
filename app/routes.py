from flask import Blueprint, render_template, request
from .github_fetcher import fetch_repo_contents
from .embedding import get_relevant_code_chunks
from .gemini import initialize_model, get_chat_response
from .rate_limiter import rate_limited

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        repo_url = request.form['repo_url']
        token = request.form.get('github_token')
        include_extensions = request.form.get('include_extensions')
        exclude_extensions = request.form.get('exclude_extensions')
        content_filter = request.form.get('content_filter')
        user_prompt = request.form.get('gemini_prompt')
        api_key = request.form.get('gemini_api_key')

        try:
            # Fetch repository contents
            content = rate_limited(fetch_repo_contents)(repo_url, token, include_extensions, exclude_extensions, content_filter)

            # Get relevant code chunks
            relevant_chunks = get_relevant_code_chunks(content, user_prompt)

            # Prepare the final prompt for Gemini
            final_prompt = f"Here are the most relevant parts of the code:\n\n{''.join(relevant_chunks)}\n\nUser's question: {user_prompt}"

            # Call Gemini API
            model = initialize_model(api_key)
            gemini_response = get_chat_response(model, final_prompt)

            return render_template('index.html', gemini_response=gemini_response, relevant_chunks=relevant_chunks)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('index.html', error=error_message)

    return render_template('index.html')
