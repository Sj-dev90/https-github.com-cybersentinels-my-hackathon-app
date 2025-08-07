import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- INITIALIZE FLASK APP AND CORS ---
app = Flask(__name__)
# This allows your HTML file to make requests to this server
CORS(app) 

# --- CONFIGURE GEMINI API ---
try:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    # This will stop the app if the API key is not configured
    raise SystemExit(f"Error configuring Gemini API: {e}")

# --- PROMPT CREATION FUNCTION (from your Streamlit code) ---
def create_prompt(style, text_input):
    """Creates the instruction prompt for the Gemini model."""
    final_prompt = f"""
    You are an expert linguistic transformer. Your task is to accurately and creatively convert the user's text into the specified style: '{style}'.
    Maintain the core message, intent, and meaning of the original text. Do not add any introductory phrases like "Transformed Text:". Just provide the direct transformation.

    Original Text:
    ---
    {text_input}
    ---

    Transformed Text:
    """
    return final_prompt

# --- API ENDPOINT FOR TRANSFORMATION ---
@app.route('/transform', methods=['POST'])
def transform_text():
    """
    Receives text and a style, sends it to Gemini, and returns the transformation.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text_input = data.get('text')
    style = data.get('style')

    if not text_input or not style:
        return jsonify({"error": "Missing 'text' or 'style' in request"}), 400

    try:
        prompt = create_prompt(style, text_input)
        response = model.generate_content(prompt)

        # Send the AI's text back as a JSON response
        return jsonify({"transformed_text": response.text})

    except Exception as e:
        # Handle potential errors from the API call
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- RUN THE FLASK APP ---
if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5000
    app.run(debug=True)