import os
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# LangChain and OpenAI specific imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict

# --- 1. SETUP AND CONFIGURATION ---

# Load environment variables from a .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please create a .env file and add your key.")

# Initialize the powerful LLM for parsing and analysis
# We use gpt-4o for its strong reasoning and JSON capabilities
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize the embedding model for semantic comparison
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 2. DEFINE DATA STRUCTURES FOR PARSING ---
# This tells the LLM exactly what format to extract the data into.

class Project(BaseModel):
    title: str = Field(description="The title of the project.")
    description: str = Field(description="A detailed description of the project, including technologies used.")

class ParsedResume(BaseModel):
    summary: str = Field(description="A brief professional summary of the candidate.")
    projects: List[Project] = Field(description="A list of the candidate's key projects.")
    skills: List[str] = Field(description="A list of technical and soft skills mentioned in the resume.")

# --- 3. FLASK APPLICATION SETUP ---

app = Flask(__name__)

# --- 4. HELPER FUNCTIONS ---

def calculate_cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    # Ensure vectors are numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Calculate dot product and norms
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)

# --- 5. API ENDPOINTS ---

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy", "message": "AI Service is running."}), 200

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    """
    Main endpoint to analyze a resume against a job description.
    """
    data = request.get_json()
    if not data or 'resume_text' not in data or 'job_description_text' not in data:
        return jsonify({"error": "Missing 'resume_text' or 'job_description_text' in request body"}), 400

    resume_text = data['resume_text']
    job_description_text = data['job_description_text']

    try:
        # --- STEP 1: INTELLIGENT PARSING OF THE RESUME ---
        parser = JsonOutputParser(pydantic_object=ParsedResume)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at parsing resumes. Extract the key information from the provided text into the specified JSON format. Focus on projects, skills, and a professional summary."),
            ("human", "Here is the resume text:\n\n{resume}\n\n{format_instructions}")
        ])
        
        chain = prompt | llm | parser
        parsed_resume_json = chain.invoke({
            "resume": resume_text,
            "format_instructions": parser.get_format_instructions()
        })

        # --- STEP 2: MULTI-VECTOR REPRESENTATION ---
        # A. Create overall embeddings for a high-level comparison
        resume_vector = embedding_model.embed_query(resume_text)
        jd_vector = embedding_model.embed_query(job_description_text)
        
        # B. Create granular embeddings for each project (if any)
        project_texts = [f"Project: {p['title']}\nDescription: {p['description']}" for p in parsed_resume_json.get('projects', [])]
        
        # --- STEP 3: TWO-LAYER COMPARISON ---
        # A. Calculate the overall similarity score
        overall_similarity = calculate_cosine_similarity(resume_vector, jd_vector)
        # Convert to a percentage and cap at 100 for display
        overall_score = min(100, int(overall_similarity * 100))

        # --- STEP 4: SYNTHESIZE A FINAL REPORT WITH AN LLM ---
        synthesis_prompt_template = """
        You are an expert recruitment analyst. Your task is to provide a detailed analysis comparing a candidate's resume to a job description.

        Here is the context:
        1. Overall Match Score: {score}%
        2. Candidate's Parsed Resume (JSON): {parsed_resume}
        3. Full Job Description: {job_description}

        Based ONLY on the provided context, generate a final report with the following structure in a JSON object:
        - "match_score": The overall score provided.
        - "summary": A 2-3 sentence executive summary explaining why the candidate is a good or poor fit.
        - "strengths": A bulleted list of the candidate's key strengths that align directly with the job description. Mention specific projects as evidence where applicable.
        - "gaps": A bulleted list of potential gaps or areas where the resume does not align with the job description.
        - "talking_points": A few suggested questions the recruiter could ask the candidate during an interview based on their projects and the job requirements.
        
        Be concise, professional, and base your entire analysis on the provided data.
        """
        
        synthesis_prompt = ChatPromptTemplate.from_template(synthesis_prompt_template)
        
        final_analysis_chain = synthesis_prompt | llm | JsonOutputParser()
        
        final_report = final_analysis_chain.invoke({
            "score": overall_score,
            "parsed_resume": str(parsed_resume_json),
            "job_description": job_description_text
        })
        
        return jsonify(final_report), 200

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred during analysis."}), 500

if __name__ == '__main__':
    # Make sure to run on 0.0.0.0 to be accessible from other services (like your Node.js backend)
    # if running in containers.
    app.run(host='0.0.0.0', port=5000)
