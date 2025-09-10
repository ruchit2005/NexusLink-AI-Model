import os
import asyncio
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import logging
from datetime import datetime

import time
import json
import re

# LangChain and OpenAI specific imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- SETUP AND CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# Use more capable models for better analysis
llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
llm_analysis = ChatOpenAI(model="gpt-4o", temperature=0.2)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

app = Flask(__name__)

# --- DATA STRUCTURES MATCHING BACKEND SCHEMA ---

class Education(BaseModel):
    degree: str
    institution: str
    startYear: int
    endYear: int

class WorkExperience(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    startDate: Optional[str] = None  # Date string
    endDate: Optional[str] = None    # Date string
    description: Optional[str] = None

class Extracurricular(BaseModel):
    role: Optional[str] = None
    organization: Optional[str] = None
    description: Optional[str] = None

class Project(BaseModel):
    title: str
    duration: Optional[str] = None
    link: Optional[str] = None
    techStack: List[str] = []
    description: Optional[str] = None

class Training(BaseModel):
    title: Optional[str] = None
    issuer: Optional[str] = None
    date: Optional[str] = None  # Date string
    description: Optional[str] = None

class Portfolio(BaseModel):
    github: Optional[str] = None
    leetcode: Optional[str] = None
    linkedin: Optional[str] = None
    otherLinks: List[str] = []

class Accomplishment(BaseModel):
    title: Optional[str] = None
    issuer: Optional[str] = None
    description: Optional[str] = None
    link: Optional[str] = None

class StudentProfile(BaseModel):
    # NFC mapping (optional)
    idcard_uid: Optional[str] = None
    
    # Basic info - matching MongoDB schema exactly
    name: str
    email: str
    phone: Optional[str] = None
    gender: str  # 'male', 'female', 'other' - required in schema
    location: Optional[str] = None
    
    # Career info
    careerObjective: Optional[str] = None
    
    # Resume download link
    resumeUrl: Optional[str] = None
    
    # Nested schemas
    education: List[Education] = []
    workExperience: List[WorkExperience] = []
    extracurriculars: List[Extracurricular] = []
    trainings: List[Training] = []
    projects: List[Project] = []
    skills: List[str] = []
    portfolio: Optional[Portfolio] = None
    accomplishments: List[Accomplishment] = []

class CompanyProfile(BaseModel):
    company_name: str = Field(description="Company name")
    primary_roles: List[str] = Field(description="Main roles they're hiring for")
    required_skills: List[str] = Field(description="Key technical skills they need")
    preferred_skills: List[str] = Field(default=[], description="Nice-to-have skills")
    preferred_majors: List[str] = Field(description="Preferred academic backgrounds")
    company_culture: List[str] = Field(description="Company culture keywords")
    team_focus: List[str] = Field(description="Specific teams or projects they're focused on")
    experience_level: Optional[str] = Field(default="entry", description="entry, mid, senior")

class MatchAnalysis(BaseModel):
    match_percentage: int = Field(description="Overall match percentage (0-100)")
    skill_alignment_score: int = Field(description="How well student's skills match requirements (0-100)")
    project_relevance_score: int = Field(description="How relevant are the student's projects (0-100)")
    experience_fit_score: int = Field(description="How well does their experience fit the role (0-100)")
    cultural_fit_score: int = Field(description="How well they might fit the company culture (0-100)")
    strengths: List[str] = Field(description="Key strengths of this student for this role")
    weaknesses: List[str] = Field(description="Areas where the student might need development")
    reasoning: str = Field(description="Detailed explanation of the match calculation")

class InstantMatchSummary(BaseModel):
    student_name: str
    match_percentage: int
    headline: str = Field(description="One-line powerful headline about the match")
    key_alignment: str = Field(description="Most compelling alignment point")
    standout_skill: str = Field(description="Student's most relevant skill with proof")
    suggested_icebreaker: str = Field(description="Specific conversation starter for recruiter")
    talking_points: List[str] = Field(description="3-4 specific things to discuss")
    red_flags: List[str] = Field(description="Any potential concerns to address")

# --- LLM-POWERED ANALYSIS FUNCTIONS ---

def analyze_match_with_llm(student_profile: StudentProfile, company_profile: CompanyProfile) -> MatchAnalysis:
    """Comprehensive LLM analysis with detailed scoring."""
    
    analysis_prompt = ChatPromptTemplate.from_template("""
    You are a senior technical recruiter with 15+ years of experience in the tech industry. Analyze this student-company match comprehensively and objectively.

    STUDENT PROFILE:
    Name: {name}
    Gender: {gender}
    Location: {location}
    Career Objective: {career_objective}
    Education: {education}
    Skills: {skills}
    Projects: {projects}
    Work Experience: {experience}
    Extracurriculars: {extracurriculars}
    Accomplishments: {accomplishments}
    Trainings: {trainings}

    COMPANY REQUIREMENTS:
    Company: {company_name}
    Roles: {roles}
    Required Skills: {required_skills}
    Preferred Skills: {preferred_skills}
    Preferred Majors: {preferred_majors}
    Company Culture: {culture}
    Team Focus: {team_focus}
    Experience Level: {experience_level}

    SCORING GUIDELINES:
    - Skill Alignment (35% weight): Count exact matches. "Python" = "Python", "React" = "React.js". Award partial credit for related skills.
    - Project Relevance (30% weight): Evaluate how projects demonstrate required skills and company focus areas.
    - Experience Fit (25% weight): Match experience level to role requirements. Entry=internships OK, Mid=2+ years, Senior=5+ years.
    - Cultural Fit (10% weight): Assess leadership, teamwork, and alignment with company values.

    MATCH PERCENTAGE RANGES:
    - 90-100%: Exceptional fit - has 80%+ required skills + strong relevant experience/projects
    - 80-89%: Excellent match - has 70%+ required skills + good projects/experience
    - 70-79%: Strong candidate - has 60%+ required skills + decent portfolio
    - 60-69%: Good potential - has 50%+ required skills + shows aptitude
    - 50-59%: Moderate fit - has 40%+ required skills + some relevant work
    - 40-49%: Possible with training - has 30%+ required skills + good foundation
    - Below 40%: Poor fit - major skill gaps

    Be precise with skill counting. List specific strengths with evidence. Identify concrete skill gaps.

    Return ONLY valid JSON:
    {{
        "match_percentage": <number 0-100>,
        "skill_alignment_score": <0-100 based on skill matches>,
        "project_relevance_score": <0-100 based on project-skill alignment>,
        "experience_fit_score": <0-100 based on experience level match>,
        "cultural_fit_score": <0-100 based on cultural indicators>,
        "strengths": ["specific strength with evidence", "another strength"],
        "weaknesses": ["specific skill/experience gap", "another weakness"],
        "reasoning": "Skills: X/Y matched. Projects demonstrate: Z. Experience: suitable for {experience_level}. Culture: shows W traits."
    }}
    """)
    
    chain = analysis_prompt | llm_analysis | JsonOutputParser()
    
    try:
        # Prepare detailed context
        education_text = " | ".join([f"{ed.degree} from {ed.institution} ({ed.startYear}-{ed.endYear})" for ed in student_profile.education]) if student_profile.education else "Not specified"
        skills_text = ", ".join(student_profile.skills) if student_profile.skills else "None listed"
        projects_text = " | ".join([f"{p.title} ({', '.join(p.techStack[:4])}): {p.description[:150] if p.description else 'No description'}" for p in student_profile.projects[:4]]) if student_profile.projects else "None listed"
        experience_text = " | ".join([f"{exp.title} at {exp.company} ({exp.startDate} to {exp.endDate}): {exp.description[:100] if exp.description else ''}" for exp in student_profile.workExperience if exp.title and exp.company]) or "Limited professional experience"
        extracurriculars_text = " | ".join([f"{ext.role} at {ext.organization}: {ext.description[:100] if ext.description else ''}" for ext in student_profile.extracurriculars if ext.role]) if student_profile.extracurriculars else "None listed"
        accomplishments_text = " | ".join([f"{acc.title} from {acc.issuer}: {acc.description[:100] if acc.description else ''}" for acc in student_profile.accomplishments if acc.title]) if student_profile.accomplishments else "None listed"
        trainings_text = " | ".join([f"{tr.title} from {tr.issuer} ({tr.date})" for tr in student_profile.trainings if tr.title]) if student_profile.trainings else "None listed"
        
        raw = chain.invoke({
            "name": student_profile.name,
            "gender": student_profile.gender,
            "location": student_profile.location or "Not specified",
            "career_objective": student_profile.careerObjective or "Not specified",
            "education": education_text,
            "skills": skills_text,
            "projects": projects_text,
            "experience": experience_text,
            "extracurriculars": extracurriculars_text,
            "accomplishments": accomplishments_text,
            "trainings": trainings_text,
            "company_name": company_profile.company_name,
            "roles": ", ".join(company_profile.primary_roles),
            "required_skills": ", ".join(company_profile.required_skills),
            "preferred_skills": ", ".join(company_profile.preferred_skills),
            "preferred_majors": ", ".join(company_profile.preferred_majors),
            "culture": ", ".join(company_profile.company_culture),
            "team_focus": ", ".join(company_profile.team_focus),
            "experience_level": company_profile.experience_level
        })
        
        # Parse and validate result
        if isinstance(raw, str):
            analysis_result = json.loads(raw)
        else:
            analysis_result = raw
            
        # Validate and clamp all scores
        for key in ["match_percentage", "skill_alignment_score", "project_relevance_score", "experience_fit_score", "cultural_fit_score"]:
            score = analysis_result.get(key, 50)
            if isinstance(score, str):
                score_match = re.search(r'(\d+)', score)
                score = int(score_match.group(1)) if score_match else 50
            analysis_result[key] = max(0, min(100, int(score)))
        
        # Ensure required fields exist
        if not isinstance(analysis_result.get("strengths"), list):
            analysis_result["strengths"] = ["Shows technical foundation"]
        if not isinstance(analysis_result.get("weaknesses"), list):
            analysis_result["weaknesses"] = []
        if not analysis_result.get("reasoning"):
            analysis_result["reasoning"] = "LLM analysis completed with comprehensive evaluation"
        
        # Cross-validate overall score with components
        component_scores = [
            analysis_result["skill_alignment_score"] * 0.35,
            analysis_result["project_relevance_score"] * 0.30, 
            analysis_result["experience_fit_score"] * 0.25,
            analysis_result["cultural_fit_score"] * 0.10
        ]
        calculated_score = int(sum(component_scores))
        
        # Adjust if there's a significant discrepancy
        llm_score = analysis_result["match_percentage"]
        if abs(calculated_score - llm_score) > 20:
            logger.info(f"Adjusting match percentage from {llm_score} to {calculated_score} based on component scores")
            analysis_result["match_percentage"] = calculated_score
            
        return MatchAnalysis(**analysis_result)
        
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        # Return minimal analysis if LLM fails
        return MatchAnalysis(
            match_percentage=50,
            skill_alignment_score=50,
            project_relevance_score=50,
            experience_fit_score=50,
            cultural_fit_score=50,
            strengths=["Technical background"],
            weaknesses=["Analysis incomplete due to processing error"],
            reasoning="LLM analysis encountered an error - manual review recommended"
        )

def generate_recruiter_summary_llm(student_profile: StudentProfile, company_profile: CompanyProfile, match_analysis: MatchAnalysis) -> InstantMatchSummary:
    """Generate detailed recruiter summary using LLM."""
    
    summary_prompt = ChatPromptTemplate.from_template("""
    Create a compelling recruiter summary for a career fair interaction. This student just approached {company_name}'s booth.

    STUDENT: {student_name} ({gender}, from {location})
    MATCH: {match_percentage}%
    KEY STRENGTHS: {strengths}
    MAIN CONCERNS: {weaknesses}
    PROJECTS: {projects}
    SKILLS: {skills}
    EXPERIENCE: {experience}
    BACKGROUND: {education}

    Create an actionable summary for the recruiter to have a meaningful conversation.

    Return ONLY valid JSON:
    {{
        "headline": "<compelling one-liner about this candidate>",
        "key_alignment": "<most impressive reason they fit this role>",
        "standout_skill": "<their best skill/achievement with specific evidence>",
        "suggested_icebreaker": "<specific, personal question to start conversation>",
        "talking_points": ["<actionable point 1>", "<actionable point 2>", "<actionable point 3>", "<actionable point 4>"],
        "red_flags": ["<specific concern 1>", "<specific concern 2>"]
    }}

    Guidelines:
    - Reference specific projects, companies, or technologies by name
    - Make icebreaker personal and engaging
    - Include actionable talking points
    - Keep red flags constructive, not dismissive
    """)

    parser = JsonOutputParser()
    chain = summary_prompt | llm_fast | parser
    
    try:
        # Extract key info for the prompt
        projects_summary = ", ".join([f"{p.title} ({', '.join(p.techStack[:3])})" for p in student_profile.projects[:3]]) if student_profile.projects else "No projects listed"
        skills_summary = ", ".join(student_profile.skills[:8]) if student_profile.skills else "Skills not specified"
        experience_summary = ", ".join([f"{exp.title} at {exp.company}" for exp in student_profile.workExperience if exp.title and exp.company][:2]) or "Limited work experience"
        education_summary = f"{student_profile.education[0].degree} from {student_profile.education[0].institution}" if student_profile.education else "Education not specified"
        
        summary_result = chain.invoke({
            "company_name": company_profile.company_name,
            "student_name": student_profile.name,
            "gender": student_profile.gender,
            "location": student_profile.location or "Location not specified",
            "match_percentage": match_analysis.match_percentage,
            "strengths": ", ".join(match_analysis.strengths[:3]),
            "weaknesses": ", ".join(match_analysis.weaknesses[:2]) if match_analysis.weaknesses else "No major concerns",
            "projects": projects_summary,
            "skills": skills_summary,
            "experience": experience_summary,
            "education": education_summary
        })
        
        # Validate and provide fallbacks
        return InstantMatchSummary(
            student_name=student_profile.name,
            match_percentage=match_analysis.match_percentage,
            headline=summary_result.get("headline", f"{match_analysis.match_percentage}% match - {student_profile.name} brings technical skills to {company_profile.company_name}"),
            key_alignment=summary_result.get("key_alignment", match_analysis.strengths[0] if match_analysis.strengths else "Technical background alignment"),
            standout_skill=summary_result.get("standout_skill", f"Proficient in {student_profile.skills[0] if student_profile.skills else 'programming'} with hands-on experience"),
            suggested_icebreaker=summary_result.get("suggested_icebreaker", f"Tell me about your experience with {student_profile.projects[0].title if student_profile.projects else 'your technical projects'}"),
            talking_points=summary_result.get("talking_points", match_analysis.strengths[:3] + ["Technical interests", "Career goals"])[:4],
            red_flags=summary_result.get("red_flags", match_analysis.weaknesses[:2] if match_analysis.weaknesses else [])
        )
        
    except Exception as e:
        logger.error(f"LLM summary generation failed: {str(e)}")
        # Generate fallback summary
        return InstantMatchSummary(
            student_name=student_profile.name,
            match_percentage=match_analysis.match_percentage,
            headline=f"{match_analysis.match_percentage}% match - {student_profile.name} shows technical potential",
            key_alignment=match_analysis.strengths[0] if match_analysis.strengths else "Technical background",
            standout_skill=f"Strong {student_profile.skills[0] if student_profile.skills else 'programming'} skills",
            suggested_icebreaker=f"What interests you most about {company_profile.primary_roles[0] if company_profile.primary_roles else 'this role'}?",
            talking_points=match_analysis.strengths[:4] if len(match_analysis.strengths) >= 4 else match_analysis.strengths + ["Academic performance", "Technical interests", "Career goals", "Project experience"][:4-len(match_analysis.strengths)],
            red_flags=match_analysis.weaknesses[:2] if match_analysis.weaknesses else []
        )

def calculate_semantic_similarity(student_profile: StudentProfile, company_profile: CompanyProfile) -> float:
    """Calculate semantic similarity using embeddings."""
    try:
        # Create comprehensive text representations
        student_text = f"""
        Student Profile: {student_profile.name}
        Career Goal: {student_profile.careerObjective or 'Software development'}
        Skills: {', '.join(student_profile.skills)}
        Education: {' | '.join([f"{ed.degree} from {ed.institution}" for ed in student_profile.education])}
        Projects: {' | '.join([f"{p.title}: {', '.join(p.techStack)} - {p.description[:100] if p.description else ''}" for p in student_profile.projects[:3]])}
        Experience: {' | '.join([f"{exp.title} at {exp.company}" for exp in student_profile.workExperience if exp.title and exp.company])}
        """
        
        company_text = f"""
        Company: {company_profile.company_name}
        Hiring for: {', '.join(company_profile.primary_roles)}
        Required Skills: {', '.join(company_profile.required_skills)}
        Preferred Skills: {', '.join(company_profile.preferred_skills)}
        Team Focus: {', '.join(company_profile.team_focus)}
        Culture: {', '.join(company_profile.company_culture)}
        Experience Level: {company_profile.experience_level}
        """
        
        # Get embeddings
        student_emb = np.array(embedding_model.embed_query(student_text))
        company_emb = np.array(embedding_model.embed_query(company_text))
        
        # Cosine similarity
        similarity = np.dot(student_emb, company_emb) / (np.linalg.norm(student_emb) * np.linalg.norm(company_emb))
        
        return max(0.0, min(1.0, float(similarity)))
        
    except Exception as e:
        logger.warning(f"Semantic similarity calculation failed: {e}")
        return 0.6  # Neutral-positive default

# --- API ENDPOINTS ---

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for NexusLink AI service."""
    return jsonify({
        "status": "healthy", 
        "service": "NexusLink AI - LLM-Powered Analysis",
        "message": "Ready for intelligent career fair matching!",
        "analysis_method": "LLM-Only",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/detailed-analysis', methods=['POST'])
def detailed_analysis():
    """
    Get detailed match analysis using LLM with comprehensive scoring.
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'student_profile' not in data or 'company_profile' not in data:
            return jsonify({"error": "Missing 'student_profile' or 'company_profile' in request body"}), 400
        
        try:
            student_profile = StudentProfile(**data['student_profile'])
        except Exception as e:
            return jsonify({"error": f"Invalid student_profile format: {str(e)}"}), 400
        
        try:
            company_profile = CompanyProfile(**data['company_profile'])
        except Exception as e:
            return jsonify({"error": f"Invalid company_profile format: {str(e)}"}), 400
        
        logger.info(f"Processing LLM analysis for {student_profile.name} ({student_profile.gender}) at {company_profile.company_name}")
        
        # Get comprehensive LLM analysis
        match_analysis = analyze_match_with_llm(student_profile, company_profile)
        
        # Generate recruiter summary
        match_summary = generate_recruiter_summary_llm(student_profile, company_profile, match_analysis)
        
        # Calculate semantic similarity for additional insight
        semantic_similarity = calculate_semantic_similarity(student_profile, company_profile)
        
        processing_time = round((time.time() - start_time) * 1000)
        
        response = {
            "success": True,
            "student_name": student_profile.name,
            "student_gender": student_profile.gender,
            "company_name": company_profile.company_name,
            "match_analysis": match_analysis.dict(),
            "recruiter_summary": match_summary.dict(),
            "semantic_similarity": round(semantic_similarity, 3),
            "processing_time_ms": processing_time,
            "analysis_method": "LLM-Powered",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"LLM analysis completed: {match_analysis.match_percentage}% match in {processing_time}ms")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Detailed analysis error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False, 
            "error": "Analysis failed",
            "details": str(e) if app.debug else "Please check your input data and try again"
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Starting NexusLink AI Service (LLM-Only Analysis) on {host}:{port}")
    logger.info("Available endpoints:")
    logger.info("  POST /detailed-analysis - Comprehensive LLM-powered analysis")
    logger.info("  GET /health - Health check")
    
    app.run(host=host, port=port, debug=os.getenv('DEBUG', 'False').lower() == 'true')