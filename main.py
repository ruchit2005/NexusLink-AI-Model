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

# --- NEW: PERSON-TO-PERSON COMPATIBILITY MODELS ---

class CompatibilityAnalysis(BaseModel):
    compatibility_percentage: int = Field(description="Overall compatibility percentage (0-100)")
    technical_synergy_score: int = Field(description="How well their technical skills complement (0-100)")
    collaboration_potential_score: int = Field(description="Potential for effective collaboration (0-100)")
    learning_exchange_score: int = Field(description="Potential for mutual learning (0-100)")
    project_alignment_score: int = Field(description="How well their project interests align (0-100)")
    shared_interests: List[str] = Field(description="Common interests, skills, or goals")
    complementary_strengths: List[str] = Field(description="How they complement each other")
    potential_challenges: List[str] = Field(description="Potential areas of conflict or mismatch")
    collaboration_opportunities: List[str] = Field(description="Specific ways they could work together")
    reasoning: str = Field(description="Detailed explanation of the compatibility assessment")

class PersonCompatibilitySummary(BaseModel):
    person1_name: str
    person2_name: str
    compatibility_percentage: int
    relationship_type: str = Field(description="Best type of relationship: 'study_partners', 'project_collaborators', 'mentorship', 'peer_learning'")
    headline: str = Field(description="One-line summary of their compatibility")
    key_synergy: str = Field(description="Most compelling reason they should connect")
    mutual_benefits: List[str] = Field(description="What each person brings to the other")
    suggested_collaboration: str = Field(description="Specific project or activity they could do together")
    conversation_starters: List[str] = Field(description="3-4 topics they could discuss")
    watch_out_for: List[str] = Field(description="Potential areas to be mindful of")

# --- EXISTING LLM-POWERED ANALYSIS FUNCTIONS ---

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

# --- NEW: PERSON-TO-PERSON COMPATIBILITY FUNCTIONS ---

def analyze_person_compatibility_llm(person1: StudentProfile, person2: StudentProfile) -> CompatibilityAnalysis:
    """Analyze compatibility between two people using LLM."""
    
    compatibility_prompt = ChatPromptTemplate.from_template("""
    You are an expert in peer relationships, collaboration, and team dynamics. Analyze the compatibility between these two people for potential collaboration, mentorship, or peer learning relationships.

    PERSON 1:
    Name: {person1_name}
    Gender: {person1_gender}
    Location: {person1_location}
    Career Goal: {person1_career}
    Education: {person1_education}
    Skills: {person1_skills}
    Projects: {person1_projects}
    Experience: {person1_experience}
    Extracurriculars: {person1_extracurriculars}
    Accomplishments: {person1_accomplishments}

    PERSON 2:
    Name: {person2_name}
    Gender: {person2_gender}
    Location: {person2_location}
    Career Goal: {person2_career}
    Education: {person2_education}
    Skills: {person2_skills}
    Projects: {person2_projects}
    Experience: {person2_experience}
    Extracurriculars: {person2_extracurriculars}
    Accomplishments: {person2_accomplishments}

    SCORING GUIDELINES:
    - Technical Synergy (30%): How their skills complement or overlap beneficially
    - Collaboration Potential (25%): Shared interests, compatible working styles
    - Learning Exchange (25%): What each can teach/learn from the other
    - Project Alignment (20%): Similar project interests or complementary expertise

    COMPATIBILITY RANGES:
    - 90-100%: Exceptional match - highly complementary skills, shared goals, great collaboration potential
    - 80-89%: Excellent compatibility - strong synergy, mutual learning opportunities
    - 70-79%: Good match - several shared interests, decent collaboration potential
    - 60-69%: Moderate compatibility - some common ground, limited but positive interaction potential
    - 50-59%: Basic compatibility - minimal overlap but no major conflicts
    - Below 50%: Limited compatibility - different paths, minimal synergy

    Analyze both shared interests AND complementary differences that create value.

    Return ONLY valid JSON:
    {{
        "compatibility_percentage": <number 0-100>,
        "technical_synergy_score": <0-100>,
        "collaboration_potential_score": <0-100>,
        "learning_exchange_score": <0-100>,
        "project_alignment_score": <0-100>,
        "shared_interests": ["specific shared interest 1", "shared interest 2"],
        "complementary_strengths": ["how person1 complements person2", "how person2 complements person1"],
        "potential_challenges": ["potential area of mismatch", "another challenge"],
        "collaboration_opportunities": ["specific way they could work together", "another opportunity"],
        "reasoning": "Detailed analysis of their compatibility based on skills, goals, and personalities."
    }}
    """)
    
    chain = compatibility_prompt | llm_analysis | JsonOutputParser()
    
    try:
        # Helper function to extract profile summary
        def extract_profile_summary(profile: StudentProfile, prefix: str) -> Dict[str, str]:
            education_text = " | ".join([f"{ed.degree} from {ed.institution} ({ed.startYear}-{ed.endYear})" for ed in profile.education]) if profile.education else "Not specified"
            skills_text = ", ".join(profile.skills) if profile.skills else "None listed"
            projects_text = " | ".join([f"{p.title} ({', '.join(p.techStack[:3])}): {p.description[:100] if p.description else 'No description'}" for p in profile.projects[:3]]) if profile.projects else "None listed"
            experience_text = " | ".join([f"{exp.title} at {exp.company}" for exp in profile.workExperience if exp.title and exp.company][:2]) or "Limited work experience"
            extracurriculars_text = " | ".join([f"{ext.role} at {ext.organization}" for ext in profile.extracurriculars if ext.role][:2]) if profile.extracurriculars else "None listed"
            accomplishments_text = " | ".join([f"{acc.title} from {acc.issuer}" for acc in profile.accomplishments if acc.title][:2]) if profile.accomplishments else "None listed"
            
            return {
                f"{prefix}_name": profile.name,
                f"{prefix}_gender": profile.gender,
                f"{prefix}_location": profile.location or "Not specified",
                f"{prefix}_career": profile.careerObjective or "Not specified",
                f"{prefix}_education": education_text,
                f"{prefix}_skills": skills_text,
                f"{prefix}_projects": projects_text,
                f"{prefix}_experience": experience_text,
                f"{prefix}_extracurriculars": extracurriculars_text,
                f"{prefix}_accomplishments": accomplishments_text
            }
        
        # Extract summaries for both people
        person1_data = extract_profile_summary(person1, "person1")
        person2_data = extract_profile_summary(person2, "person2")
        
        # Combine data for prompt
        prompt_data = {**person1_data, **person2_data}
        
        raw = chain.invoke(prompt_data)
        
        # Parse and validate result
        if isinstance(raw, str):
            compatibility_result = json.loads(raw)
        else:
            compatibility_result = raw
            
        # Validate and clamp all scores
        for key in ["compatibility_percentage", "technical_synergy_score", "collaboration_potential_score", "learning_exchange_score", "project_alignment_score"]:
            score = compatibility_result.get(key, 50)
            if isinstance(score, str):
                score_match = re.search(r'(\d+)', score)
                score = int(score_match.group(1)) if score_match else 50
            compatibility_result[key] = max(0, min(100, int(score)))
        
        # Ensure required fields exist
        for field, default in [
            ("shared_interests", ["Technical backgrounds"]),
            ("complementary_strengths", ["Different perspectives"]),
            ("potential_challenges", []),
            ("collaboration_opportunities", ["Study together"]),
            ("reasoning", "Compatibility analysis completed")
        ]:
            if not isinstance(compatibility_result.get(field), list if field != "reasoning" else str):
                compatibility_result[field] = default
        
        return CompatibilityAnalysis(**compatibility_result)
        
    except Exception as e:
        logger.error(f"Person compatibility analysis failed: {e}")
        # Return fallback analysis
        return CompatibilityAnalysis(
            compatibility_percentage=60,
            technical_synergy_score=60,
            collaboration_potential_score=60,
            learning_exchange_score=60,
            project_alignment_score=60,
            shared_interests=["Technical backgrounds", "Similar academic paths"],
            complementary_strengths=["Different perspectives", "Diverse skill sets"],
            potential_challenges=["Analysis incomplete due to processing error"],
            collaboration_opportunities=["Study together", "Share knowledge"],
            reasoning="LLM compatibility analysis encountered an error - manual review recommended"
        )

def generate_compatibility_summary_llm(person1: StudentProfile, person2: StudentProfile, compatibility_analysis: CompatibilityAnalysis) -> PersonCompatibilitySummary:
    """Generate actionable compatibility summary using LLM."""
    
    summary_prompt = ChatPromptTemplate.from_template("""
    Create an actionable compatibility summary for two people who might want to connect, collaborate, or learn from each other.

    PERSON 1: {person1_name} ({person1_gender})
    Skills: {person1_skills}
    Projects: {person1_projects}
    Goal: {person1_career}

    PERSON 2: {person2_name} ({person2_gender})
    Skills: {person2_skills}
    Projects: {person2_projects}
    Goal: {person2_career}

    COMPATIBILITY: {compatibility_percentage}%
    SHARED INTERESTS: {shared_interests}
    COMPLEMENTARY STRENGTHS: {complementary_strengths}
    OPPORTUNITIES: {opportunities}

    Determine the best relationship type and create actionable guidance for connection.

    Return ONLY valid JSON:
    {{
        "relationship_type": "<one of: study_partners, project_collaborators, mentorship, peer_learning>",
        "headline": "<compelling one-liner about their compatibility>",
        "key_synergy": "<most compelling reason they should connect>",
        "mutual_benefits": ["<what person1 offers person2>", "<what person2 offers person1>"],
        "suggested_collaboration": "<specific project or activity they could do together>",
        "conversation_starters": ["<topic 1>", "<topic 2>", "<topic 3>", "<topic 4>"],
        "watch_out_for": ["<potential challenge 1>", "<potential challenge 2>"]
    }}

    Guidelines:
    - Be specific about mutual benefits
    - Suggest concrete collaboration ideas
    - Include diverse conversation starters
    - Keep challenges constructive
    """)

    parser = JsonOutputParser()
    chain = summary_prompt | llm_fast | parser
    
    try:
        # Extract key information
        person1_skills = ", ".join(person1.skills[:6]) if person1.skills else "Skills not specified"
        person1_projects = ", ".join([p.title for p in person1.projects[:3]]) if person1.projects else "No projects listed"
        person2_skills = ", ".join(person2.skills[:6]) if person2.skills else "Skills not specified"
        person2_projects = ", ".join([p.title for p in person2.projects[:3]]) if person2.projects else "No projects listed"
        
        summary_result = chain.invoke({
            "person1_name": person1.name,
            "person1_gender": person1.gender,
            "person1_skills": person1_skills,
            "person1_projects": person1_projects,
            "person1_career": person1.careerObjective or "Not specified",
            "person2_name": person2.name,
            "person2_gender": person2.gender,
            "person2_skills": person2_skills,
            "person2_projects": person2_projects,
            "person2_career": person2.careerObjective or "Not specified",
            "compatibility_percentage": compatibility_analysis.compatibility_percentage,
            "shared_interests": ", ".join(compatibility_analysis.shared_interests[:3]),
            "complementary_strengths": ", ".join(compatibility_analysis.complementary_strengths[:2]),
            "opportunities": ", ".join(compatibility_analysis.collaboration_opportunities[:2])
        })
        
        # Validate and provide fallbacks
        return PersonCompatibilitySummary(
            person1_name=person1.name,
            person2_name=person2.name,
            compatibility_percentage=compatibility_analysis.compatibility_percentage,
            relationship_type=summary_result.get("relationship_type", "peer_learning"),
            headline=summary_result.get("headline", f"{compatibility_analysis.compatibility_percentage}% compatibility - {person1.name} and {person2.name} could learn from each other"),
            key_synergy=summary_result.get("key_synergy", compatibility_analysis.shared_interests[0] if compatibility_analysis.shared_interests else "Shared technical interests"),
            mutual_benefits=summary_result.get("mutual_benefits", [f"{person1.name} offers diverse perspective", f"{person2.name} brings unique skills"]),
            suggested_collaboration=summary_result.get("suggested_collaboration", "Work on a joint technical project or study together"),
            conversation_starters=summary_result.get("conversation_starters", ["Technical interests", "Career goals", "Project experiences", "Learning objectives"])[:4],
            watch_out_for=summary_result.get("watch_out_for", compatibility_analysis.potential_challenges[:2] if compatibility_analysis.potential_challenges else [])
        )
        
    except Exception as e:
        logger.error(f"Compatibility summary generation failed: {str(e)}")
        # Generate fallback summary
        return PersonCompatibilitySummary(
            person1_name=person1.name,
            person2_name=person2.name,
            compatibility_percentage=compatibility_analysis.compatibility_percentage,
            relationship_type="peer_learning",
            headline=f"{compatibility_analysis.compatibility_percentage}% compatibility - {person1.name} and {person2.name} have potential for collaboration",
            key_synergy=compatibility_analysis.shared_interests[0] if compatibility_analysis.shared_interests else "Technical backgrounds",
            mutual_benefits=[f"{person1.name} offers unique perspective", f"{person2.name} brings different skills"],
            suggested_collaboration="Study together or collaborate on a project",
            conversation_starters=["Technical skills", "Projects", "Career goals", "Learning interests"],
            watch_out_for=compatibility_analysis.potential_challenges[:2] if compatibility_analysis.potential_challenges else []
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

def calculate_person_semantic_similarity(person1: StudentProfile, person2: StudentProfile) -> float:
    """Calculate semantic similarity between two people using embeddings."""
    try:
        # Helper function to create person text representation
        def create_person_text(profile: StudentProfile) -> str:
            return f"""
            Person: {profile.name}
            Career Goal: {profile.careerObjective or 'Not specified'}
            Skills: {', '.join(profile.skills)}
            Education: {' | '.join([f"{ed.degree} from {ed.institution}" for ed in profile.education])}
            Projects: {' | '.join([f"{p.title}: {', '.join(p.techStack)} - {p.description[:100] if p.description else ''}" for p in profile.projects[:3]])}
            Experience: {' | '.join([f"{exp.title} at {exp.company}" for exp in profile.workExperience if exp.title and exp.company])}
            Extracurriculars: {' | '.join([f"{ext.role} at {ext.organization}" for ext in profile.extracurriculars if ext.role])}
            Interests: {profile.careerObjective or 'Technology'}
            """
        
        # Create text representations
        person1_text = create_person_text(person1)
        person2_text = create_person_text(person2)
        
        # Get embeddings
        person1_emb = np.array(embedding_model.embed_query(person1_text))
        person2_emb = np.array(embedding_model.embed_query(person2_text))
        
        # Cosine similarity
        similarity = np.dot(person1_emb, person2_emb) / (np.linalg.norm(person1_emb) * np.linalg.norm(person2_emb))
        
        return max(0.0, min(1.0, float(similarity)))
        
    except Exception as e:
        logger.warning(f"Person semantic similarity calculation failed: {e}")
        return 0.6  # Neutral-positive default

# --- API ENDPOINTS ---

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for NexusLink AI service."""
    return jsonify({
        "status": "healthy", 
        "service": "NexusLink AI - LLM-Powered Analysis",
        "message": "Ready for intelligent career fair matching and person-to-person compatibility analysis!",
        "analysis_methods": ["LLM-Company-Match", "LLM-Person-Compatibility"],
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

@app.route('/person-compatibility', methods=['POST'])
def person_compatibility():
    """
    Analyze compatibility between two people using LLM.
    Expected payload: { "person1": {...}, "person2": {...} }
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'person1' not in data or 'person2' not in data:
            return jsonify({"error": "Missing 'person1' or 'person2' in request body"}), 400
        
        try:
            person1 = StudentProfile(**data['person1'])
        except Exception as e:
            return jsonify({"error": f"Invalid person1 format: {str(e)}"}), 400
        
        try:
            person2 = StudentProfile(**data['person2'])
        except Exception as e:
            return jsonify({"error": f"Invalid person2 format: {str(e)}"}), 400
        
        logger.info(f"Processing compatibility analysis between {person1.name} ({person1.gender}) and {person2.name} ({person2.gender})")
        
        # Get comprehensive compatibility analysis
        compatibility_analysis = analyze_person_compatibility_llm(person1, person2)
        
        # Generate actionable compatibility summary
        compatibility_summary = generate_compatibility_summary_llm(person1, person2, compatibility_analysis)
        
        # Calculate semantic similarity for additional insight
        semantic_similarity = calculate_person_semantic_similarity(person1, person2)
        
        processing_time = round((time.time() - start_time) * 1000)
        
        response = {
            "success": True,
            "person1_name": person1.name,
            "person1_gender": person1.gender,
            "person2_name": person2.name,
            "person2_gender": person2.gender,
            "compatibility_analysis": compatibility_analysis.dict(),
            "compatibility_summary": compatibility_summary.dict(),
            "semantic_similarity": round(semantic_similarity, 3),
            "processing_time_ms": processing_time,
            "analysis_method": "LLM-Powered-Compatibility",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Compatibility analysis completed: {compatibility_analysis.compatibility_percentage}% compatibility in {processing_time}ms")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Person compatibility analysis error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False, 
            "error": "Compatibility analysis failed",
            "details": str(e) if app.debug else "Please check your input data and try again"
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Starting NexusLink AI Service (Enhanced with Person Compatibility) on {host}:{port}")
    logger.info("Available endpoints:")
    logger.info("  POST /detailed-analysis - Comprehensive LLM-powered student-company analysis")
    logger.info("  POST /person-compatibility - LLM-powered person-to-person compatibility analysis")
    logger.info("  GET /health - Health check")
    
    app.run(host=host, port=port, debug=os.getenv('DEBUG', 'False').lower() == 'true')