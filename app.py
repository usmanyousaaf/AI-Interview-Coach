import streamlit as st
import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import uuid
from groq import Groq
import re
import json
import os
# -------------------- Configuration --------------------
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS with Dark Theme


st.markdown("""
    <style>
    .main {background-color: #121212; color: #ffffff;}
    .stButton>button {background-color: #4A90E2; color: white; border-radius: 8px; padding: 0.5rem 1rem;}
    .question-card {background: #1A1A1A; border-radius: 15px; padding: 2rem; margin: 1.5rem 0; border: 1px solid #333333; animation: fadeIn 0.5s ease-in;}
    .welcome-card {background: #1A1A1A; border-radius: 15px; padding: 2rem; margin: 1.5rem 0; border: 1px solid #4A90E2; animation: glow 2s infinite alternate;}
    .final-report {background: #1A1A1A; border-radius: 15px; padding: 2rem; margin: 1rem 0; border: 1px solid #333333;}
    .feedback-card {background: #2D2D2D; border-left: 4px solid #4A90E2; border-radius: 8px; padding: 1.5rem; margin: 1rem 0;}
    .resource-card {background: #2D2D2D; border-radius: 10px; padding: 1rem; margin: 1rem 0; animation: slideIn 0.5s ease-out;}
    .correct-answer {color: #4CD964; border-left: 4px solid #4CD964; padding-left: 1rem; margin: 1rem 0;}
    .wrong-answer {color: #FF3B30; border-left: 4px solid #FF3B30; padding-left: 1rem; margin: 1rem 0;}
    .topic-chip {display: inline-block; background: #333333; padding: 5px 10px; margin: 5px; border-radius: 15px; font-size: 0.8rem;}
    .stTextInput>div>div>input {background-color: #2D2D2D !important; color: #FFFFFF !important; border-radius: 8px;}
    .stTextArea>div>div>textarea {background-color: #2D2D2D !important; color: #FFFFFF !important; border-radius: 8px;}
    
    /* New Enhanced Styles */
    .report-question { 
        background: linear-gradient(145deg, #1E1E1E 0%, #2D2D2D 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        border-left: 4px solid #4A90E2;
    }
    .report-question:hover {
        transform: translateY(-3px);
    }
    .question-header {
        color: #4A90E2;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #333333;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .user-answer {
        background: #333333;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        position: relative;
    }
    .user-answer::before {
        content: "🗣️ Your Answer";
        font-size: 0.8rem;
        color: #888888;
        position: absolute;
        top: -10px;
        left: 15px;
        background: #2D2D2D;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .analysis-section {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .strength-badge {
        background: rgba(76, 217, 100, 0.15);
        color: #4CD964;
        padding: 8px 15px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin: 5px;
    }
    .improvement-badge {
        background: rgba(255, 59, 48, 0.15);
        color: #FF3B30;
        padding: 8px 15px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin: 5px;
    }
    .topic-pill {
        background: rgba(74, 144, 226, 0.15);
        color: #4A90E2;
        padding: 8px 20px;
        border-radius: 20px;
        margin: 5px;
        display: inline-block;
        transition: all 0.3s ease;
    }
    .topic-pill:hover {
        transform: scale(1.05);
        background: rgba(74, 144, 226, 0.25);
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    @keyframes slideIn {
        from {transform: translateX(-20px); opacity: 0;}
        to {transform: translateX(0); opacity: 1;}
    }
    @keyframes glow {
        0% {box-shadow: 0 0 5px rgba(74, 144, 226, 0.5);}
        100% {box-shadow: 0 0 20px rgba(74, 144, 226, 0.8);}
    }
    </style>
""", unsafe_allow_html=True)



# -------------------- Core Functions --------------------
@st.cache_resource
def setup_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def setup_chromadb():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="resumes")

def extract_text_from_resume(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text("text") for page in doc])
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

def extract_candidate_name(resume_text):
    # Simple regex to extract names (look for first capitalized words)
    name_match = re.search(r"([A-Z][a-z]+\s+[A-Z][a-z]+)", resume_text[:500])
    if name_match:
        return name_match.group(1)
    return "Candidate"

def store_resume(text, user_id):
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.embed_query(chunk)
        collection.add(
            ids=[f"{user_id}-{i}"],
            embeddings=[embedding],
            metadatas=[{"text": chunk}]
        )
    return extract_candidate_name(text)

def retrieve_resume(user_id, query):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return "\n".join([doc["text"] for doc in results["metadatas"][0]])

def generate_groq_response(prompt, agent_type, temperature=0.7):
    # Different system prompts based on agent type
    system_prompts = {
        "zero_agent": """You are the initial interviewer. Your role is to warmly greet the candidate by name and ask general background questions to make them comfortable before transitioning to technical topics. Be conversational, friendly, and engaging. Focus on understanding their motivation, work history, and personality.""",
        
        "technical_agent": """You are an expert technical interviewer. Analyze the candidate's resume thoroughly and ask highly relevant technical questions that demonstrate your understanding of their background. Your questions should be challenging but fair, focusing on their claimed skills and past projects. Phrase questions clearly and directly.""",
        
        "clarification_agent": """You are a supportive interviewer who helps clarify questions when candidates need assistance. When a candidate seems confused or directly asks for clarification, explain the question in simpler terms with examples. If they give a partial answer, ask follow-up questions to help them elaborate. Your goal is to maintain conversation flow and help candidates showcase their knowledge.""",
        
        "report_agent": """You are an interview assessment specialist. Create a detailed, constructive report of the interview without scoring or grading the candidate. Identify correct answers with green text and areas for improvement with red text. Focus on suggesting specific technical topics the candidate should study further rather than platforms or resources. Be encouraging and specific in your feedback."""
    }
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompts.get(agent_type, "You are an AI interview coach.")},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=800
    )
    return response.choices[0].message.content

# -------------------- Agent Functions --------------------
def zero_agent_greeting(resume_data, candidate_name):
    prompt = f"""
    Resume Data: {resume_data}
    Candidate Name: {candidate_name}
    
    Generate a brief, warm greeting for {candidate_name}. The greeting should:
    1. Begin with "Hello [Candidate Name]" 
    2. Very briefly mention something from their resume (one skill or experience)
    3. Ask ONE simple question about their most recent job or experience
    4. Keep it extremely concise (2-3 short sentences maximum)
    
    The greeting must be brief as it will be converted to voice later.
    """
    return generate_groq_response(prompt, "zero_agent", temperature=0.7)

def technical_agent_question(resume_data, interview_history, question_count):
    difficulty = "introductory" if question_count < 2 else "intermediate" if question_count < 4 else "advanced"
    
    prompt = f"""
    Resume Data: {resume_data}
    Interview History: {interview_history}
    Question Number: {question_count + 1}
    Difficulty: {difficulty}
    
    Generate a relevant technical interview question based on the candidate's resume. The question should:
    1. Be specific to skills or experiences mentioned in their resume
    2. Feel like it's coming from someone who has read their background
    3. Be appropriately challenging based on their experience level
    4. Be directly relevant to their field
    5. Be clearly phrased as a question (no preambles or explanations)
    """
    return generate_groq_response(prompt, "technical_agent", temperature=0.7)

def clarification_agent_response(question, candidate_response, resume_data):
    # Check if the response indicates confusion or asks for clarification
    needs_clarification = any(phrase in candidate_response.lower() for phrase in 
                             ["i don't understand", "can you explain", "not sure", "what do you mean", 
                              "confused", "unclear", "can you clarify", "don't know what", "?"])
    
    if needs_clarification:
        prompt = f"""
        Original Question: {question}
        Candidate Response: {candidate_response}
        Resume Data: {resume_data}
        
        The candidate needs clarification. Your task is to:
        1. Acknowledge their confusion
        2. Explain the question in simpler terms
        3. Provide a concrete example to illustrate what you're asking
        4. Rephrase the question in a more approachable way
        
        IMPORTANT: Respond in a direct, conversational manner WITHOUT any explanation of your reasoning.
        """
        return generate_groq_response(prompt, "clarification_agent", temperature=0.6)
    else:
        # Check if the answer is incomplete and needs a follow-up
        prompt = f"""
        Original Question: {question}
        Candidate Response: {candidate_response}
        Resume Data: {resume_data}
        
        Evaluate if this response is complete or needs a follow-up.
        If the response is thorough and complete, respond with "COMPLETE".
        If the response is partial or could benefit from elaboration, provide a specific follow-up question.
        If the response is off-topic, provide a more specific version of the original question.
        
        IMPORTANT: If providing a follow-up question, give ONLY the question itself without any explanation of why you're asking it.
        """
        follow_up = generate_groq_response(prompt, "clarification_agent", temperature=0.6)
        
        if "COMPLETE" in follow_up:
            return None
        else:
            # Filter out any reasoning or explanation before the question
            # This regex attempts to find the actual question
            question_match = re.search(r"(?:To help|I would|Let me|Could you|What|How|Why|Can you|Tell me|Describe|Explain).*\?", follow_up)
            if question_match:
                return question_match.group(0)
            return follow_up
        
def strip_markdown(text):
    """Remove markdown formatting from text"""
    # Remove bold/italic markers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove backticks
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Remove links
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    return text

def report_agent_feedback(interview_data, resume_data):
    questions_answers = "\n\n".join([
        f"Q{i+1}: {qa['question']}\nAnswer: {qa['answer']}" 
        for i, qa in enumerate(interview_data)
    ])
    
    prompt = f"""
    Resume Data: {resume_data}
    
    Interview Transcript:
    {questions_answers}
    
    Generate a detailed, visually appealing interview report that:
    1. Analyzes each answer without scoring or grading
    2. Identifies correct information (prefix with "CORRECT: ")
    3. Identifies areas for improvement (prefix with "IMPROVE: ")
    4. Recommends 3-5 specific technical topics (not platforms) the candidate should focus on
    
    Format guidelines:
    - Use emojis to make sections more engaging (✅ for correct points, 💡 for improvement areas)
    - ABSOLUTELY NO MARKDOWN SYNTAX - use plain text only without asterisks, backticks, hashes, etc.
    - Use simple formatting that works well in HTML
    - For each question, provide concise bullet-point style feedback
    - Keep language encouraging and constructive
    
    Format the report with these sections:
    - QUESTION ANALYSIS (for each question)
    - KEY STRENGTHS
    - FOCUS AREAS
    - RECOMMENDED TOPICS
    
    Do not include any numerical scores or grades.
    """
    feedback = generate_groq_response(prompt, "report_agent", temperature=0.7)
    return strip_markdown(feedback)  # Apply the markdown stripper

def strict_agent_monitor(candidate_response):
    prompt = f"""
    Candidate Response: "{candidate_response}"

    Check for these behaviors strictly but fairly:
    1. Repeated gibberish or nonsensical keyboard smashing.
    2. Harsh, rude, or aggressive language.
    3. Profanity or clearly offensive content.

    If clearly inappropriate (repeated profanity/aggression/gibberish), respond:
    "INAPPROPRIATE: [reason]"

    If minor awkwardness, occasional mistakes, or nervousness, respond simply:
    "ACCEPTABLE"

    Be forgiving, human-like, and flexible—only flag clear and serious issues.

    Be human-like: allow up to two minor instances before marking responses as inappropriate. 
    Only flag as inappropriate after clear repeated offenses (3 or more times) or severe disrespect/profanity.
    """
    return generate_groq_response(prompt, "technical_agent", temperature=0.1)

# -------------------- Initialize Components --------------------
embedding_model = setup_embeddings()
collection = setup_chromadb()

# -------------------- Session State --------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "interview_active" not in st.session_state:
    st.session_state.interview_active = False
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "interview_phase" not in st.session_state:
    st.session_state.interview_phase = "greeting"  # greeting, technical, wrap_up
if "questions" not in st.session_state:
    st.session_state.questions = []
if "responses" not in st.session_state:
    st.session_state.responses = []
if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = "Candidate"
if "needs_clarification" not in st.session_state:
    st.session_state.needs_clarification = False
if "clarification_response" not in st.session_state:
    st.session_state.clarification_response = None

# -------------------- UI Components --------------------
def show_message(message, is_question=True):
    style_class = "question-card" if is_question else "feedback-card"
    st.markdown(f"""
        <div class="{style_class}">
            <p style="color: #FFFFFF;">{message}</p>
        </div>
    """, unsafe_allow_html=True)

def show_welcome(greeting):
    st.markdown(f"""
        <div class="welcome-card">
            <h3 style="color: #4A90E2; margin-bottom: 1rem;">👋 Welcome to Your Interview Session</h3>
            <p style="color: #FFFFFF;">{greeting}</p>
        </div>
    """, unsafe_allow_html=True)

# -------------------- Main Application Flow --------------------
st.title("💼 AI-Powered Interview Coach")
st.markdown("Upload your resume for a personalized mock interview session")

# Resume Upload Section
with st.expander("📄 Upload Your Resume", expanded=True):
    uploaded_file = st.file_uploader("Choose PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file and not st.session_state.interview_active:
        with st.spinner("Processing your resume..."):
            resume_text = extract_text_from_resume(uploaded_file)
            st.session_state.candidate_name = store_resume(resume_text, st.session_state.user_id)
            st.success("Resume analysis completed!")

# Interview Control
if not st.session_state.interview_active and uploaded_file:
    if st.button("🚀 Start Interview Session"):
        st.session_state.interview_active = True
        st.session_state.current_step = 0
        st.session_state.interview_phase = "greeting"
        st.session_state.questions = []
        st.session_state.responses = []
        st.rerun()

# Interview Session
if st.session_state.interview_active:
    # Greeting Phase
    if st.session_state.interview_phase == "greeting" and not st.session_state.questions:
        with st.spinner("Preparing your interview..."):
            resume_data = retrieve_resume(st.session_state.user_id, "background experience")
            greeting = zero_agent_greeting(resume_data, st.session_state.candidate_name)
            st.session_state.questions.append(greeting)
            show_welcome(greeting)
    
    # Show current message/question
    if st.session_state.needs_clarification and st.session_state.clarification_response:
        show_message(st.session_state.clarification_response, is_question=True)
    elif st.session_state.questions and st.session_state.current_step < len(st.session_state.questions):
        current_question = st.session_state.questions[st.session_state.current_step]
        if st.session_state.current_step > 0 or st.session_state.interview_phase != "greeting":
            show_message(current_question, is_question=True)
    
    # Answer input
    answer = st.text_area(
        "Your Response:", 
        key=f"answer_{st.session_state.current_step}_{st.session_state.needs_clarification}",
        height=150,
        placeholder="Type your response here..."
    )
    
    if st.button("Submit Response"):
        if answer.strip():
            with st.spinner("Processing your response..."):
                appropriateness_check = strict_agent_monitor(answer)
                if "INAPPROPRIATE:" in appropriateness_check:
                    reason = appropriateness_check.split("INAPPROPRIATE:")[1].strip()
                    
                    # End the interview with a popup
                    st.session_state.interview_active = False
                    st.error(f"⚠️ Interview Terminated")
                    
                    st.markdown(f"""
                    <div style="background:#FF3B30; padding:1.5rem; border-radius:10px; color:white; text-align:center;">
                        <h3 style="margin:0 0 1rem 0;">Interview Terminated</h3>
                        <p style="margin:0;">{reason}</p>
                        <p style="margin:1rem 0 0 0; font-size:0.9rem;">Professional communication is essential in interview settings. 
                        Please restart the interview and maintain appropriate professional discourse.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # No further processing needed
                    st.rerun()
                
                current_question = st.session_state.questions[st.session_state.current_step]
                
                # Handle clarification request if needed
                if st.session_state.needs_clarification:
                    st.session_state.needs_clarification = False
                    st.session_state.responses[-1]['clarification'] = st.session_state.clarification_response
                    st.session_state.responses[-1]['clarification_response'] = answer
                    st.session_state.clarification_response = None
                    
                    # Move to next question
                    if st.session_state.interview_phase == "greeting":
                        st.session_state.interview_phase = "technical"
                        resume_data = retrieve_resume(st.session_state.user_id, "technical skills")
                        new_question = technical_agent_question(resume_data, "", 0)
                        st.session_state.questions.append(new_question)
                        st.session_state.current_step += 1
                    elif len(st.session_state.responses) >= 6:  # Limit to 5 technical questions + greeting
                        st.session_state.interview_active = False
                    else:
                        interview_history = "\n".join([
                            f"Q: {item['question']}\nA: {item['answer']}" 
                            for item in st.session_state.responses
                        ])
                        resume_data = retrieve_resume(st.session_state.user_id, "technical skills")
                        new_question = technical_agent_question(
                            resume_data, 
                            interview_history, 
                            len(st.session_state.responses) - 1
                        )
                        st.session_state.questions.append(new_question)
                        st.session_state.current_step += 1
                else:
                    # Store the response
                    st.session_state.responses.append({
                        'question': current_question,
                        'answer': answer
                    })
                    
                    # Check if clarification is needed
                    resume_data = retrieve_resume(st.session_state.user_id, current_question)
                    clarification = clarification_agent_response(
                        current_question, 
                        answer,
                        resume_data
                    )
                    
                    if clarification:
                        st.session_state.needs_clarification = True
                        st.session_state.clarification_response = clarification
                    else:
                        # No clarification needed, proceed to next question
                        if st.session_state.interview_phase == "greeting":
                            st.session_state.interview_phase = "technical"
                            resume_data = retrieve_resume(st.session_state.user_id, "technical skills")
                            new_question = technical_agent_question(resume_data, "", 0)
                            st.session_state.questions.append(new_question)
                            st.session_state.current_step += 1
                        elif len(st.session_state.responses) >= 6:  # Limit to 5 technical questions + greeting
                            st.session_state.interview_active = False
                        else:
                            interview_history = "\n".join([
                                f"Q: {item['question']}\nA: {item['answer']}" 
                                for item in st.session_state.responses
                            ])
                            resume_data = retrieve_resume(st.session_state.user_id, "technical skills")
                            new_question = technical_agent_question(
                                resume_data, 
                                interview_history, 
                                len(st.session_state.responses) - 1
                            )
                            st.session_state.questions.append(new_question)
                            st.session_state.current_step += 1
                
                st.rerun()

# Final Report
if not st.session_state.interview_active and st.session_state.responses:
    st.balloons()
    st.markdown("---")
    st.subheader("📊 Interview Feedback Report")
    
    with st.spinner("Generating comprehensive feedback..."):
        resume_data = retrieve_resume(st.session_state.user_id, "complete profile")
        feedback = report_agent_feedback(st.session_state.responses, resume_data)
        
        # Process the feedback to extract correct/improve sections
        processed_feedback = []
        for qa_index, qa in enumerate(st.session_state.responses):
            question_section = f"Q{qa_index+1}: {qa['question']}"
            answer_section = f"Answer: {qa['answer']}"
            
            # Find analysis for this question
            correct_parts = re.findall(r"CORRECT:(.*?)(?=IMPROVE:|$)", feedback, re.DOTALL)
            improve_parts = re.findall(r"IMPROVE:(.*?)(?=CORRECT:|$)", feedback, re.DOTALL)

            correct_html = ""
            if qa_index < len(correct_parts) and correct_parts[qa_index].strip():
                correct_text = strip_markdown(correct_parts[qa_index].strip())
                correct_html = f"""
                <div class="correct-answer">
                    <h4 style="color: #4CD964; margin:0;">✅ Strong Points</h4>
                    <p style="color: #CCCCCC; margin-top:0.5rem;">{correct_text}</p>
                </div>
                """
                
            improve_html = ""
            if qa_index < len(improve_parts) and improve_parts[qa_index].strip():
                improve_html = f"""
                <div class="wrong-answer">
                    <h4 style="color: #FF3B30; margin:0;">💡 Areas to Develop</h4>
                    <p style="color: #CCCCCC; margin-top:0.5rem;">{improve_parts[qa_index].strip()}</p>
                </div>
                """
                
            processed_feedback.append({
                "question": question_section,
                "answer": answer_section,
                "correct_html": correct_html,
                "improve_html": improve_html
            })

        # Extract recommended topics
        topic_match = re.search(r"RECOMMENDED TOPICS:(.*?)(?=$)", feedback, re.DOTALL)
        topics = []
        if topic_match:
            topics_text = topic_match.group(1).strip()
            topics = [topic.strip() for topic in re.split(r'\d+\.\s+', topics_text) if topic.strip()]
            topics = [topic for topic in topics if len(topic) > 3]  # Filter out short/empty topics
    
    with st.container():
        st.markdown("""
            <div class='final-report'>
                <h3 style='color: #4A90E2; margin-bottom: 1.5rem;'>Interview Summary Report</h3>
        """, unsafe_allow_html=True)
        
        # Interview Overview
        st.markdown("""
            <div style="background:#2D2D2D; padding:1.5rem; border-radius:10px; margin:2rem 0;">
                <h4 style="margin:0; color:#FFFFFF;">Interview Overview</h4>
                <p style="margin:1rem 0 0 0; color:#CCCCCC;">Below is a detailed breakdown of your interview responses with constructive feedback to help you improve your technical skills.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Detailed Responses
        st.markdown("<h4 style='color: #FFFFFF; margin-bottom:1rem;'>Question-by-Question Analysis</h4>", unsafe_allow_html=True)
        for idx, response in enumerate(processed_feedback):
            with st.expander(f"Question {idx+1}", expanded=False):
                st.markdown(f"""
                    <div style='margin-bottom: 1.5rem;'>
                        <p style='font-weight: 500; color: #FFFFFF; font-size: 1.1rem;'>❝{response['question']}❞</p>
                        
                        <div style='background: #333333; padding:1rem; border-radius:8px; margin:1rem 0;'>
                            <p style='color: #888888; margin:0;'>Your Answer:</p>
                            <p style='color: #FFFFFF; margin:0.5rem 0;'>{response['answer']}</p>
                        </div>
                        
                        {response['correct_html']}
                        {response['improve_html']}
                    </div>
                """, unsafe_allow_html=True)
        
        # Improvement Recommendations
        st.markdown("<h4 style='color: #FFFFFF; margin:2rem 0 1rem 0;'>📚 Focus Areas for Improvement</h4>", unsafe_allow_html=True)
        
        if topics:
            st.markdown("""
                <div style="background:#2D2D2D; padding:1.5rem; border-radius:10px; margin:1rem 0;">
                    <h4 style="margin:0; color:#FFFFFF;">Recommended Topics to Study</h4>
                    <p style="margin:1rem 0; color:#CCCCCC;">Based on your interview responses, we recommend focusing on these key areas:</p>
                    <div style="margin-top:1rem;">
            """, unsafe_allow_html=True)
            
            for topic in topics:
                st.markdown(f"""
                    <div class="topic-chip">{topic}</div>
                """, unsafe_allow_html=True)
                
            st.markdown("""
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Restart button
    if st.button("Start New Interview"):
        st.session_state.interview_active = False
        st.session_state.current_step = 0
        st.session_state.interview_phase = "greeting"
        st.session_state.questions = []
        st.session_state.responses = []
        st.session_state.needs_clarification = False
        st.session_state.clarification_response = None
        st.rerun()

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888888; margin: 2rem 0;'>Structured practice interviews to enhance your technical communication skills</div>", unsafe_allow_html=True)



