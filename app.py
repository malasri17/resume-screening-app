import streamlit as st
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import docx

# Page Configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# Main Title & Subtitle
st.title("📄 AI Resume Screening System")
st.caption("An Intelligent System to Screen Resumes and Shortlist Candidates Against Job Descriptions")

# -------------------------------------------------------------
# SKILL TAXONOMY & SYNONYMS
# -------------------------------------------------------------
SKILLS = {
    "python": ["python", "py"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural networks"],
    "nlp": ["nlp", "natural language processing"],
    "sql": ["sql", "mysql", "postgresql"],
    "pandas": ["pandas"],
    "tensorflow": ["tensorflow", "keras"],
    "pytorch": ["pytorch"],
    "statistics": ["statistics", "statistical analysis"],
    "tableau": ["tableau", "power bi"],
    "excel": ["excel", "spreadsheets"],
    "java": ["java"],
    "spring": ["spring boot", "spring"],
    "rest api": ["rest api", "restful services"],
    "docker": ["docker", "containers"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "ci/cd": ["ci/cd", "jenkins", "github actions"],
    "linux": ["linux", "bash"],
    "javascript": ["javascript", "js"],
    "react": ["react", "reactjs"],
    "html": ["html", "css"],
    "node": ["node.js", "nodejs", "express"],
    "git": ["git", "github"],
    "networking": ["networking", "tcp/ip"],
    "security": ["cybersecurity", "information security"],
}

SURFACE = {form: canon for canon, forms in SKILLS.items() for form in forms}

# -------------------------------------------------------------
# FILE EXTRACTION & AGENTS
# -------------------------------------------------------------
def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = pypdf.PdfReader(uploaded_file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return " ".join([p.text for p in doc.paragraphs])
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")

def parse_agent(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_agent(clean):
    found = set()
    for surf, canon in SURFACE.items():
        if re.search(r"(?<![a-z0-9])" + re.escape(surf) + r"(?![a-z0-9])", clean):
            found.add(canon)
            
    m = re.findall(r"(\d+)\+?\s*(?:years|yrs)", clean)
    years = max(map(int, m)) if m else 0
    
    edu = 0
    if "ph.d" in clean or "phd" in clean:
        edu = 3
    elif any(d in clean for d in ["m.tech", "mba", "m.sc", "ms", "master"]):
        edu = 2
    elif any(d in clean for d in ["b.tech", "b.e", "b.sc", "bachelor"]):
        edu = 1
        
    return {"skills": found, "years": years, "edu": edu, "text": clean}

def jd_requirements(jd_text):
    clean = parse_agent(jd_text)
    info = extract_agent(clean)
    m = re.search(r"minimum\s*(\d+)\s*(?:years|yrs)", clean)
    info["min_years"] = int(m.group(1)) if m else 2
    return info

class MatcherAgent:
    W = (0.45, 0.35, 0.12, 0.08)

    def __init__(self, jd_info):
        self.jd = jd_info

    def fit_score(self, cands):
        docs = [self.jd["text"]] + [c["text"] for c in cands]
        tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit_transform(docs)
        sem = cosine_similarity(tf[0:1], tf[1:]).ravel()
        
        req = self.jd["skills"]
        cov = np.array([len(req & c["skills"]) / max(len(req), 1) for c in cands])
        exp = np.array([min(c["years"] / max(self.jd["min_years"], 1), 1.0) for c in cands])
        edu = np.array([min(c["edu"] / 2.0, 1.0) for c in cands])
        
        w = self.W
        final_scores = (w[0] * sem) + (w[1] * cov) + (w[2] * exp) + (w[3] * edu)
        
        breakdowns = []
        for i in range(len(cands)):
            breakdowns.append({
                "semantic": sem[i],
                "coverage": cov[i],
                "exp": exp[i],
                "edu": edu[i],
                "matched_skills": sorted(list(req & cands[i]["skills"])),
                "missing_skills": sorted(list(req - cands[i]["skills"]))
            })
        return final_scores, breakdowns

def ranker_agent(scores, threshold=0.45):
    order = np.argsort(-scores)
    shortlisted = scores >= threshold
    return order, shortlisted

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Resume Screening", "About Project"])

# -------------------------------------------------------------
# 1. DASHBOARD PAGE
# -------------------------------------------------------------
if page == "Dashboard":
    st.subheader("Recruiter Dashboard")
    
    total = len(st.session_state.get("results", {}).get("names", []))
    shortlisted = sum(st.session_state.get("results", {}).get("shortlisted", []))
    rejected = total - shortlisted
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Resumes", total)
    col2.metric("Shortlisted", shortlisted)
    col3.metric("Rejected", rejected)
    
    st.markdown("---")
    st.subheader("Welcome, Recruiter!")
    st.write("Upload job descriptions and candidate resumes to begin intelligent resume screening.")

# -------------------------------------------------------------
# 2. RESUME SCREENING PAGE
# -------------------------------------------------------------
elif page == "Resume Screening":
    st.subheader("1. Job Description")
    default_jd = "Looking for a Data Scientist with minimum 3 years of experience. Required skills: Python, Machine Learning, SQL, Pandas, Statistics."
    jd_input = st.text_area("Enter Job Description (JD):", value=default_jd, height=120)
    
    st.subheader("2. Upload Resumes")
    uploaded_files = st.file_uploader(
        "Drag and drop files here (Limit 200MB per file • PDF, DOCX)", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    st.subheader("3. Matching Settings")
    threshold_pct = st.slider("Shortlisting threshold (%)", min_value=10, max_value=90, value=45, step=1)
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} resume(s) uploaded successfully!")
        
    if st.button("Start Resume Screening", type="primary"):
        if not jd_input.strip():
            st.error("Please enter a valid Job Description!")
        elif not uploaded_files:
            st.error("Please upload at least one resume!")
        else:
            with st.spinner("Processing Resumes via Multi-Agent Pipeline..."):
                jd_info = jd_requirements(jd_input)
                
                cands = []
                cand_names = []
                for f in uploaded_files:
                    raw = extract_text_from_file(f)
                    clean = parse_agent(raw)
                    cand_info = extract_agent(clean)
                    cands.append(cand_info)
                    cand_names.append(f.name)
                
                matcher = MatcherAgent(jd_info)
                scores, breakdowns = matcher.fit_score(cands)
                
                thr = threshold_pct / 100.0
                ranked_order, shortlisted_flags = ranker_agent(scores, thr)
                
                st.session_state["results"] = {
                    "names": cand_names,
                    "scores": scores,
                    "breakdowns": breakdowns,
                    "ranked_order": ranked_order,
                    "shortlisted": shortlisted_flags,
                    "cands": cands,
                    "jd_info": jd_info
                }
            st.success("Screening Completed Successfully!")

    # Display Results
    if "results" in st.session_state:
        res = st.session_state["results"]
        ranked_order = res["ranked_order"]
        total_req = len(res["jd_info"]["skills"])
        
        st.markdown("---")
        st.subheader("📊 Recruiter Dashboard - Ranked Shortlist")
        
        table_rows = []
        for rank, idx in enumerate(ranked_order, start=1):
            name = res["names"][idx]
            final_s = res["scores"][idx]
            is_pass = res["shortlisted"][idx]
            matched_len = len(res["breakdowns"][idx]["matched_skills"])
            exp_found = res["cands"][idx]["years"]
            
            table_rows.append({
                "Rank": rank,
                "Candidate": name,
                "Experience": f"{exp_found} yrs",
                "Skills matched": f"{matched_len}/{total_req}",
                "Score": f"{final_s:.2f}",
                "Status": "SHORTLISTED" if is_pass else "REJECTED"
            })
            
        df = pd.DataFrame(table_rows)
        st.table(df)
        
        st.markdown("---")
        st.subheader("🔍 Score Breakdown & Skill Gap View")
        
        selected_candidate = st.selectbox("Select Candidate:", [res["names"][i] for i in ranked_order])
        sel_idx = res["names"].index(selected_candidate)
        b_down = res["breakdowns"][sel_idx]
        
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown(f"**Final Score:** `{res['scores'][sel_idx]:.2f}`")
            st.write(f"• Semantic similarity (w=0.45): `{b_down['semantic']:.2f}`")
            st.write(f"• Skill coverage (w=0.35): `{b_down['coverage']:.2f}`")
            st.write(f"• Experience (w=0.12): `{b_down['exp']:.2f}`")
            st.write(f"• Education (w=0.08): `{b_down['edu']:.2f}`")
            
        with col_right:
            m_skills = ", ".join(b_down["matched_skills"]) if b_down["matched_skills"] else "None"
            miss_skills = ", ".join(b_down["missing_skills"]) if b_down["missing_skills"] else "None"
            st.success(f"**Matched skills:** {m_skills}")
            st.error(f"**Missing skills:** {miss_skills}")

# -------------------------------------------------------------
# 3. ABOUT PROJECT PAGE
# -------------------------------------------------------------
elif page == "About Project":
    st.subheader("About the Project")
    st.write("**Title:** AN INTELLIGENT SYSTEM TO SCREEN RESUMES AND SHORTLIST CANDIDATES AGAINST JOB DESCRIPTIONS")
    st.write("**Department:** Artificial Intelligence and Data Science")
    st.write("**Institution:** Ramachandra College of Engineering (Autonomous)")
    st.markdown("""
    **Modules & Agents:**
    - **Parser Agent:** Cleans text, handles symbols and formatting.
    - **Extractor Agent:** Extracts skills with synonyms, detects years of experience & degrees.
    - **Matcher Agent:** TF-IDF Cosine Similarity + Weighted Multi-Criteria Scoring.
    - **Ranker Agent:** Automatic sorting and threshold-based shortlisting.
    """)