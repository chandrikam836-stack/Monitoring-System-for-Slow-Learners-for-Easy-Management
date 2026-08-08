import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import os
import base64
from database.db_connection import get_connection

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from fpdf import FPDF

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Student Monitoring System",
    page_icon="🎓",
    layout="wide"
)
#UNIQUE RAPPER FOR LOGGIN PAGE


# ==================================================
# SESSION STATE
# ==================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ==================================================
# CREATE FOLDERS
# ==================================================
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ==================================================
# DATABASE CONNECTION
# ==================================================
conn = get_connection()

# ==================================================
# BACKGROUND / ASSET HELPERS
# ==================================================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles/style.css")

# ==================================================
# UI HELPERS (presentation only — no business logic)
# ==================================================
def section_header(title, subtitle=None, eyebrow=None):
    """Styled drop-in replacement for st.header(); same title text, premium markup."""
    eyebrow_html = f"<div class='section-eyebrow'>{eyebrow}</div>" if eyebrow else ""
    subtitle_html = f"<div class='section-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div class="section-header-block">
        {eyebrow_html}
        <div class="section-title">{title}</div>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

def _tint(hex_color, ratio=0.16):
    """Blend a hex color with white for a soft icon-chip background. Pure presentation helper."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r * ratio + 255 * (1 - ratio))
    g = int(g * ratio + 255 * (1 - ratio))
    b = int(b * ratio + 255 * (1 - ratio))
    return f"rgb({r},{g},{b})"

def render_kpi_row(cards):
    """cards: list of dicts -> label, value, icon, color, suffix, subtitle (optional).
    Animated count-up, same underlying values as before — presentation only."""
    card_html = ""
    for c in cards:
        subtitle_html = f"<div class='kpi-sub'>{c['subtitle']}</div>" if c.get('subtitle') else ""
        card_html += f"""
        <div class="kpi-card" style="--accent:{c['color']}">
            <div class="kpi-icon" style="background:{_tint(c['color'])}; color:{c['color']}">{c['icon']}</div>
            <div class="kpi-label">{c['label']}</div>
            <div class="kpi-value" data-target="{c['value']}">0{c.get('suffix','')}</div>
            {subtitle_html}
        </div>
        """

    html = f"""
    <style>
        .kpi-row {{
            display: flex; gap: 16px; margin-bottom: 18px; font-family: 'Plus Jakarta Sans', sans-serif;
        }}
        .kpi-card {{
            position: relative;
            flex: 1; background: rgba(255,255,255,0.78); backdrop-filter: blur(16px);
            border: 1px solid rgba(37,99,235,0.10); border-radius: 18px; padding: 20px 20px 16px;
            box-shadow: 0 8px 30px rgba(15,23,42,0.06);
            transition: transform 0.25s ease, box-shadow 0.25s ease; animation: kpiIn 0.45s ease both;
            overflow: hidden;
        }}
        .kpi-card::before {{
            content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px;
            background: var(--accent); border-radius: 18px 18px 0 0;
        }}
        .kpi-card:hover {{ transform: translateY(-4px); box-shadow: 0 16px 36px rgba(37,99,235,0.18); }}
        .kpi-icon {{
            width: 40px; height: 40px; border-radius: 12px; display:flex; align-items:center; justify-content:center;
            font-size: 18px; margin-bottom: 12px;
        }}
        .kpi-label {{ font-size: 13px; font-weight: 700; color: #1E293B; margin-bottom: 4px; }}
        .kpi-value {{
            font-family: 'DM Mono', monospace; font-size: 27px; font-weight: 700; color: #0F172A;
            margin-bottom: 4px;
        }}
        .kpi-sub {{ font-size: 11.5px; color: #94A3B8; font-weight: 500; }}
        @keyframes kpiIn {{ from {{ opacity:0; transform: translateY(8px); }} to {{ opacity:1; transform: translateY(0); }} }}
    </style>
    <div class="kpi-row">{card_html}</div>
    <script>
        const vals = document.querySelectorAll('.kpi-value');
        vals.forEach(el => {{
            const target = parseFloat(el.getAttribute('data-target'));
            const suffix = el.textContent.replace(/[0-9.\\-]/g, '');
            let cur = 0;
            const steps = 24;
            const inc = target / steps;
            const isInt = Number.isInteger(target);
            const timer = setInterval(() => {{
                cur += inc;
                if (cur >= target) {{ cur = target; clearInterval(timer); }}
                el.textContent = (isInt ? Math.round(cur) : cur.toFixed(1)) + suffix;
            }}, 16);
        }});
    </script>
    """
    components.html(html, height=148)

def apply_table_search(df, key):
    """Adds a search box above a table and returns the filtered dataframe. Display-only filter."""
    term = st.text_input("🔍 Search this table", key=key, placeholder="Type to filter rows…")
    if not term:
        return df
    # regex=False -> treat the search term as a literal string so characters like
    # ( ) [ ] * + ? don't get interpreted as regex and crash the search.
    mask = df.astype(str).apply(lambda col: col.str.contains(term, case=False, na=False, regex=False))
    return df[mask.any(axis=1)]

def export_button(df, filename, key, label="⬇ Export view (CSV)"):
    st.download_button(label, df.to_csv(index=False), file_name=filename, key=key)


# ==================================================
# LOGIN PAGE — SPLIT-SCREEN GLASS LAYOUT
# (Streamlit cannot nest live widgets inside a single
#  st.markdown div, so the split is built with real
#  st.columns — left = illustration, right = form.
#  Auth logic below is 100% unchanged.)
# ==================================================
if not st.session_state.logged_in:

    illustration_path = "assets/login_bg.jpg"
    illu_bg_style = ""
    if os.path.exists(illustration_path):
        # File is PNG-encoded despite the .jpg extension; use the correct MIME type.
        illu_bg_style = f"background-image: url('data:image/png;base64,{get_base64(illustration_path)}');"

    st.markdown('<div class="login-shell">', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.05, 1], gap="small")

    # -------------------- LEFT: ILLUSTRATION PANEL --------------------
    with left_col:
        st.markdown(f"""
        <div class="login-illustration" style="{illu_bg_style}">
          <div class="illu-content">
            <div class="illu-badge">🎓 Aditya University</div>
            <h2>Empowering Every Learner,<br>Every Day</h2>
            <p>Track academic performance, flag at-risk students early, and act on AI-driven recommendations — all in one place.</p>
            <div class="illu-stats">
              <div class="illu-stat-card"><div class="illu-icon">🎓</div><div><div class="s-num">Smart Academic</div><div class="s-label">Monitoring</div></div></div>
              <div class="illu-stat-card"><div class="illu-icon">📊</div><div><div class="s-num">Student Analysis</div><div class="s-label">and Learn Classification</div></div></div>
              <div class="illu-stat-card"><div class="illu-icon">🛡️</div><div><div class="s-num">Academic Performance</div><div class="s-label">Recommendations</div></div></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------- RIGHT: LOGIN FORM PANEL --------------------
    with right_col:
        # LOGO (falls back to college logo if a dedicated logo.png isn't provided)
        logo_path = None
        if os.path.exists("assets/logo.png"):
            logo_path = "assets/logo.png"
        elif os.path.exists("assets/college_logo.jpg"):
            logo_path = "assets/college_logo.jpg"

        if logo_path:
            ext = "png" if logo_path.endswith("png") else "jpeg"
            st.markdown(f"""
            <div class="login-logo login-logo-center">
                <img src="data:image/{ext};base64,{get_base64(logo_path)}">
                <div class="login-logo-wordmark">
                    <span class="lw-orange">ADITYA</span><span class="lw-blue">UNIVERSITY</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # TITLE
        st.markdown("<div class='sub-title'>WELCOME BACK</div>", unsafe_allow_html=True)
        st.markdown("<div class='main-title'>Monitoring System for Slow Learners</div>", unsafe_allow_html=True)
        st.markdown("<div class='tag-line'>Sign in to access your dashboard</div>", unsafe_allow_html=True)

        st.markdown('<div class="field-label">Username</div>', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed", key="login_username")

        st.markdown('<div class="field-label">Password</div>', unsafe_allow_html=True)
        password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed", key="login_password")

        st.markdown('<div class="field-label">Select Role</div>', unsafe_allow_html=True)
        role = st.selectbox("Select Role", ["Admin", "Faculty", "Mentor"], label_visibility="collapsed", key="login_role")

        remember_col, forgot_col = st.columns([1, 1])
        with remember_col:
            st.checkbox("Remember me", value=True, key="login_remember")
        with forgot_col:
            st.markdown('<div class="forgot-link">Forgot Password?</div>', unsafe_allow_html=True)

        login_clicked = st.button("➝  Login", key="login_submit", use_container_width=True)

        st.markdown("""
        <div class='bottom-line'>
        “Every learner has potential. Our mission is to identify, support, and help them grow.”
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if login_clicked:
        users = pd.read_sql(
            "SELECT * FROM users",
             conn
        )

        user = users[
            (users['username'] == username) &
            (users['password'] == password) &
            (users['role'] == role)
        ]

        if not user.empty:

            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = role
            st.session_state.department = user.iloc[0]['department']

            st.success("Login Successful")
            st.rerun()

        else:
            st.error("Invalid Credentials")

    st.stop()

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_sql(
    "SELECT * FROM students",
    conn
)
# ==================================================
# ROLE BASED ACCESS (FIXED MENTOR)
# ==================================================
user_role = st.session_state.role
user_department = st.session_state.department

if user_role == "Admin":
    filtered_role_df = df.copy()

elif user_role == "Faculty":
    filtered_role_df = df[df['Department'] == user_department]

elif user_role == "Mentor":
    # FIXED: department-based access (NO ID LIST)
    filtered_role_df = df[df['Department'] == user_department]

else:
    filtered_role_df = df.copy()

df = filtered_role_df

# ==================================================
# REQUIRED COLUMNS
# ==================================================
required_columns = [

    "Department",

    "Hackathons",
    "Ideathons",
    "Quizzes",
    "Coding_Contests",
    "Workshops",
    "Certifications",
    "Sports",
    "NSS_NCC",
    "Clubs",
    "Projects",

    "Attendance",
    "Backlogs"
]

for col in required_columns:

    if col not in df.columns:

        if col == "Department":
            df[col] = "CSE"

        elif col == "Attendance":
            # Fail-safe default: 0 (not 100) so missing attendance data is
            # flagged as "At Risk" for review, rather than silently hidden.
            df[col] = 0

        else:
            df[col] = 0

# ==================================================
# ACTIVITY SCORE
# ==================================================
def calculate_activity_score(row):

    score = (
        row.get("Hackathons", 0) * 10 +
        row.get("Ideathons", 0) * 8 +
        row.get("Coding_Contests", 0) * 7 +
        row.get("Certifications", 0) * 5 +
        row.get("Workshops", 0) * 4 +
        row.get("Quizzes", 0) * 3 +
        row.get("Sports", 0) * 2 +
        row.get("Clubs", 0) * 2 +
        row.get("Projects", 0) * 10 +
        row.get("NSS_NCC", 0) * 3
    )

    return min(score, 100)


df["CoCurricular_Score"] = df.apply(
    calculate_activity_score,
    axis=1
)


# ==================================================
# CLASSIFICATION
# ==================================================
def classify(row):

    marks = []

    # School marks
    if row.get("SSC_Percentage", 0) > 0:
        marks.append(row["SSC_Percentage"])

    if row.get("Twelfth_Percentage", 0) > 0:
        marks.append(row["Twelfth_Percentage"])

    # Semester marks
    semester_columns = [
        "First_Sem",
        "Second_Sem",
        "Third_Sem",
        "Fourth_Sem",
        "Fifth_Sem",
        "Sixth_Sem",
        "Seventh_Sem",
        "Eighth_Sem"
    ]

    for col in semester_columns:
        if row.get(col, 0) > 0:
            marks.append(row[col])

    academic_avg = sum(marks) / len(marks)

    overall_score = (
        academic_avg * 0.8 +
        row["CoCurricular_Score"] * 0.2
    )

    if overall_score >= 75:
        return "Fast Learner"

    elif overall_score >= 60:
        return "Average"

    else:
        return "Slow Learner"


# ==================================================
# RISK STATUS
# ==================================================
def risk(row):

    marks = []

    semester_columns = [
        "First_Sem",
        "Second_Sem",
        "Third_Sem",
        "Fourth_Sem",
        "Fifth_Sem",
        "Sixth_Sem",
        "Seventh_Sem",
        "Eighth_Sem"
    ]

    for col in semester_columns:
        if row.get(col, 0) > 0:
            marks.append(row[col])

    if len(marks) > 0:
        avg = sum(marks) / len(marks)
    else:
        avg = (
            row.get("SSC_Percentage", 0) +
            row.get("Twelfth_Percentage", 0)
        ) / 2

    if (
        avg < 60 or
        row.get("Attendance", 100) < 75 or
        row.get("Backlogs", 0) > 2
    ):
        return "At Risk"

    return "Normal"


# ==================================================
# RISK SCORE
# ==================================================
def calculate_risk_score(row):

    marks = []

    semester_columns = [
        "First_Sem",
        "Second_Sem",
        "Third_Sem",
        "Fourth_Sem",
        "Fifth_Sem",
        "Sixth_Sem",
        "Seventh_Sem",
        "Eighth_Sem"
    ]

    for col in semester_columns:
        if row.get(col, 0) > 0:
            marks.append(row[col])

    if len(marks) > 0:
        avg = sum(marks) / len(marks)
    else:
        avg = (
            row.get("SSC_Percentage", 0) +
            row.get("Twelfth_Percentage", 0)
        ) / 2

    score = 0

    if avg < 60:
        score += 40

    if row.get("Attendance", 100) < 75:
        score += 25

    if row.get("Backlogs", 0) > 2:
        score += 25

    if row.get("Projects", 0) == 0:
        score += 10

    return min(score, 100)


# ==================================================
# RECOMMENDATION
# ==================================================
def recommendation(row):

    recs = []

    if row["Category"] == "Slow Learner":
        recs.append("Attend remedial classes and mentoring sessions.")

    if row["Risk"] == "At Risk":
        recs.append("Meet the faculty mentor for academic guidance.")

    if row.get("Attendance", 100) < 75:
        recs.append("Maintain attendance above 75%.")

    if row.get("Backlogs", 0) > 0:
        recs.append("Clear all pending backlogs.")

    if row.get("Projects", 0) == 0:
        recs.append("Complete at least one academic project.")

    if row.get("Certifications", 0) == 0:
        recs.append("Earn an industry-recognized certification.")

    if (
        row.get("Hackathons", 0) == 0 and
        row.get("Coding_Contests", 0) == 0
    ):
        recs.append("Participate in hackathons and coding contests.")

    if len(recs) == 0:
        recs.append("Excellent performance. Continue maintaining your academic progress.")

    return ", ".join(recs)


# ==================================================
# APPLY LOGIC
# ==================================================
df["Category"] = df.apply(classify, axis=1)

df["Risk"] = df.apply(risk, axis=1)

df["Risk_Score"] = df.apply(calculate_risk_score, axis=1)

df["Recommendation"] = df.apply(recommendation, axis=1)
# ==================================================
# TITLE
# ==================================================
_logo_path = "assets/college_logo.jpg"
_logo_b64 = get_base64(_logo_path) if os.path.exists(_logo_path) else ""
_logo_mark_html = f"""
<div class="app-university-mark">
    <img src="data:image/jpeg;base64,{_logo_b64}" alt="Aditya University" />
    <div>
        <div class="au-name"><span class="au-orange">ADITYA</span> <span class="au-blue">UNIVERSITY</span></div>
        <div class="au-sub">EXCELLENCE IN EDUCATION</div>
    </div>
</div>
""" if _logo_b64 else ""

st.markdown(f"""
<div class="app-title-row">
    <div class="app-title-text">
        <div class="app-title">🎓 Monitoring System for Slow Learners</div>
    </div>
    {_logo_mark_html}
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
_sidebar_university_html = f"""
<div class="sidebar-university">
    <img src="data:image/jpeg;base64,{_logo_b64}" alt="Aditya University" />
    <div class="su-name">
    <span class="su-orange">ADITYA</span>
    <span class="su-blue">UNIVERSITY</span>
     </div>
</div>
""" if _logo_b64 else ""

st.sidebar.markdown(f"""
{_sidebar_university_html}

""", unsafe_allow_html=True)

st.sidebar.success(
    f"{st.session_state.username} ({st.session_state.role})"
)

if st.sidebar.button("Logout"):

    st.session_state.logged_in = False

    st.rerun()

# ==================================================
# ROLE BASED MENU
# ==================================================
if st.session_state.role == "Admin":

    menu = st.sidebar.radio(
    "📌 Navigation",
    [
        "🏠 Dashboard",
        "🔍 Student Analysis",
        "🤖 ML Prediction",
        "🏆 Co-Curricular Analytics",
        "📄 Reports",
        "📂 Upload Data",
        "👨‍💼 User Management"
    ]
)

elif st.session_state.role == "Faculty":

    menu = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Student Analysis",
            "🏆 Co-Curricular Analytics",
            "📄 Reports",
            "📂 Upload Data"
        ]
    )

else:

    menu = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Student Analysis",
            "🏆 Co-Curricular Analytics",
            "📂 Upload Data"
        ]
    )

# ==================================================
# FILTERS
# ==================================================
st.sidebar.subheader("Filters")
stage_filter = st.sidebar.selectbox(
    "🎓 Academic Stage",
    [
        "All",
        "Before B.Tech",
        "1st Year",
        "2nd Year",
        "3rd Year",
        "Final Year"
    ]
)

departments = ["All"] + sorted(df['Department'].unique())

department_filter = st.sidebar.selectbox(
    "Department",
    departments
)

category_filter = st.sidebar.selectbox(
    "Category",
    ["All", "Slow Learner", "Average", "Fast Learner"]
)

filtered_df = df.copy()

if department_filter != "All":

    filtered_df = filtered_df[
        filtered_df['Department'] == department_filter
    ]

if category_filter != "All":

    filtered_df = filtered_df[
        filtered_df['Category'] == category_filter
    ]
stage_columns = {

    "Before B.Tech": [
        "Student_ID",
        "Name",
        "Department",
        "SSC_Percentage",
        "Twelfth_Percentage"
    ],

    "1st Year": [
        "Student_ID",
        "Name",
        "Department",
        "SSC_Percentage",
        "Twelfth_Percentage",
        "First_Mid",
        "Second_Mid",
        "First_Sem",
        "Second_Sem"
    ],

    "2nd Year": [
        "Student_ID",
        "Name",
        "Department",
        "SSC_Percentage",
        "Twelfth_Percentage",
        "First_Mid",
        "Second_Mid",
        "First_Sem",
        "Second_Sem",
        "Third_Mid",
        "Fourth_Mid",
        "Third_Sem",
        "Fourth_Sem"
    ],

    "3rd Year": [
        "Student_ID",
        "Name",
        "Department",
        "SSC_Percentage",
        "Twelfth_Percentage",
        "First_Mid",
        "Second_Mid",
        "First_Sem",
        "Second_Sem",
        "Third_Mid",
        "Fourth_mid",
        "Third_Sem",
        "Fourth_Sem",
        "Fifth_Mid",
        "Sixth_Mid",
        "Fifth_Sem",
        "Sixth_Sem"
    ],

    "Final Year": [
        "Student_ID",
        "Name",
        "Department",
        "SSC_Percentage",
        "Twelfth_Percentage",
        "First_Mid",
        "Second_Mid",
        "First_Sem",
        "Second_Sem",
        "Third_Mid",
        "Fourth_Mid",
        "Third_Sem",
        "Fourth_Sem",
        "Fifth_Mid",
        "Sixth_Mid",
        "Fifth_Sem",
        "Sixth_Sem",
        "Seventh_Mid",
        "Eighth_Mid",
        "Seventh_Sem",
        "Eighth_Sem"
    ]
}
# ==================================================
# STAGE FILTERING
# ==================================================

display_df = filtered_df.copy()

if stage_filter != "All":

    available_cols = [
        col for col in stage_columns[stage_filter]
        if col in filtered_df.columns
    ]

    display_df = filtered_df[available_cols + ['Category']]
else:

    display_df = filtered_df.copy()
# ==================================================
# DASHBOARD
# ==================================================
if menu == "🏠 Dashboard":

    section_header("Dashboard", eyebrow="Overview", subtitle="Live snapshot of student performance and risk across your scope")

    render_kpi_row([
        {"label": "Total Students", "value": len(filtered_df), "icon": "👥", "color": "#2563EB", "subtitle": "All registered students"},
        {"label": "Slow Learners", "value": len(filtered_df[filtered_df['Category'] == "Slow Learner"]), "icon": "🎓", "color": "#10B981", "subtitle": "Students needing support"},
        {"label": "At Risk", "value": len(filtered_df[filtered_df['Risk'] == "At Risk"]), "icon": "⚠️", "color": "#F59E0B", "subtitle": "High risk students"},
        {"label": "Departments", "value": filtered_df['Department'].nunique(), "icon": "🏛️", "color": "#2563EB", "subtitle": "Active departments"},
    ])

    # ================= TABLE COLORS =================

    def highlight_students(row):

     if 'Category' not in row.index:
        return [''] * len(row)

     if row['Category'] == "Slow Learner":
        return ['background-color: #ffcccc; color: black'] * len(row)

     elif row['Category'] == "Average":
        return ['background-color: #fff3cd; color: black'] * len(row)

     elif row['Category'] == "Fast Learner":
        return ['background-color: #d4edda; color: black'] * len(row)

     return [''] * len(row)

    st.markdown("<div class='section-eyebrow' style='margin-top:6px;'>STUDENT RECORDS</div>", unsafe_allow_html=True)
    search_col, export_col = st.columns([4, 1])
    with search_col:
        table_view_df = apply_table_search(display_df, key="dash_table_search")
    with export_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        export_button(table_view_df, "dashboard_students.csv", key="dash_table_export")

    styled_df = table_view_df.style.apply(
      highlight_students,
      axis=1
    )

    st.dataframe(
     styled_df,
     use_container_width=True
    )

    # CATEGORY GRAPH
    fig1 = px.pie(
        filtered_df,
        names='Category',
        title='Category Distribution'
    )

    st.plotly_chart(fig1, use_container_width=True)

    # DEPARTMENT SLOW LEARNERS
    st.markdown("<div class='section-eyebrow' style='margin-top:10px;'>📊 DEPARTMENT WISE SLOW LEARNERS</div>", unsafe_allow_html=True)

    slow_df = filtered_df[
        filtered_df['Category'] == "Slow Learner"
    ]

    dept_slow = slow_df.groupby(
        'Department'
    ).size().reset_index(name='Count')

    fig_slow = px.bar(

        dept_slow,

        x='Department',

        y='Count',

        color='Department',

        title="Department Wise Slow Learners"
    )

    st.plotly_chart(fig_slow, use_container_width=True)

    # MENTOR DETAILS
    if st.session_state.role == "Mentor":

        st.markdown("<div class='section-eyebrow' style='margin-top:10px;'>👨‍🏫 ASSIGNED STUDENTS</div>", unsafe_allow_html=True)

        mentor_view = filtered_df[
            [
                'Student_ID',
                'Name',
                'Department',
                'Category',
                'Risk',
                'Recommendation'
            ]
        ]

        m_search_col, m_export_col = st.columns([4, 1])
        with m_search_col:
            mentor_view = apply_table_search(mentor_view, key="mentor_table_search")
        with m_export_col:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            export_button(mentor_view, "assigned_students.csv", key="mentor_table_export")

        st.dataframe(
            mentor_view,
            use_container_width=True
        )

# ==================================================
# STUDENT ANALYSIS
# ==================================================
if menu == "🔍 Student Analysis":

    section_header("Student Analysis", eyebrow="Deep Dive", subtitle="Search and review individual academic and risk profiles")

    if filtered_df.empty:
        st.warning("No students available.")
        st.stop()

    student_id = st.selectbox(
       "Select Student ID",
        sorted(filtered_df["Student_ID"].unique())
    )

    student = filtered_df[
        filtered_df["Student_ID"] == student_id
    ]

    student_name = student["Name"].values[0]

    st.markdown(f"""
    <div class="profile-card">
    <h2>{student_name}</h2>

    <p><b>Student ID:</b> {student_id}</p>

    <p><b>Department:</b> {student['Department'].values[0]}</p>

    <p><b>Category:</b> {student['Category'].values[0]}</p>

    <p><b>Risk:</b> {student['Risk'].values[0]}</p>
    </div>
    """, unsafe_allow_html=True)

    if stage_filter != "All":

        cols = [
            col for col in stage_columns[stage_filter]
            if col in student.columns
        ]

        st.dataframe(
            student[cols],
            use_container_width=True
        )

    else:

        st.dataframe(
            student,
            use_container_width=True
        )

    export_button(
       student,
       f"{student_id}_{student_name}.csv",
       key="student_record_export",
       label="⬇ Export this record (CSV)"
    )
    st.markdown("<div class='section-eyebrow' style='margin-top:10px;'>📌 RECOMMENDATION</div>", unsafe_allow_html=True)
    st.info(student['Recommendation'].values[0])

    # ==================================================
    # PERFORMANCE GRAPH
    # ==================================================
    st.markdown(
        "<div class='section-eyebrow' style='margin-top:10px;'>📈 PERFORMANCE TREND</div>",
        unsafe_allow_html=True
    )

    performance_map = {
       "Before B.Tech": [
          ("SSC", "SSC_Percentage"),
          ("12th", "Twelfth_Percentage")
       ],

        "1st Year": [
           ("SSC", "SSC_Percentage"),
           ("12th", "Twelfth_Percentage"),
           ("1st Sem", "First_Sem"),
           ("2nd Sem", "Second_Sem")
        ],

        "2nd Year": [
           ("SSC", "SSC_Percentage"),
           ("12th", "Twelfth_Percentage"),
           ("1st Sem", "First_Sem"),
           ("2nd Sem", "Second_Sem"),
           ("3rd Sem", "Third_Sem"),
           ("4th Sem", "Fourth_Sem")
        ],

        "3rd Year": [
           ("SSC", "SSC_Percentage"),
           ("12th", "Twelfth_Percentage"),
           ("1st Sem", "First_Sem"),
           ("2nd Sem", "Second_Sem"),
           ("3rd Sem", "Third_Sem"),
           ("4th Sem", "Fourth_Sem"),
           ("5th Sem", "Fifth_Sem"),
           ("6th Sem", "Sixth_Sem")
        ],
 
        "Final Year": [
           ("SSC", "SSC_Percentage"),
           ("12th", "Twelfth_Percentage"),
           ("1st Sem", "First_Sem"),
           ("2nd Sem", "Second_Sem"),
           ("3rd Sem", "Third_Sem"),
           ("4th Sem", "Fourth_Sem"),
           ("5th Sem", "Fifth_Sem"),
           ("6th Sem", "Sixth_Sem"),
           ("7th Sem", "Seventh_Sem"),
           ("8th Sem", "Eighth_Sem")
        ]
    }

    # If All is selected, show complete academic history
    if stage_filter == "All":
       graph_data = performance_map["Final Year"]
    else:
        graph_data = performance_map[stage_filter]

    exam_names = []
    marks = []

    for exam, column in graph_data:

        if column in student.columns:

           value = student[column].values[0]

        if pd.notna(value):

             exam_names.append(exam)
             marks.append(value)

    performance_data = pd.DataFrame({
       "Exam": exam_names,
       "Marks": marks
    })

    fig_perf = px.line(
       performance_data,
       x="Exam",
       y="Marks",
       markers=True,
       title=f"{stage_filter} Academic Performance"
    )

    fig_perf.update_layout(
        xaxis_title="Examinations",
        yaxis_title="Marks (%)"
    )

    st.plotly_chart(fig_perf, use_container_width=True)
# ==================================================
# ML PREDICTION
# ==================================================

if menu == "🤖 ML Prediction":

    section_header(
        "AI Student Assessment",
        eyebrow="Machine Learning",
        subtitle="Predict Student Learning Category using Decision Tree"
    )

    # ==================================================
    # TRAIN MODEL
    # ==================================================
    # NOTE: The true "Category" label (see classify() above) is a
    # deterministic function of academic marks + co-curricular score
    # ONLY (Academic_Avg*0.8 + CoCurricular_Score*0.2). Earlier this
    # model was also trained on Attendance/Backlogs/Projects/raw
    # semester marks. Because the synthetic training data ties
    # Attendance to the same generation "tier" as marks, the tree
    # learned to shortcut almost entirely on Attendance (~90% feature
    # importance) instead of the real rule - so real students whose
    # attendance doesn't match their marks got misclassified. Training
    # on the same engineered feature the rule actually uses fixes this.

    semester_columns_ml = [
        "First_Sem", "Second_Sem", "Third_Sem", "Fourth_Sem",
        "Fifth_Sem", "Sixth_Sem", "Seventh_Sem", "Eighth_Sem"
    ]

    def compute_academic_avg(row):
        marks = []
        if row.get("SSC_Percentage", 0) > 0:
            marks.append(row["SSC_Percentage"])
        if row.get("Twelfth_Percentage", 0) > 0:
            marks.append(row["Twelfth_Percentage"])
        for col in semester_columns_ml:
            if row.get(col, 0) > 0:
                marks.append(row[col])
        return sum(marks) / len(marks) if marks else 0

    df["Academic_Avg"] = df.apply(compute_academic_avg, axis=1)

    features = [
        "Academic_Avg",
        "CoCurricular_Score"
    ]

    X = df[features].fillna(0)

    y = df["Category"].map({
        "Slow Learner":0,
        "Average":1,
        "Fast Learner":2
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    model.fit(X_train,y_train)

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    st.success(
        f"✅ Decision Tree Accuracy : {accuracy*100:.2f}%"
    )

    st.markdown("---")

    st.subheader("📝 Enter New Student Details")

    stage = st.selectbox(
        "Academic Stage",
        [
            "Before B.Tech",
            "1st Year",
            "2nd Year",
            "3rd Year",
            "Final Year"
        ]
    )

    # default values

    First=Second=Third=Fourth=0
    Fifth=Sixth=Seventh=Eighth=0

    col1,col2=st.columns(2)

    with col1:

        Tenth=st.number_input(
            "SSC Percentage",
            0.0,100.0,80.0
        )

        Twelfth=st.number_input(
            "12th Percentage",
            0.0,100.0,80.0
        )

        if stage!="Before B.Tech":

            First=st.number_input(
                "1st Semester",
                0.0,100.0,70.0
            )

            Second=st.number_input(
                "2nd Semester",
                0.0,100.0,70.0
            )

        if stage in ["2nd Year","3rd Year","Final Year"]:

            Third=st.number_input(
                "3rd Semester",
                0.0,100.0,70.0
            )

            Fourth=st.number_input(
                "4th Semester",
                0.0,100.0,70.0
            )

        if stage in ["3rd Year","Final Year"]:

            Fifth=st.number_input(
                "5th Semester",
                0.0,100.0,70.0
            )

            Sixth=st.number_input(
                "6th Semester",
                0.0,100.0,70.0
            )

        if stage=="Final Year":

            Seventh=st.number_input(
                "7th Semester",
                0.0,100.0,70.0
            )

            Eighth=st.number_input(
                "8th Semester",
                0.0,100.0,70.0
            )

    with col2:

        Attendance=st.number_input(
            "Attendance (%)",
            0.0,100.0,80.0
        )

        Backlogs=st.number_input(
            "Backlogs",
            0,10,0
        )

        Projects=st.number_input(
            "Projects Completed",
            0,20,2
        )

        Cocurricular=st.number_input(
            "Co-Curricular Score",
            0,200,40
        )

    predict=st.button(
        "🚀 Predict Student Category",
        use_container_width=True
    )

    if predict:

        entered_marks = [
            m for m in
            [Tenth, Twelfth, First, Second, Third, Fourth, Fifth, Sixth, Seventh, Eighth]
            if m > 0
        ]
        entered_academic_avg = (
            sum(entered_marks) / len(entered_marks) if entered_marks else 0
        )

        new_student = pd.DataFrame({
            "Academic_Avg": [entered_academic_avg],
            "CoCurricular_Score": [Cocurricular]
        })
        # ==================================================
        # ML PREDICTION
        # ==================================================

        prediction = model.predict(new_student)[0]

        probabilities = model.predict_proba(new_student)[0]
        confidence = max(probabilities) * 100

        category_map = {
            0: "Slow Learner",
            1: "Average Learner",
            2: "Fast Learner"
        }

        category = category_map[prediction]

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 0:
                st.error("🔴 Slow Learner")

            elif prediction == 1:
                st.warning("🟡 Average Learner")

            else:
                st.success("🟢 Fast Learner")

        with col2:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )
        # ==================================================
        # PERFORMANCE TREND
        # ==================================================

        graph_labels = ["SSC", "12th"]
        graph_marks = [Tenth, Twelfth]

        if stage != "Before B.Tech":
            graph_labels.extend(["1st Sem", "2nd Sem"])
            graph_marks.extend([First, Second])

        if stage in ["2nd Year", "3rd Year", "Final Year"]:
            graph_labels.extend(["3rd Sem", "4th Sem"])
            graph_marks.extend([Third, Fourth])

        if stage in ["3rd Year", "Final Year"]:
            graph_labels.extend(["5th Sem", "6th Sem"])
            graph_marks.extend([Fifth, Sixth])

        if stage == "Final Year":
            graph_labels.extend(["7th Sem", "8th Sem"])
            graph_marks.extend([Seventh, Eighth])

        graph_df = pd.DataFrame({
            "Exam": graph_labels,
            "Marks": graph_marks
        })

        fig = px.line(
            graph_df,
            x="Exam",
            y="Marks",
            markers=True,
            title="Student Performance Trend"
        )

        fig.update_layout(
            yaxis_title="Marks (%)",
            xaxis_title="Academic Progress"
        )

        st.plotly_chart(fig, use_container_width=True)
        # ============================================
        # PERFORMANCE ANALYSIS
        # ============================================

        st.markdown("---")
        st.subheader("📈 Performance Analysis")

        if len(graph_marks) >= 2:

          if graph_marks[-1] > graph_marks[-2]:
            st.success("📈 Performance is improving.")

          elif graph_marks[-1] < graph_marks[-2]:
             st.warning("📉 Performance is declining.")

          else:
             st.info("➖ Performance is stable.")

        elif len(graph_marks) == 1:
           st.info("Only one semester is available for analysis.")

        else:
           st.info("No semester performance available.")
        
        # ==================================================
        # PERFORMANCE ANALYSIS
        # ==================================================

        semester_marks = [m for m in [First, Second, Third, Fourth,
                              Fifth, Sixth, Seventh, Eighth] if m > 0]

        if semester_marks:
            average_sem = sum(semester_marks) / len(semester_marks)
        else:
            average_sem = (Tenth + Twelfth) / 2
        valid_marks = [m for m in graph_marks if m > 0]
        highest = max(valid_marks)
        lowest = min(valid_marks)
        latest = valid_marks[-1]

        st.subheader("📊 Performance Analysis")

        c1, c2, c3 = st.columns(3)

        c1.metric("Average", f"{average_sem:.2f}%")
        c2.metric("Highest", f"{highest}%")
        c3.metric("Latest Score", f"{latest}%")

        # ==================================================
        # WHY THIS CATEGORY?
        # ==================================================

        st.subheader(" Why this Prediction?")

        reason = []

        if latest >= 80:
            reason.append("Latest semester performance is excellent.")

        elif latest >= 70:
            reason.append("Latest semester performance is good.")

        elif latest >= 60:
            reason.append("Latest semester performance is satisfactory.")

        else:
            reason.append("Latest semester performance is poor.")

        if Attendance >= 75:
            reason.append("Attendance is satisfactory.")
        else:
            reason.append("Low attendance affects learning.")

        if Backlogs == 0:
            reason.append("Student has no backlogs.")
        else:
            reason.append(f"{Backlogs} backlog(s) found.")

        if Projects >= 2:
           reason.append("Student has good project experience.")

        if Cocurricular >= 40:
            reason.append("Student actively participates in co-curricular activities.")

        for r in reason:
            st.write("✔", r)

        
        # ==================================================
        # ACADEMIC RECOMMENDATION
        # ==================================================

        st.markdown("---")
        st.subheader("💡 Suggested Improvements")

        tips = []

        if latest < 75:
            tips.append("Improve semester marks through regular practice.")

        if Attendance < 75:
           tips.append("Maintain attendance above 75%.")

        if Backlogs > 0:
            tips.append("Clear all pending backlogs.")

        if Projects < 2:
           tips.append("Participate in more academic projects.")

        if Cocurricular < 40:
           tips.append("Join workshops, hackathons and technical events.")

        if not tips:
           tips.append("Maintain the current academic performance.")

        for t in tips:
            st.write("•", t)
        
        # ==================================================
        # FINAL SUMMARY
        # ==================================================

        st.markdown("---")
        st.success(
            f"""
### Prediction Summary

**Predicted Category:** {category}

**Model Confidence:** {confidence:.2f}%

The Decision Tree analyzed the student's academic history,
attendance, projects, backlogs and co-curricular activities
to predict the student's learning category.
"""
        )
# ==================================================
# CO-CURRICULAR ANALYTICS
# ==================================================
if menu == "🏆 Co-Curricular Analytics":

    section_header("Co-Curricular Analytics", eyebrow="Activity Insights", subtitle="Participation across hackathons, workshops, certifications and more")

    activity_cols = [

        "Hackathons",
        "Ideathons",
        "Coding_Contests",
        "Workshops",
        "Certifications",
        "Sports",
        "Projects"
    ]

    activity_data = filtered_df[
        activity_cols
    ].sum().reset_index()

    activity_data.columns = ["Activity", "Count"]

    fig = px.bar(
        activity_data,
        x="Activity",
        y="Count",
        title="Student Participation Activities"
    )

    st.plotly_chart(fig, use_container_width=True)

    # DEPARTMENT WISE PERFORMANCE
    st.markdown("<div class='section-eyebrow' style='margin-top:10px;'>🏫 DEPARTMENT WISE ACTIVITY PERFORMANCE</div>", unsafe_allow_html=True)

    dept_activity = filtered_df.groupby('Department')[

        [
            "Hackathons",
            "Ideathons",
            "Coding_Contests",
            "Workshops",
            "Certifications",
            "Sports",
            "Projects"
        ]

    ].sum().reset_index()

    dept_activity_melted = dept_activity.melt(

        id_vars="Department",

        var_name="Activity",

        value_name="Count"
    )

    fig_dept = px.bar(

        dept_activity_melted,

        x="Department",

        y="Count",

        color="Activity",

        barmode="group",

        title="Department Wise Co-Curricular Performance"
    )

    st.plotly_chart(fig_dept, use_container_width=True)

# ==================================================
# REPORTS
# ==================================================
if menu == "📄 Reports":

    section_header("Academic Report Center", eyebrow="Reports", subtitle="Generate a polished PDF performance report for any student")

    st.markdown('<div class="report-card-label">🔎 Look up a student</div>', unsafe_allow_html=True)
    Student_ID = st.text_input("Enter Student ID")

    if Student_ID:

       student = filtered_df[
           filtered_df['Student_ID'].astype(str).str.upper()
           == Student_ID.upper()
       ]

       if student.empty:
         st.error("Student ID not found")

       else:
        st.info("Enter a Student ID and click Generate Report")

        if st.button("Generate Report"):

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # ---------------- TITLE ----------------
            logo_path = "assets/college_logo.png"
            if not os.path.exists(logo_path):
                logo_path = "assets/college_logo.jpg"

            if os.path.exists(logo_path):
               pdf.image(logo_path, x=10, y=8, w=25)

            pdf.set_y(12)

            pdf.set_font("Arial", "B", 20)
            pdf.cell(0, 10, "ADITYA UNIVERSITY", ln=True, align="C")

            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "STUDENT MONITORING REPORT", ln=True, align="C")

            
            pdf.ln(8)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

            # ---------------- STUDENT INFO ----------------

            pdf.set_font("Arial","B",14)

            pdf.cell(190,10,"Student Information",ln=True)

            pdf.set_font("Arial","",12)

            pdf.cell(190,8,f"Name : {student['Name'].values[0]}",ln=True)

            pdf.cell(190,8,f"Student ID : {student['Student_ID'].values[0]}",ln=True)

            pdf.cell(190,8,f"Department : {student['Department'].values[0]}",ln=True)

            pdf.cell(190,8,f"Category : {student['Category'].values[0]}",ln=True)

            pdf.cell(190,8,f"Risk Level : {student['Risk'].values[0]}",ln=True)

            pdf.cell(190,8,f"Risk Score : {student['Risk_Score'].values[0]}/100",ln=True)

            pdf.ln(5)

            pdf.line(10,pdf.get_y(),200,pdf.get_y())

            pdf.ln(5)
            # ---------------- ACADEMIC PERFORMANCE ----------------

            # Require stage selection
            if stage_filter == "All":
                st.warning("⚠ Please select an Academic Stage from the sidebar before generating the report.")
                st.stop()

            pdf.set_font("Arial", "B", 14)
            pdf.cell(190, 8, "Academic Performance", ln=True)

            # Table Header
            pdf.set_font("Arial", "B", 11)
            pdf.cell(95, 8, "Academic Item", border=1, align="C")
            pdf.cell(95, 8, "Score", border=1, align="C")
            pdf.ln()

            pdf.set_font("Arial", "", 10)

            # Columns that should not appear in the table
            exclude_cols = [
                "Student_ID",
                 "Name",
                 "Department"
            ]

            # Columns according to selected stage
            report_columns = [
                 col for col in stage_columns[stage_filter]
                 if col not in exclude_cols and col in student.columns
            ]

            # Common fields
            report_columns.extend([
                   "Attendance",
                   "Backlogs",
                   "Projects",
                   "CoCurricular_Score"
            ])

            # Display names
            column_names = {
               "SSC_Percentage": "SSC Percentage",
               "Twelfth_Percentage": "12th Percentage",
               "First_Mid": "1st Mid",
               "Second_Mid": "2nd Mid",
               "Third_Mid": "3rd Mid",
               "Fourth_Mid": "4th Mid",
               "Fifth_Mid": "5th Mid",
               "Sixth_Mid": "6th Mid",
               "Seventh_Mid": "7th Mid",
               "Eighth_Mid": "8th Mid",
               "First_Sem": "1st Semester",
               "Second_Sem": "2nd Semester",
               "Third_Sem": "3rd Semester",
               "Fourth_Sem": "4th Semester",
               "Fifth_Sem": "5th Semester",
               "Sixth_Sem": "6th Semester",
               "Seventh_Sem": "7th Semester",
               "Eighth_Sem": "8th Semester",
               "Attendance": "Attendance",
               "Backlogs": "Backlogs",
               "Projects": "Projects",
               "CoCurricular_Score": "Co-Curricular Score"
            }

            # Print table
            for col in report_columns:

               value = student[col].values[0]

               if col == "Attendance":
                    value = f"{value}%"

               pdf.cell(95, 7, column_names.get(col, col), border=1)
               pdf.cell(95, 7, str(value), border=1, align="C")
               pdf.ln()

            pdf.ln(4)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)
            
            
            
            # ---------------- RECOMMENDATIONS ----------------

            pdf.set_font("Arial","B",14)

            pdf.cell(190,10," Recommendations",ln=True)

            pdf.set_font("Arial","",12)

            recommendations = student['Recommendation'].values[0].split(",")

            for rec in recommendations:

                pdf.set_x(10)  # reset to left margin — multi_cell leaves x at the right edge otherwise
                pdf.multi_cell(0,8,f"* {rec.strip()}")

            pdf.ln(5)

            pdf.line(10,pdf.get_y(),200,pdf.get_y())

            pdf.ln(5)

            # ---------------- STATUS ----------------

            pdf.set_font("Arial","B",14)

            pdf.cell(190,10,"Overall Status",ln=True)

            pdf.set_font("Arial","",12)

            if student['Risk'].values[0] == "At Risk":

                status = "Needs Immediate Academic Support"

            else:

                status = "Student Performance is Satisfactory"

            pdf.multi_cell(0,8,status)
            # Line before footer
            pdf.ln(5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)
            
            pdf.set_font("Arial", "I", 10)
            pdf.cell(
                 0,
                 8,
                 "Generated by Aditya Student Monitoring System",
                 align="C"
            )

            report_path = f"reports/{student['Student_ID'].values[0]}.pdf"

            pdf.output(report_path)

            st.markdown(
                f"<div class='report-ready-banner'>✅ Report generated for "
                f"<strong>{student['Name'].values[0]}</strong> ({student['Student_ID'].values[0]})</div>",
                unsafe_allow_html=True
            )

            with open(report_path,"rb") as f:

                st.download_button(
                    "📥 Download Report",
                    f,
                    file_name=f"{student['Student_ID'].values[0]}.pdf"
                )

# ==================================================
# USER MANAGEMENT
# ==================================================
if menu == "👨‍💼 User Management":

    section_header("User Management", eyebrow="Admin", subtitle="Add, review, and remove faculty/mentor accounts")

    users_df = pd.read_sql(
       "SELECT * FROM users",
        conn
    )

    st.markdown("<div class='section-eyebrow' style='margin-top:6px;'>📋 EXISTING USERS</div>", unsafe_allow_html=True)

    u_search_col, u_export_col = st.columns([4, 1])
    with u_search_col:
        users_view_df = apply_table_search(users_df, key="users_table_search")
    with u_export_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        export_button(users_view_df, "users.csv", key="users_table_export")

    st.dataframe(users_view_df, use_container_width=True)

    st.markdown("<div class='section-eyebrow' style='margin-top:14px;'>➕ ADD USER</div>", unsafe_allow_html=True)

    new_username = st.text_input("Username")

    new_password = st.text_input(
        "Password",
        type="password"
    )

    new_role = st.selectbox(
        "Role",
        ["Faculty", "Mentor"]
    )

    new_department = st.selectbox(
        "Department",
        ["CSE", "AIML", "ECE", "MECH"]
    )

    assigned_students = st.text_input(
        "Assigned Student IDs"
    )

    if st.button("Add User"):

        if not new_username or not new_password:
            st.error("Username and Password are required.")

        else:
            new_user = pd.DataFrame({

                "username": [new_username],

                "password": [new_password],

                "role": [new_role],

                "department": [new_department]

                # NOTE: "assigned_students" is intentionally not stored — the live
                # `users` table has no such column, and access control is handled
                # purely by department match (see ROLE BASED ACCESS section above).
            })

            new_user.to_sql(
                "users",
                 conn,
                 if_exists="append",
                 index=False
            )

            st.success("✅ User Added")

    st.markdown("<div class='section-eyebrow' style='margin-top:14px;'>❌ REMOVE USER</div>", unsafe_allow_html=True)

    if users_df.empty:
        st.info("No users available to remove.")

    else:
        remove_user = st.selectbox(
            "Select User",
            users_df['username']
        )

        if st.button("Delete User"):

            cursor = conn.cursor()

            cursor.execute(
               "DELETE FROM users WHERE username = ?",
               (remove_user,)
            )

            conn.commit()
            st.success("✅ User Removed")

# ==================================================
# UPLOAD DATA
# ==================================================
if menu == "📂 Upload Data":

    section_header("Upload Student Data", eyebrow="Data Ingestion", subtitle="Drag and drop a CSV to add new student records to the database")

    st.markdown("""
    <div class="upload-card-hint">
        <span class="upload-card-hint-icon">📁</span>
        <div>
            <div class="upload-card-hint-title">Drop your CSV file below</div>
            <div class="upload-card-hint-sub">Must include a <code>Student_ID</code> column · duplicate IDs are automatically skipped</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=['csv'],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        progress = st.progress(0, text="Reading file…")

        new_data = pd.read_csv(uploaded_file)

        progress.progress(45, text="Validating columns…")

        if "Student_ID" not in new_data.columns:

            progress.empty()
            st.error("❌ CSV must contain a Student_ID column")

        else:

            progress.progress(70, text="Checking for duplicate Student IDs…")

            # Guard against duplicate Student_IDs: the database has no enforced
            # primary key, so without this check, re-uploading the same student
            # (or any overlapping ID) silently creates duplicate rows.
            existing_ids = pd.read_sql("SELECT Student_ID FROM students", conn)['Student_ID'].astype(str).str.upper()
            new_ids = new_data['Student_ID'].astype(str).str.upper()

            duplicate_mask = new_ids.isin(existing_ids)
            duplicates = new_data[duplicate_mask]
            clean_data = new_data[~duplicate_mask]

            if not duplicates.empty:
                st.warning(
                    f"⚠️ {len(duplicates)} row(s) skipped — Student_ID already exists: "
                    f"{', '.join(duplicates['Student_ID'].astype(str).unique())}"
                )

            if clean_data.empty:
                progress.empty()
                st.error("No new rows to add — all Student_IDs already exist.")

            else:
                progress.progress(90, text="Saving to database…")

                clean_data.to_sql(
                  "students",
                   conn,
                   if_exists="append",
                   index=False
                )

                progress.progress(100, text="Done!")
                progress.empty()

                st.markdown(
                    f"<div class='report-ready-banner'>✅ {len(clean_data)} row(s) uploaded successfully</div>",
                    unsafe_allow_html=True
                )

# ==================================================
# DOWNLOAD
# ==================================================
st.sidebar.download_button(
    "⬇ Download Dataset",
    filtered_df.to_csv(index=False),
    file_name="students.csv"
)
conn.close()