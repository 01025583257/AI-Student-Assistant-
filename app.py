import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from ultralytics import YOLO


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart AI Student Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fb;
    }

    .card {
        padding: 25px;
        border-radius: 18px;
        background-color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        min-height: 220px;
    }

    .card h2 {
        color: #1e3a8a;
    }

    .card p {
        color: #555;
        font-size: 16px;
    }

    .success-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
    }

    .info-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "student_data" not in st.session_state:
    st.session_state.student_data = {
        "name": "Student",
        "department": "Electronics Engineering",
        "level": "Fourth Year"
    }


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_yolo_model():

    try:
        model = YOLO("yolo11n.pt")

    except Exception:

        model = YOLO("yolov8n.pt")

    return model


# =========================================================
# ADD HISTORY FUNCTION
# =========================================================

def add_history(activity_type, user_input, result):

    st.session_state.history.append(
        {
            "Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "Type": activity_type,

            "Input": user_input,

            "Result": result
        }
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🎓 Student Profile")

    student_name = st.text_input(
        "Student Name",
        value=st.session_state.student_data["name"]
    )

    department = st.selectbox(
        "Department",
        [
            "Electronics Engineering",
            "Electrical Engineering",
            "Communication Engineering",
            "Computer Engineering",
            "Artificial Intelligence"
        ]
    )

    academic_level = st.selectbox(
        "Academic Level",
        [
            "First Year",
            "Second Year",
            "Third Year",
            "Fourth Year"
        ]
    )

    st.session_state.student_data = {
        "name": student_name,
        "department": department,
        "level": academic_level
    }

    st.divider()

    st.title("📌 Navigation")

    page = st.radio(
        "Choose Page",
        [
            "🏠 Home",
            "📊 Dashboard",
            "🤖 AI Chatbot",
            "🎯 Recommendation",
            "📷 Image Detection",
            "📈 Student Prediction",
            "🕘 History"
        ]
    )


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("🚀 Smart AI Student Assistant")

    st.subheader(
        f"Welcome {student_name}! 👋"
    )

    st.write(
        """
        An intelligent platform that combines:

        🤖 Artificial Intelligence

        📊 Machine Learning

        🎯 Recommendation Systems

        📷 Computer Vision

        📈 Student Performance Prediction
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="card">

            <h1>🤖</h1>

            <h2>AI Chatbot</h2>

            <p>
            Ask academic questions and get intelligent assistance.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            <h1>🎯</h1>

            <h2>Recommendation</h2>

            <p>
            Get personalized learning recommendations.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="card">

            <h1>📷</h1>

            <h2>Image Detection</h2>

            <p>
            Upload images and detect objects using AI.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="card">

            <h1>📈</h1>

            <h2>Prediction</h2>

            <p>
            Predict student performance using Machine Learning.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# DASHBOARD
# =========================================================

elif page == "📊 Dashboard":

    st.title("📊 Student Dashboard")

    st.subheader(
        f"Student: {student_name}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Department",
            department
        )

    with col2:

        st.metric(
            "Academic Level",
            academic_level
        )

    with col3:

        st.metric(
            "AI Features",
            "5"
        )

    with col4:

        st.metric(
            "Activities",
            len(st.session_state.history)
        )

    st.divider()

    st.subheader(
        "📈 Student Performance Overview"
    )

    subjects = [
        "Machine Learning",
        "Python",
        "Electronics",
        "Communication",
        "Data Science"
    ]

    marks = [
        85,
        78,
        72,
        88,
        91
    ]

    df = pd.DataFrame(
        {
            "Subject": subjects,
            "Marks": marks
        }
    )

    col1, col2 = st.columns(2)

    with col1:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        fig, ax = plt.subplots()

        ax.bar(
            df["Subject"],
            df["Marks"]
        )

        ax.set_title(
            "Student Marks"
        )

        ax.set_ylabel(
            "Marks"
        )

        ax.set_ylim(
            0,
            100
        )

        plt.xticks(
            rotation=30
        )

        plt.tight_layout()

        st.pyplot(fig)


# =========================================================
# AI CHATBOT
# =========================================================

elif page == "🤖 AI Chatbot":

    st.title("🤖 AI Academic Chatbot")

    st.write(
        """
        Ask questions about:

        Python

        Machine Learning

        Artificial Intelligence

        SVM

        Neural Networks

        Arduino

        Electronics

        Communication Systems
        """
    )

    question = st.text_area(
        "💬 Ask your question",
        height=150
    )

    if st.button(
        "🚀 Ask AI",
        use_container_width=True
    ):

        if question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            question_lower = question.lower()

            if "python" in question_lower:

                answer = """
Python is a high-level programming language.

It is widely used in:

- Artificial Intelligence
- Machine Learning
- Data Science
- Automation
- Web Development

Important libraries:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
"""

            elif (
                "machine learning" in question_lower
                or "ml" in question_lower
            ):

                answer = """
Machine Learning is a branch of Artificial Intelligence.

It allows computers to learn patterns from data.

Main types:

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning
"""

            elif "svm" in question_lower:

                answer = """
SVM means Support Vector Machine.

It is a supervised Machine Learning algorithm.

It is used for:

- Classification
- Regression

SVM finds the best decision boundary
between different classes.
"""

            elif (
                "neural network" in question_lower
                or "neural" in question_lower
            ):

                answer = """
A Neural Network is inspired by the human brain.

Main components:

- Input Layer
- Hidden Layers
- Output Layer

Applications:

- Image Recognition
- Speech Recognition
- Prediction
- NLP
"""

            elif "arduino" in question_lower:

                answer = """
Arduino is a microcontroller development platform.

It can control:

- Sensors
- Motors
- LEDs
- Displays
- Communication Modules
"""

            elif (
                "electronics" in question_lower
                or "communication" in question_lower
            ):

                answer = """
Electronics and Communication Engineering
includes many important topics such as:

- Digital Communication
- Analog Communication
- Signals and Systems
- Microcontrollers
- Embedded Systems
- Electronics Circuits
"""

            else:

                answer = """
I can help you with:

- Python
- Machine Learning
- Artificial Intelligence
- SVM
- Neural Networks
- Arduino
- Electronics
- Communication Systems

Try asking about one of these topics.
"""

            st.success(
                "🤖 AI Answer"
            )

            st.markdown(
                answer
            )

            add_history(
                "Chatbot",
                question,
                answer
            )


# =========================================================
# RECOMMENDATION SYSTEM
# =========================================================

elif page == "🎯 Recommendation":

    st.title(
        "🎯 Personalized Learning Recommendation"
    )

    st.write(
        "Answer a few questions to get personalized recommendations."
    )

    interest = st.selectbox(
        "What is your main interest?",
        [
            "Artificial Intelligence",
            "Machine Learning",
            "Computer Vision",
            "Data Science",
            "Embedded Systems",
            "Communication Systems"
        ]
    )

    skill_level = st.selectbox(
        "Your Skill Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    study_hours = st.slider(
        "Study Hours Per Week",
        1,
        40,
        10
    )

    if st.button(
        "🎯 Get Recommendation",
        use_container_width=True
    ):

        recommendations = {

            "Artificial Intelligence": [
                "Python",
                "Machine Learning",
                "Neural Networks",
                "Natural Language Processing"
            ],

            "Machine Learning": [
                "Python",
                "NumPy",
                "Pandas",
                "Scikit-learn",
                "Deep Learning"
            ],

            "Computer Vision": [
                "Python",
                "OpenCV",
                "YOLO",
                "CNN",
                "Image Processing"
            ],

            "Data Science": [
                "Python",
                "Pandas",
                "Data Visualization",
                "Statistics",
                "Machine Learning"
            ],

            "Embedded Systems": [
                "C/C++",
                "Arduino",
                "Microcontrollers",
                "Sensors",
                "Embedded Systems"
            ],

            "Communication Systems": [
                "Signals and Systems",
                "Digital Communication",
                "Modulation",
                "MATLAB",
                "Wireless Communication"
            ]

        }

        topics = recommendations[interest]

        st.subheader(
            "📚 Recommended Topics"
        )

        for topic in topics:

            st.success(
                f"✅ {topic}"
            )

        st.info(
            f"""
            Skill Level: {skill_level}

            Recommended Study Time:
            {study_hours} hours per week
            """
        )

        add_history(
            "Recommendation",
            interest,
            ", ".join(topics)
        )


# =========================================================
# IMAGE DETECTION
# =========================================================

elif page == "📷 Image Detection":

    st.title(
        "🔍 AI Image Detection"
    )

    st.write(
        """
        Upload an image and YOLO will detect objects
        using Computer Vision.
        """
    )

    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "🔍 Detect Objects",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "AI is detecting objects..."
                ):

                    model = load_yolo_model()

                    results = model(
                        image
                    )

                result_image = results[0].plot()

                st.subheader(
                    "🎯 Detection Result"
                )

                st.image(
                    result_image,
                    caption="Detected Objects",
                    use_container_width=True
                )

                detected_objects = []

                for result in results:

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0]
                        )

                        class_name = model.names[
                            class_id
                        ]

                        confidence = float(
                            box.conf[0]
                        )

                        detected_objects.append(
                            {
                                "Object": class_name,
                                "Confidence": round(
                                    confidence * 100,
                                    2
                                )
                            }
                        )

                if detected_objects:

                    detection_df = pd.DataFrame(
                        detected_objects
                    )

                    st.subheader(
                        "📋 Detected Objects"
                    )

                    st.dataframe(
                        detection_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    detected_names = [
                        obj["Object"]
                        for obj in detected_objects
                    ]

                    add_history(
                        "Image Detection",
                        uploaded_file.name,
                        ", ".join(
                            detected_names
                        )
                    )

                else:

                    st.warning(
                        "No objects were detected."
                    )

            except Exception as error:

                st.error(
                    f"Detection Error: {error}"
                )


# =========================================================
# STUDENT PREDICTION
# =========================================================

elif page == "📈 Student Prediction":

    st.title(
        "📈 Student Performance Prediction"
    )

    st.write(
        "Enter student information to predict academic performance."
    )

    study_hours = st.slider(
        "📚 Study Hours Per Day",
        0.0,
        12.0,
        5.0
    )

    attendance = st.slider(
        "📅 Attendance Percentage",
        0,
        100,
        80
    )

    previous_marks = st.slider(
        "📝 Previous Marks",
        0,
        100,
        70
    )

    assignments = st.slider(
        "📋 Assignments Completed",
        0,
        10,
        7
    )

    if st.button(
        "📊 Predict Performance",
        use_container_width=True
    ):

        score = (

            study_hours * 5

            + attendance * 0.3

            + previous_marks * 0.4

            + assignments * 2

        )

        score = min(
            score,
            100
        )

        if score >= 75:

            prediction = (
                "Excellent Performance 🟢"
            )

        elif score >= 50:

            prediction = (
                "Good Performance 🟡"
            )

        else:

            prediction = (
                "Needs Improvement 🔴"
            )

        st.metric(
            "Predicted Score",
            f"{score:.2f}%"
        )

        st.success(
            prediction
        )

        add_history(
            "Student Prediction",
            f"Study Hours: {study_hours}",
            f"{prediction} - {score:.2f}%"
        )


# =========================================================
# HISTORY
# =========================================================

elif page == "🕘 History":

    st.title(
        "🕘 Activity History"
    )

    if len(
        st.session_state.history
    ) == 0:

        st.info(
            "No activities yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button(
            "🗑️ Clear History",
            use_container_width=True
        ):

            st.session_state.history = []

            st.success(
                "History cleared successfully."
            )

            st.rerun()
