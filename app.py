import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from ultralytics import YOLO


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart AI Student Assistant",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

h1, h2, h3 {
    color: #1f2937;
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
""", unsafe_allow_html=True)


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
        return model

    except:

        model = YOLO("yolov8n.pt")
        return model


# =========================================================
# SIDEBAR - STUDENT PROFILE
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
        "An intelligent platform that combines Artificial Intelligence, "
        "Machine Learning, Recommendation Systems and Computer Vision."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown("""
        <div class="card">

        <h1>🤖</h1>

        <h2>AI Chatbot</h2>

        <p>
        Ask academic questions and get intelligent assistance.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

        <h1>🎯</h1>

        <h2>Recommendation</h2>

        <p>
        Get personalized learning recommendations.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="card">

        <h1>📷</h1>

        <h2>Image Detection</h2>

        <p>
        Upload images and detect objects using AI.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown("""
        <div class="card">

        <h1>📈</h1>

        <h2>Prediction</h2>

        <p>
        Predict student performance using Machine Learning.
        </p>

        </div>
        """, unsafe_allow_html=True)


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

    st.subheader("📈 Student Performance Overview")

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

    df = pd.DataFrame({
        "Subject": subjects,
        "Marks": marks
    })

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

        ax.set_title("Student Marks")

        ax.set_ylabel("Marks")

        plt.xticks(
            rotation=30
        )

        st.pyplot(fig)


# =========================================================
# AI CHATBOT
# =========================================================

elif page == "🤖 AI Chatbot":

    st.title("🤖 AI Academic Chatbot")

    st.write(
        "Ask me questions about programming, AI, Machine Learning and Engineering."
    )

    question = st.text_area(
        "💬 Ask your question"
    )

    if st.button("🚀 Ask AI"):

        if question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            question_lower = question.lower()

            if "python" in question_lower:

                answer = """
Python is a high-level programming language used in:

- Artificial Intelligence
- Machine Learning
- Data Science
- Web Development
- Automation

Important libraries include:

NumPy, Pandas, Matplotlib, Scikit-learn and TensorFlow.
"""

            elif "machine learning" in question_lower:

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

It is a supervised Machine Learning algorithm used for:

- Classification
- Regression

SVM tries to find the best decision boundary separating different classes.
"""

            elif "neural network" in question_lower:

                answer = """
A Neural Network is inspired by the human brain.

It consists mainly of:

Input Layer
Hidden Layers
Output Layer

It is widely used in:

- Image Recognition
- Speech Recognition
- Prediction
- Natural Language Processing
"""

            elif "arduino" in question_lower:

                answer = """
Arduino is a microcontroller development platform.

It can be used to control:

- Sensors
- Motors
- LEDs
- Displays
- Communication Modules
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

            st.success("🤖 AI Answer")

            st.write(answer)

            st.session_state.history.append({

                "Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "Type": "Chatbot",

                "Input": question,

                "Result": answer

            })


# =========================================================
# RECOMMENDATION SYSTEM
# =========================================================

elif page == "🎯 Recommendation":

    st.title("🎯 Personalized Learning Recommendation")

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

    if st.button("🎯 Get Recommendation"):

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

        st.subheader(
            "📚 Recommended Topics"
        )

        for topic in recommendations[interest]:

            st.success(
                f"✅ {topic}"
            )

        st.info(
            f"Recommended study time: {study_hours} hours per week"
        )

        st.session_state.history.append({

            "Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "Type": "Recommendation",

            "Input": interest,

            "Result": ", ".join(
                recommendations[interest]
            )

        })


# =========================================================
# IMAGE DETECTION
# =========================================================

elif page == "📷 Image Detection":

    st.title("🔍 AI Image Detection")

    st.write(
        "Upload an image and the AI model will detect real objects."
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

        st.success(
            "Image uploaded successfully!"
        )

        if st.button("🔍 Detect Objects"):

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

                    detected_objects.append({

                        "Object": class_name,

                        "Confidence": round(
                            confidence * 100,
                            2
                        )

                    })

            if detected_objects:

                st.subheader(
                    "📋 Detected Objects"
                )

                detection_df = pd.DataFrame(
                    detected_objects
                )

                st.dataframe(
                    detection_df,
                    use_container_width=True,
                    hide_index=True
                )

                for obj in detected_objects:

                    st.success(
                        f"✅ {obj['Object']} - "
                        f"{obj['Confidence']}%"
                    )

                st.session_state.history.append({

                    "Time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                    "Type": "Image Detection",

                    "Input": uploaded_file.name,

                    "Result": ", ".join(
                        [
                            obj["Object"]
                            for obj in detected_objects
                        ]
                    )

                })

            else:

                st.warning(
                    "No objects were detected."
                )


# =========================================================
# STUDENT PREDICTION
# =========================================================

elif page == "📈 Student Prediction":

    st.title("📈 Student Performance Prediction")

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

    if st.button("📊 Predict Performance"):

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

            prediction = "Excellent Performance 🟢"

        elif score >= 50:

            prediction = "Good Performance 🟡"

        else:

            prediction = "Needs Improvement 🔴"

        st.metric(
            "Predicted Score",
            f"{score:.2f}%"
        )

        st.success(
            prediction
        )

        st.session_state.history.append({

            "Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "Type": "Student Prediction",

            "Input": f"Study Hours: {study_hours}",

            "Result": f"{prediction} - {score:.2f}%"

        })


# =========================================================
# HISTORY
# =========================================================

elif page == "🕘 History":

    st.title("🕘 Activity History")

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
            "🗑️ Clear History"
        ):

            st.session_state.history = []

            st.success(
                "History cleared successfully."
            )

            st.rerun()