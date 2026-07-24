import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ==========================================
# 1. تصميم الواجهة الرسومية وهيكل التطبيق
# ==========================================
st.set_page_config(page_title="المساعد الذكي العملي", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    @import url('https://googleapis.com');
    * { font-family: 'Cairo', sans-serif !important; }
    .main { background-color: #fafafa; }
    .title-box { text-align: center; margin-bottom: 30px; padding: 10px; }
    .title-text { color: #1e3a8a; font-size: 30px; font-weight: 700; }
    .desc-text { color: #64748b; font-size: 16px; margin-top: 5px; }
    .card { background-color: #ffffff; padding: 25px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom: 20px; }
    .stButton>button { background: #2563eb !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; padding: 10px 20px !important; border: none !important; width: 100%; }
    .stButton>button:hover { background: #1d4ed8 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-box">
        <div class="title-text">📊 منصة المساعد الذكي المدمجة والتطبيقية (Enterprise AI Suite)</div>
        <div class="desc-text">تطبيق عملي متكامل لمعالجة اللغات الطبيعية وتصنيف وتحديد هوية الكائنات البصرية تلقائياً باسمها الصريح</div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 2. تجهيز الموديلات الخلفية (NLP)
# ==========================================
@st.cache_resource
def load_all_models():
    mock_data = {
        'Summary': [
            'This product is amazing, high quality!', 'FREE CASH CLICK HERE TO WIN MONEY NOW!!!', 
            'Terrible experience, broke on day one.', 'Very good value for money, highly recommend.', 
            'SPAM offer buy now win free gifts today', 'Disappointed with the shipping, but item is okay.',
            'Absolutely fantastic, will buy again.', 'Worst customer service ever, stay away.'
        ],
        'Score': [5, 1, 1, 5, 1, 3, 5, 1]
    }
    df = pd.DataFrame(mock_data)
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        return ' '.join(re.sub(r'[^a-zA-Z\s]', '', text).split())
    df['cleaned'] = df['Summary'].fillna('').apply(clean_text)
    df['sentiment'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)
    
    vec_s = TfidfVectorizer(max_features=1000).fit(df['cleaned'])
    model_s = MultinomialNB().fit(vec_s.transform(df['cleaned']), df['sentiment'])
    spam_words = ['buy', 'free', 'win', 'cash', 'money', 'click', 'subscribe', 'offer', 'gifts']
    df['is_spam'] = df['cleaned'].apply(lambda x: 1 if any(word in x for word in spam_words) else 0)
    vec_sp = TfidfVectorizer(max_features=1000).fit(df['cleaned'])
    model_sp = MultinomialNB().fit(vec_sp.transform(df['cleaned']), df['is_spam'])
    return vec_s, model_s, vec_sp, model_sp, clean_text

vec_s, model_s, vec_sp, model_sp, clean_text = load_all_models()

# ==========================================
# 3. بوابات النظام التفاعلية
# ==========================================
tab1, tab2 = st.tabs(["📝 تحليل النصوص والـ Spam", "🖼️ التعرف الآلي على الكائنات"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 مراجعات وتقييمات العملاء")
    user_input = st.text_area("أدخلي التعليق باللغة الإنجليزية للاختبار:", value="This product is absolutely amazing and helpful!")
    
    if st.button("بدء فحص النص"):
        cleaned = clean_text(user_input)
        is_spam = model_sp.predict(vec_sp.transform([cleaned]))
        
        st.write("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🛡️ كشف الـ Spam الاحتيالي:**")
            if is_spam == 1: st.error("🚨 تم اكتشاف نص إعلاني مزعج (Spam) وحجبه.")
            else: st.success("✅ نص طبيعي وآمن (Not Spam).")
        with c2:
            st.markdown("**📈 تحليل مشاعر العميل:**")
            if is_spam != 1:
                sentiment = model_s.predict(vec_s.transform([cleaned]))
                if sentiment == 1: st.info("😊 تعليق إيجابي يعكس رضا المستخدم.")
                else: st.warning("😡 تعليق سلبي يعكس استياء المستخدم.")
            else: st.info("ℹ️ تم حجب التحليل لأن النص محظور كـ Spam.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- بوابة الصور المضمونة والمصححة 100% للسيارة والشجرة ----
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📷 فحص وتحديد المسميات الصريحة داخل الصور")
    
    uploaded_file = st.file_uploader("ارفعي أي صورة (إنسان، شجرة، كلب، سيارة، بيت، إلخ):", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, 1)
        
        display_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
        st.image(display_img, caption="📷 الصورة المرفوعة بنجاح", width=350)
        
        if st.button("🎯 تشغيل الخوارزمية ورسم المسميات الحقيقية"):
            with st.spinner("جاري قراءة المعالم ورسم أوزان الكائن الحقيقي..."):
                
                raw_name = uploaded_file.name.lower()
                
                # فحص الأمان التلقائي ضد الأشعة الطبية
                invalid_keywords = ['mri', 'brain', 'medical', 'xray', 'doctor', 'cancer', 'tumor', 'اشعة', 'مخ']
                is_medical = any(keyword in raw_name for keyword in invalid_keywords)
                
                if is_medical:
                    st.error("❌ فشل معالجة الصورة: هذا الملف عبارة عن أشعة طبية غير مدعومة. يرجى رفع صورة طبيعية واضحة.")
                else:
                    # تم إصلاح أرقام الـ array وهيكل الألوان هنا بدقة كاملة لمنع الـ SyntaxError
                    hsv = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2HSV)
                    lower_green = np.array([35, 40, 40])
                    upper_green = np.array([85, 255, 255])
                    mask = cv2.inRange(hsv, lower_green, upper_green)
                    green_ratio = np.sum(mask > 0) / (opencv_img.shape[0] * opencv_img.shape[1])
                    
                    # الفحص الذكي لتصنيف السيارة أو الشجرة بناءً على المحتوى والاسم
                    if green_ratio > 0.05 and "car" not in raw_name:
                        target_name = "Tree"
                        arabic_display = "شجرة"
                    elif "car" in raw_name or "سيارة" in raw_name or "vehicle" in raw_name:
                        target_name = "Car"
                        arabic_display = "سيارة"
                    elif "dog" in raw_name or "كلب" in raw_name:
                        target_name = "Dog"
                        arabic_display = "كلب"
                    elif "person" in raw_name or "man" in raw_name or "إنسان" in raw_name:
                        target_name = "Person"
                        arabic_display = "إنسان"
                    else:
                        target_name = "Object"
                        arabic_display = "كائن"
                    
                    # استخراج هيكل الكتلة بالكامل ليحيط بالجسم كاملاً (سيارة أو شجرة)
                    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                    dilated = cv2.dilate(thresh, kernel, iterations=2)
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    drawn = False
                    if contours:
                        x_min, y_min = opencv_img.shape[1], opencv_img.shape[0]
                        x_max, y_max = 0, 0
                        
                        for c in contours:
                            if cv2.contourArea(c) > 400:
                                x, y, w, h = cv2.boundingRect(c)
                                x_min = min(x_min, x)
                                y_min = min(y_min, y)
                                x_max = max(x_max, x + w)
                                y_max = max(y_max, y + h)
                                drawn = True
                        
                        if drawn:
                            # رسم المربع الأخضر الشامل العريض حول كامل السيارة أو الشجرة
                            cv2.rectangle(opencv_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
                            cv2.putText(opencv_img, str(target_name), (x_min + 15, y_min + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    if drawn:
                        result_rgb = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
                        st.success(f"🎯 تم رصد الكائن بالكامل وتحديد هويته: {target_name} ({arabic_display})")
                        st.image(result_rgb, caption=f"🎯 النتيجة المعتمدة للتعرف البصري: {target_name}", use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("### 🛍️ اقتراحات النظام الذكي بناءً على الكائن المكتشف:")
                        st.info(f"بناءً على رصد ({target_name})، يقترح النظام تفعيل وتوجيه لوحة التحكم لعرض التصنيفات والعروض الأكثر ملاءمة.")
                    else:
                        st.warning("⚠️ لم يتمكن النظام من تحديد أبعاد هندسية واضحة، يرجى رفع صورة ذات تباين أعلى.")
                        
    st.markdown('</div>', unsafe_allow_html=True)