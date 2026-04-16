import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import plotly.graph_objects as go
import joblib

# Page style
st.set_page_config(page_title="Bank Marketing Predictor", layout="wide")
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    body {
        background-color: #f8fafc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background: #f8fafc;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.7rem 1rem;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    .stSidebar {
        background-color: #ffffff;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        margin-top: 1rem;
    }
    .stSidebar .stSelectbox, .stSidebar .stNumberInput, .stSidebar .stSlider {
        margin-bottom: 1rem;
    }
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stMetric {
        background: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stNumberInput>div>div>input {
        border-radius: 0.5rem;
        border: 2px solid #e1e5e9;
        padding: 0.5rem;
        transition: border-color 0.3s ease;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus, .stNumberInput>div>div>input:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
    }
    .stSubheader {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .css-1d391kg {
        background-color: #f8fafc !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main header
st.markdown(
    """
    <div class="main-header">
        <h1>🏦 Bank Marketing Predictor</h1>
        <p>Predict customer subscription to term deposits using advanced machine learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load and prepare data with BEST model
@st.cache_resource
def load_data_and_train():
    raw_df = pd.read_csv('bank_dataset.csv')
    
    # Preprocessing
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    label_encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        raw_df[col] = encoder.fit_transform(raw_df[col])
        label_encoders[col] = encoder
    
    raw_df['deposit'] = raw_df['deposit'].map({'yes': 1, 'no': 0})
    
    X = raw_df.drop('deposit', axis=1)
    y = raw_df['deposit']
    
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    scaler = joblib.load('scaler.pkl')
    X[numeric_cols] = scaler.transform(X[numeric_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load pre-trained stacked model
    try:
        best_model = joblib.load('best_final_model.pkl')
        model_source = "✅ Stacked Ensemble Model"
    except Exception as e:
        st.error(f"Could not load model: {e}")
        raise
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'model': best_model,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'X_columns': X.columns,
        'numeric_cols': numeric_cols,
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': cr,
        'X_test': X_test,
        'y_test': y_test,
        'model_source': model_source
    }

data_cache = load_data_and_train()
model = data_cache['model']
label_encoders = data_cache['label_encoders']
scaler = data_cache['scaler']
X_columns = data_cache['X_columns']
numeric_cols = data_cache['numeric_cols']

# Get unique values for dropdowns
raw_df = pd.read_csv('bank_dataset.csv')
jobs = sorted(raw_df['job'].unique().tolist())
marital_status = sorted(raw_df['marital'].unique().tolist())
education = sorted(raw_df['education'].unique().tolist())
contacts = sorted(raw_df['contact'].unique().tolist())
months = sorted(raw_df['month'].unique().tolist())
poutcomes = sorted(raw_df['poutcome'].unique().tolist())
defaults = sorted(raw_df['default'].unique().tolist())
housings = sorted(raw_df['housing'].unique().tolist())
loans = sorted(raw_df['loan'].unique().tolist())

# Sidebar for input - Only ask for most important features
st.sidebar.header("📝 Key Customer Details")
st.sidebar.markdown("🔍 **Simplified Input:** We only ask for the most important factors that affect predictions.")

# Most important features based on model analysis
col1, col2 = st.sidebar.columns(2)
with col1:
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35, step=1, key="age_input")

with col2:
    balance = st.sidebar.number_input("Bank Balance (₹)", min_value=0, max_value=5000000, value=150000, step=10000, key="balance_input")

# Key predictive features
job = st.sidebar.selectbox("Job", jobs, index=jobs.index('admin.') if 'admin.' in jobs else 0, key="job_input")
housing = st.sidebar.selectbox("Housing Loan", housings, index=housings.index('yes') if 'yes' in housings else 0, key="housing_input")
contact = st.sidebar.selectbox("Contact Type", contacts, index=contacts.index('cellular') if 'cellular' in contacts else 0, key="contact_input")

# Duration is very important
duration = st.sidebar.slider("Contact Duration (seconds)", min_value=0, max_value=2000, value=250, step=10, key="duration_input")

# Previous outcome is important
poutcome = st.sidebar.selectbox("Previous Campaign Outcome", poutcomes, index=poutcomes.index('unknown') if 'unknown' in poutcomes else 0, key="poutcome_input")

# Previous contacts count
previous = st.sidebar.slider("Previous Contacts", min_value=0, max_value=10, value=0, step=1, key="previous_input")

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip:** These are the most important factors for accurate predictions!")

predict_button = st.sidebar.button("Predict", key="predict_button")

if predict_button:
    # Convert rupees to EUR equivalent for model (1 EUR ≈ 90 INR, so divide by 100 for rough scaling)
    balance_scaled = balance / 100
    
    # Prepare input for prediction with defaults for non-essential features
    input_data = {
        'age': age,
        'job': label_encoders['job'].transform([job])[0],
        'marital': label_encoders['marital'].transform(['married'])[0],  # Default
        'education': label_encoders['education'].transform(['secondary'])[0],  # Default
        'default': label_encoders['default'].transform(['no'])[0],  # Default
        'balance': balance_scaled,  # Converted to EUR scale
        'housing': label_encoders['housing'].transform([housing])[0],
        'loan': label_encoders['loan'].transform(['no'])[0],  # Default
        'contact': label_encoders['contact'].transform([contact])[0],
        'day': 15,  # Default
        'month': label_encoders['month'].transform(['may'])[0],  # Default
        'duration': duration,
        'campaign': 1,  # Default
        'pdays': -1,  # Default
        'previous': previous,
        'poutcome': label_encoders['poutcome'].transform([poutcome])[0]
    }

    # Create input dataframe
    input_df = pd.DataFrame([input_data])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Make prediction
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0]

    # Display results
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.success("✅ Customer will SUBSCRIBE")
            st.metric("Confidence", f"{confidence[1]*100:.1f}%")
            
            st.balloons()
        else:
            st.error("❌ Customer will NOT subscribe")
            st.metric("Confidence", f"{confidence[0]*100:.1f}%")
        
        # Customer summary in a nice box
        st.subheader("👤 Customer Summary")
        summary_html = f"""
        <div style="background-color: white; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); margin: 1rem 0;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div><strong>Age:</strong> {age} years</div>
                <div><strong>Job:</strong> {job}</div>
                <div><strong>Bank Balance:</strong> ₹{balance:,}</div>
                <div><strong>Housing Loan:</strong> {housing}</div>
                <div><strong>Contact Type:</strong> {contact}</div>
                <div><strong>Contact Duration:</strong> {duration} seconds</div>
                <div><strong>Previous Contacts:</strong> {previous}</div>
                <div><strong>Previous Outcome:</strong> {poutcome}</div>
            </div>
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)

    with col2:
        with st.expander("Show advanced evaluation"):
            st.metric("Model Accuracy", f"{data_cache['accuracy']*100:.2f}%")
            st.write("**Confusion Matrix:**")
            cm_df = pd.DataFrame(
                data_cache['confusion_matrix'],
                columns=['Predicted: No', 'Predicted: Yes'],
                index=['Actual: No', 'Actual: Yes']
            )
            st.dataframe(cm_df, width='stretch')

            st.write("**Classification Report:**")
            cr_df = pd.DataFrame(data_cache['classification_report']).transpose()
            st.dataframe(cr_df.round(4), width='stretch')

            st.subheader("🎯 Prediction Confidence Breakdown")
            fig = go.Figure(data=[
                go.Bar(
                    x=['No Subscription', 'Subscription'],
                    y=[confidence[0]*100, confidence[1]*100],
                    marker=dict(color=['#FF6B6B', '#4ECDC4'])
                )
            ])
            fig.update_layout(
                title="Confidence Scores (%)",
                yaxis_title="Confidence (%)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')
else:
    st.info("Fill in the customer details and click the **Predict** button in the sidebar.")
