import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="E-commerce Customer Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2e8b57;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load and preprocess the e-commerce customer data"""
    try:
        # Try to load from the current directory
        df = pd.read_csv('E-commerce Customer Behavior.csv')
        
        # Handle missing/blank satisfaction values by normalizing to 'Unknown'
        df['Satisfaction Level'] = df['Satisfaction Level'].fillna('Unknown')
        df['Satisfaction Level'] = df['Satisfaction Level'].replace('', 'Unknown')
        
        # Log if any Unknown values were found
        unknown_count = (df['Satisfaction Level'] == 'Unknown').sum()
        if unknown_count > 0:
            print(f"ℹ️ Normalized {unknown_count} blank satisfaction values to 'Unknown'")

        # Add data validation (but keep all records including Unknown satisfaction)
        initial_count = len(df)

        # Additional validation for other fields
        df = df[
            (df['Age'] > 0) & (df['Age'] < 150) &
            (df['Total Spend'] >= 0) &
            (df['Items Purchased'] >= 0) &
            (df['Average Rating'] >= 0) & (df['Average Rating'] <= 5) &
            (df['Days Since Last Purchase'] >= 0)
        ]

        filtered_count = len(df)
        if initial_count != filtered_count:
            st.info(f"Data Quality: Filtered out {initial_count - filtered_count} invalid records. Using {filtered_count} valid customer records for analysis.")

    except FileNotFoundError:
        # Create sample data if file not found (for demo purposes)
        st.warning("Dataset file not found. Using sample data for demonstration.")
        np.random.seed(42)
        
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia']
        membership_types = ['Bronze', 'Silver', 'Gold', 'Premium']
        satisfaction_levels = ['Unsatisfied', 'Neutral', 'Satisfied', 'Unknown']
        
        df = pd.DataFrame({
            'Customer ID': range(1, 351),
            'Gender': np.random.choice(['Male', 'Female'], 350),
            'Age': np.random.randint(18, 70, 350),
            'City': np.random.choice(cities, 350),
            'Membership Type': np.random.choice(membership_types, 350),
            'Total Spend': np.random.uniform(50, 2000, 350).round(2),
            'Items Purchased': np.random.randint(1, 20, 350),
            'Average Rating': np.random.uniform(1, 5, 350).round(1),
            'Discount Applied': np.random.choice([True, False], 350),
            'Days Since Last Purchase': np.random.randint(1, 365, 350),
            'Satisfaction Level': np.random.choice(satisfaction_levels, 350)
        })
    
    return df

@st.cache_data
def create_customer_segments(df):
    """Create customer segments based on RFM analysis"""
    # Calculate quartiles
    recency_q = df['Days Since Last Purchase'].quantile([0.25, 0.5, 0.75])
    frequency_q = df['Items Purchased'].quantile([0.25, 0.5, 0.75])
    monetary_q = df['Total Spend'].quantile([0.25, 0.5, 0.75])
    
    def segment_customer(row):
        # Recency score (inverted)
        if row['Days Since Last Purchase'] <= recency_q[0.25]:
            r_score = 4
        elif row['Days Since Last Purchase'] <= recency_q[0.5]:
            r_score = 3
        elif row['Days Since Last Purchase'] <= recency_q[0.75]:
            r_score = 2
        else:
            r_score = 1
        
        # Frequency score
        if row['Items Purchased'] <= frequency_q[0.25]:
            f_score = 1
        elif row['Items Purchased'] <= frequency_q[0.5]:
            f_score = 2
        elif row['Items Purchased'] <= frequency_q[0.75]:
            f_score = 3
        else:
            f_score = 4
        
        # Monetary score
        if row['Total Spend'] <= monetary_q[0.25]:
            m_score = 1
        elif row['Total Spend'] <= monetary_q[0.5]:
            m_score = 2
        elif row['Total Spend'] <= monetary_q[0.75]:
            m_score = 3
        else:
            m_score = 4
        
        total_score = r_score + f_score + m_score
        
        if total_score >= 10:
            return 'Champions'
        elif total_score >= 8:
            return 'Loyal Customers'
        elif total_score >= 6:
            return 'Potential Loyalists'
        elif total_score >= 4:
            return 'At Risk'
        else:
            return 'Lost Customers'
    
    df['Customer_Segment'] = df.apply(segment_customer, axis=1)
    return df

@st.cache_data
def train_spending_model(df):
    """Train a model to predict customer spending"""
    # Prepare features
    df_model = df.copy()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_city = LabelEncoder()
    le_membership = LabelEncoder()
    le_satisfaction = LabelEncoder()
    
    df_model['Gender_encoded'] = le_gender.fit_transform(df_model['Gender'])
    df_model['City_encoded'] = le_city.fit_transform(df_model['City'])
    df_model['Membership_encoded'] = le_membership.fit_transform(df_model['Membership Type'])
    df_model['Satisfaction_encoded'] = le_satisfaction.fit_transform(df_model['Satisfaction Level'])
    df_model['Discount_numeric'] = df_model['Discount Applied'].astype(int)
    
    # Features and target
    features = ['Age', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase',
               'Gender_encoded', 'City_encoded', 'Membership_encoded', 
               'Satisfaction_encoded', 'Discount_numeric']
    
    X = df_model[features]
    y = df_model['Total Spend']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Calculate performance
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    
    return model, features, r2, mae, (le_gender, le_city, le_membership, le_satisfaction)

# Load data
df = load_data()
df = create_customer_segments(df)

# Train model
model, feature_names, model_r2, model_mae, encoders = train_spending_model(df)

def create_api_endpoints():
    """Create comprehensive API endpoints for external consumption with CORS support"""
    
    # Check if API mode is requested
    query_params = st.query_params
    
    if query_params.get('api') == 'true':
        endpoint = query_params.get('endpoint', 'overview')
        
        # Add CORS headers using Streamlit's HTML injection
        st.markdown("""
        <script>
        // Add CORS headers to the response
        if (window.parent && window.parent.postMessage) {
            window.parent.postMessage({
                type: 'cors-headers',
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                }
            }, '*');
        }
        </script>
        """, unsafe_allow_html=True)
        
        api_data = {}
        
        if endpoint == 'overview' or endpoint == 'all':
            api_data['overview'] = {
                "total_customers": int(len(df)),
                "total_revenue": float(df['Total Spend'].sum()),
                "average_order_value": float(df['Total Spend'].mean()),
                "satisfaction_rate": float((df['Satisfaction Level'] == 'Satisfied').mean() * 100),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        if endpoint == 'segments' or endpoint == 'all':
            segment_data = df.groupby('Customer_Segment').agg({
                'Customer ID': 'count',
                'Total Spend': ['mean', 'sum'],
                'Items Purchased': 'mean',
                'Days Since Last Purchase': 'mean',
                'Average Rating': 'mean'
            }).round(2)
            
            segments = {}
            for segment in segment_data.index:
                segments[segment] = {
                    "customer_count": int(segment_data.loc[segment, ('Customer ID', 'count')]),
                    "avg_spend": float(segment_data.loc[segment, ('Total Spend', 'mean')]),
                    "total_revenue": float(segment_data.loc[segment, ('Total Spend', 'sum')]),
                    "avg_items": float(segment_data.loc[segment, ('Items Purchased', 'mean')]),
                    "avg_days_since": float(segment_data.loc[segment, ('Days Since Last Purchase', 'mean')]),
                    "avg_rating": float(segment_data.loc[segment, ('Average Rating', 'mean')])
                }
            
            api_data['segments'] = segments
        
        if endpoint == 'revenue' or endpoint == 'all':
            discount_effect = df.groupby('Discount Applied')['Total Spend'].mean()
            discount_lift = ((discount_effect[True] - discount_effect[False]) / discount_effect[False] * 100) if True in discount_effect.index and False in discount_effect.index else 0
            
            city_revenue = df.groupby('City')['Total Spend'].sum().to_dict()
            membership_revenue = df.groupby('Membership Type')['Total Spend'].mean().to_dict()
            
            api_data['revenue'] = {
                "discount_effectiveness": float(discount_lift),
                "revenue_by_city": {str(k): float(v) for k, v in city_revenue.items()},
                "avg_spend_by_membership": {str(k): float(v) for k, v in membership_revenue.items()},
                "correlation_items_spend": float(df['Items Purchased'].corr(df['Total Spend']))
            }
        
        if endpoint == 'satisfaction' or endpoint == 'all':
            satisfaction_dist = df['Satisfaction Level'].value_counts().to_dict()
            rating_by_segment = df.groupby('Customer_Segment')['Average Rating'].mean().to_dict()
            
            api_data['satisfaction'] = {
                "distribution": {str(k): int(v) for k, v in satisfaction_dist.items()},
                "average_rating": float(df['Average Rating'].mean()),
                "rating_by_segment": {str(k): float(v) for k, v in rating_by_segment.items()}
            }
        
        if endpoint == 'predictions' or endpoint == 'all':
            feature_importance_dict = dict(zip(feature_names, model.feature_importances_.tolist()))
            
            # CLV calculation
            clv_data = df.groupby('Customer_Segment').agg({
                'Total Spend': 'mean',
                'Days Since Last Purchase': 'mean'
            })
            clv_estimates = (clv_data['Total Spend'] * (365 / clv_data['Days Since Last Purchase'].clip(lower=1))).to_dict()
            
            # Model predictions for accuracy
            le_gender, le_city, le_membership, le_satisfaction = encoders
            df_model = df.copy()
            df_model['Gender_encoded'] = le_gender.transform(df_model['Gender'])
            df_model['City_encoded'] = le_city.transform(df_model['City'])
            df_model['Membership_encoded'] = le_membership.transform(df_model['Membership Type'])
            df_model['Satisfaction_encoded'] = le_satisfaction.transform(df_model['Satisfaction Level'])
            df_model['Discount_numeric'] = df_model['Discount Applied'].astype(int)
            
            X = df_model[feature_names]
            predictions = model.predict(X)
            accuracy = r2_score(df['Total Spend'], predictions)
            
            api_data['predictions'] = {
                "model_accuracy": float(accuracy),
                "model_mae": float(model_mae),
                "feature_importance": {str(k): float(v) for k, v in feature_importance_dict.items()},
                "clv_by_segment": {str(k): float(v) for k, v in clv_estimates.items()}
            }
        
        if endpoint == 'insights' or endpoint == 'all':
            top_segment = df.groupby('Customer_Segment')['Total Spend'].sum().idxmax()
            best_membership = df.groupby('Membership Type')['Total Spend'].mean().idxmax()
            top_city = df.groupby('City')['Total Spend'].sum().idxmax()
            
            api_data['insights'] = {
                "top_revenue_segment": str(top_segment),
                "best_membership_type": str(best_membership),
                "top_revenue_city": str(top_city),
                "churn_risk_customers": int(len(df[df['Days Since Last Purchase'] > 180])),
                "high_value_customers": int(len(df[df['Total Spend'] > df['Total Spend'].quantile(0.8)])),
                "recommendations": [
                    f"Focus on {top_segment} segment for highest revenue impact",
                    f"Promote {best_membership} membership tier for better AOV",
                    f"Expand marketing in {top_city} for geographic growth",
                    "Implement churn prevention for at-risk customers",
                    "Develop loyalty programs for high-value customers"
                ]
            }
        
        # Return JSON with proper content type
        st.json(api_data)
        st.stop()

create_api_endpoints()

# Main title
st.markdown('<h1 class="main-header">🛒 E-commerce Customer Analytics Dashboard</h1>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔍 Dashboard Filters")

# Filter options
selected_segments = st.sidebar.multiselect(
    "Select Customer Segments:",
    options=df['Customer_Segment'].unique(),
    default=df['Customer_Segment'].unique()
)

selected_membership = st.sidebar.multiselect(
    "Select Membership Types:",
    options=df['Membership Type'].unique(),
    default=df['Membership Type'].unique()
)

selected_gender = st.sidebar.multiselect(
    "Select Gender:",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

age_range = st.sidebar.slider(
    "Age Range:",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

spend_range = st.sidebar.slider(
    "Total Spend Range ($):",
    min_value=float(df['Total Spend'].min()),
    max_value=float(df['Total Spend'].max()),
    value=(float(df['Total Spend'].min()), float(df['Total Spend'].max())),
    step=10.0
)

# Apply filters
filtered_df = df[
    (df['Customer_Segment'].isin(selected_segments)) &
    (df['Membership Type'].isin(selected_membership)) &
    (df['Gender'].isin(selected_gender)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1]) &
    (df['Total Spend'] >= spend_range[0]) &
    (df['Total Spend'] <= spend_range[1])
]

# Main dashboard tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "👥 Customer Segments", "💰 Revenue Analysis", "😊 Satisfaction", "🔮 Predictions"])

with tab1:
    st.header("📊 Business Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Total Customers",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df)} from total"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Total Revenue",
            value=f"${filtered_df['Total Spend'].sum():,.2f}",
            delta=f"${filtered_df['Total Spend'].sum() - df['Total Spend'].sum():,.2f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Avg Order Value",
            value=f"${filtered_df['Total Spend'].mean():.2f}",
            delta=f"${filtered_df['Total Spend'].mean() - df['Total Spend'].mean():.2f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        satisfaction_rate = (filtered_df['Satisfaction Level'] == 'Satisfied').mean() * 100
        total_satisfaction_rate = (df['Satisfaction Level'] == 'Satisfied').mean() * 100
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Satisfaction Rate",
            value=f"{satisfaction_rate:.1f}%",
            delta=f"{satisfaction_rate - total_satisfaction_rate:.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(
            filtered_df, 
            x='Age', 
            nbins=20,
            title="Customer Age Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        fig_age.update_layout(showlegend=False)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Spending distribution
        fig_spend = px.histogram(
            filtered_df, 
            x='Total Spend', 
            nbins=20,
            title="Spending Distribution",
            color_discrete_sequence=['#ff7f0e']
        )
        fig_spend.update_layout(showlegend=False)
        st.plotly_chart(fig_spend, use_container_width=True)
    
    # Geographic and demographic analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # City analysis
        city_data = filtered_df.groupby('City').agg({
            'Total Spend': 'sum',
            'Customer ID': 'count'
        }).reset_index()
        city_data.columns = ['City', 'Total_Revenue', 'Customer_Count']
        
        fig_city = px.bar(
            city_data, 
            x='City', 
            y='Total_Revenue',
            title="Revenue by City",
            color='Customer_Count',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_city, use_container_width=True)
    
    with col2:
        # Gender and membership analysis
        gender_membership = filtered_df.groupby(['Gender', 'Membership Type']).size().reset_index(name='Count')
        
        fig_gender = px.sunburst(
            gender_membership,
            path=['Gender', 'Membership Type'],
            values='Count',
            title="Customer Distribution: Gender & Membership"
        )
        st.plotly_chart(fig_gender, use_container_width=True)

with tab2:
    st.header("👥 Customer Segmentation Analysis")
    
    # Segment overview
    segment_summary = filtered_df.groupby('Customer_Segment').agg({
        'Customer ID': 'count',
        'Total Spend': ['mean', 'sum'],
        'Items Purchased': 'mean',
        'Days Since Last Purchase': 'mean',
        'Average Rating': 'mean'
    }).round(2)
    
    segment_summary.columns = ['Count', 'Avg_Spend', 'Total_Revenue', 'Avg_Items', 'Avg_Days_Since', 'Avg_Rating']
    segment_summary['Revenue_Share'] = (segment_summary['Total_Revenue'] / segment_summary['Total_Revenue'].sum() * 100).round(1)
    
    st.subheader("📈 Segment Performance Summary")
    st.dataframe(segment_summary, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment distribution
        segment_counts = filtered_df['Customer_Segment'].value_counts()
        fig_segments = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Segment Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_segments, use_container_width=True)
    
    with col2:
        # Segment revenue contribution
        fig_revenue = px.bar(
            x=segment_summary.index,
            y=segment_summary['Total_Revenue'],
            title="Revenue by Customer Segment",
            color=segment_summary['Total_Revenue'],
            color_continuous_scale='viridis'
        )
        fig_revenue.update_xaxes(title="Customer Segment")
        fig_revenue.update_yaxes(title="Total Revenue ($)")
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Segment characteristics heatmap
    st.subheader("🔥 Segment Characteristics Heatmap")
    
    # Normalize data for heatmap
    heatmap_data = segment_summary[['Avg_Spend', 'Avg_Items', 'Avg_Rating']].copy()
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    
    fig_heatmap = px.imshow(
        heatmap_normalized.T,
        x=heatmap_normalized.index,
        y=heatmap_normalized.columns,
        color_continuous_scale='RdYlBu_r',
        title="Normalized Segment Characteristics"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Actionable insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("💡 Segment Insights & Recommendations")
    
    top_segment = segment_summary.loc[segment_summary['Total_Revenue'].idxmax()]
    highest_value_segment = segment_summary.loc[segment_summary['Avg_Spend'].idxmax()]
    
    st.write(f"🏆 **Highest Revenue Segment**: {top_segment.name} (${top_segment['Total_Revenue']:,.2f})")
    st.write(f"💎 **Highest Value Customers**: {highest_value_segment.name} (${highest_value_segment['Avg_Spend']:.2f} avg)")
    
    # Strategic recommendations
    st.write("**📋 Strategic Recommendations:**")
    for segment in segment_summary.index:
        if segment == 'Champions':
            st.write(f"• **{segment}**: Maintain loyalty with exclusive offers and early access to new products")
        elif segment == 'Loyal Customers':
            st.write(f"• **{segment}**: Upsell premium products and encourage referrals")
        elif segment == 'Potential Loyalists':
            st.write(f"• **{segment}**: Implement loyalty programs and personalized recommendations")
        elif segment == 'At Risk':
            st.write(f"• **{segment}**: Launch reactivation campaigns with special discounts")
        elif segment == 'Lost Customers':
            st.write(f"• **{segment}**: Win-back campaigns with significant incentives")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.header("💰 Revenue Analysis")
    
    # Revenue metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        discount_effect = filtered_df.groupby('Discount Applied')['Total Spend'].mean()
        discount_lift = ((discount_effect[True] - discount_effect[False]) / discount_effect[False] * 100) if False in discount_effect.index and True in discount_effect.index else 0
        
        st.metric(
            label="Discount Effectiveness",
            value=f"{discount_lift:.1f}%",
            help="Revenue lift from applying discounts"
        )
    
    with col2:
        membership_revenue = filtered_df.groupby('Membership Type')['Total Spend'].mean()
        best_membership = membership_revenue.idxmax()
        
        st.metric(
            label="Best Membership Type",
            value=best_membership,
            delta=f"${membership_revenue[best_membership]:.2f} avg"
        )
    
    with col3:
        correlation = filtered_df['Items Purchased'].corr(filtered_df['Total Spend'])
        st.metric(
            label="Items-Revenue Correlation",
            value=f"{correlation:.3f}",
            help="Correlation between items purchased and total spend"
        )
    
    # Revenue analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by membership type
        membership_stats = filtered_df.groupby('Membership Type').agg({
            'Total Spend': ['mean', 'sum', 'count']
        }).round(2)
        membership_stats.columns = ['Avg_Spend', 'Total_Revenue', 'Customer_Count']
        
        fig_membership = px.bar(
            x=membership_stats.index,
            y=membership_stats['Avg_Spend'],
            title="Average Spending by Membership Type",
            color=membership_stats['Customer_Count'],
            color_continuous_scale='plasma'
        )
        fig_membership.update_xaxes(title="Membership Type")
        fig_membership.update_yaxes(title="Average Spend ($)")
        st.plotly_chart(fig_membership, use_container_width=True)
    
    with col2:
        # Discount vs no discount analysis
        discount_analysis = filtered_df.groupby(['Discount Applied', 'Membership Type'])['Total Spend'].mean().reset_index()
        
        fig_discount = px.bar(
            discount_analysis,
            x='Membership Type',
            y='Total Spend',
            color='Discount Applied',
            barmode='group',
            title="Spending: Discount vs No Discount",
            color_discrete_map={True: '#ff7f0e', False: '#1f77b4'}
        )
        st.plotly_chart(fig_discount, use_container_width=True)
    
    # Spending patterns analysis
    st.subheader("📈 Spending Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age vs Spending
        fig_age_spend = px.scatter(
            filtered_df,
            x='Age',
            y='Total Spend',
            color='Customer_Segment',
            size='Items Purchased',
            title="Age vs Spending Pattern",
            hover_data=['Average Rating', 'Days Since Last Purchase']
        )
        st.plotly_chart(fig_age_spend, use_container_width=True)
    
    with col2:
        # Items vs Spending
        fig_items_spend = px.scatter(
            filtered_df,
            x='Items Purchased',
            y='Total Spend',
            color='Membership Type',
            size='Average Rating',
            title="Items Purchased vs Total Spend",
            hover_data=['Age', 'Days Since Last Purchase']
        )
        st.plotly_chart(fig_items_spend, use_container_width=True)
    
    # Revenue insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("💡 Revenue Insights")
    
    total_revenue = filtered_df['Total Spend'].sum()
    avg_order = filtered_df['Total Spend'].mean()
    top_city_revenue = filtered_df.groupby('City')['Total Spend'].sum().idxmax()
    
    st.write(f"💰 **Total Revenue**: ${total_revenue:,.2f}")
    st.write(f"🛒 **Average Order Value**: ${avg_order:.2f}")
    st.write(f"🏙️ **Top Revenue City**: {top_city_revenue}")
    
    if discount_lift > 0:
        st.write(f"✅ **Discount Strategy**: Effective (+{discount_lift:.1f}% revenue lift)")
    else:
        st.write(f"⚠️ **Discount Strategy**: May need optimization ({discount_lift:.1f}% impact)")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.header("😊 Customer Satisfaction Analysis")
    
    # Satisfaction metrics
    satisfaction_dist = filtered_df['Satisfaction Level'].value_counts()
    satisfaction_rate = (filtered_df['Satisfaction Level'] == 'Satisfied').mean() * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Satisfaction Rate",
            value=f"{satisfaction_rate:.1f}%"
        )
    
    with col2:
        avg_rating = filtered_df['Average Rating'].mean()
        st.metric(
            label="Average Rating",
            value=f"{avg_rating:.2f}/5.0"
        )
    
    with col3:
        rating_satisfaction_corr = filtered_df.groupby('Satisfaction Level')['Average Rating'].mean()
        satisfied_rating = rating_satisfaction_corr.get('Satisfied', 0)
        st.metric(
            label="Satisfied Customers Avg Rating",
            value=f"{satisfied_rating:.2f}/5.0"
        )
    
    # Satisfaction analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction distribution
        fig_satisfaction = px.pie(
            values=satisfaction_dist.values,
            names=satisfaction_dist.index,
            title="Customer Satisfaction Distribution",
            color_discrete_sequence=['#ff6b6b', '#feca57', '#48dbfb']
        )
        st.plotly_chart(fig_satisfaction, use_container_width=True)
    
    with col2:
        # Satisfaction by membership type
        satisfaction_membership = pd.crosstab(
            filtered_df['Membership Type'], 
            filtered_df['Satisfaction Level'], 
            normalize='index'
        ) * 100
        
        fig_sat_membership = px.bar(
            satisfaction_membership,
            title="Satisfaction by Membership Type (%)",
            color_discrete_sequence=['#ff6b6b', '#feca57', '#48dbfb']
        )
        fig_sat_membership.update_xaxes(title="Membership Type")
        fig_sat_membership.update_yaxes(title="Percentage")
        st.plotly_chart(fig_sat_membership, use_container_width=True)
    
    # Rating analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Average rating by segment
        rating_by_segment = filtered_df.groupby('Customer_Segment')['Average Rating'].mean().sort_values(ascending=False)
        
        fig_rating_segment = px.bar(
            x=rating_by_segment.index,
            y=rating_by_segment.values,
            title="Average Rating by Customer Segment",
            color=rating_by_segment.values,
            color_continuous_scale='viridis'
        )
        fig_rating_segment.update_xaxes(title="Customer Segment")
        fig_rating_segment.update_yaxes(title="Average Rating")
        st.plotly_chart(fig_rating_segment, use_container_width=True)
    
    with col2:
        # Satisfaction vs Spending
        satisfaction_spending = filtered_df.groupby('Satisfaction Level')['Total Spend'].mean()
        
        fig_sat_spend = px.bar(
            x=satisfaction_spending.index,
            y=satisfaction_spending.values,
            title="Average Spending by Satisfaction Level",
            color=satisfaction_spending.values,
            color_continuous_scale='blues'
        )
        fig_sat_spend.update_xaxes(title="Satisfaction Level")
        fig_sat_spend.update_yaxes(title="Average Spend ($)")
        st.plotly_chart(fig_sat_spend, use_container_width=True)
    
    # Satisfaction drivers analysis
    st.subheader("🔍 Satisfaction Drivers")
    
    # Calculate correlation between satisfaction and other factors
    satisfaction_numeric = filtered_df['Satisfaction Level'].map({'Unsatisfied': 1, 'Neutral': 2, 'Satisfied': 3})
    correlations = pd.DataFrame({
        'Factor': ['Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase'],
        'Correlation': [
            satisfaction_numeric.corr(filtered_df['Total Spend']),
            satisfaction_numeric.corr(filtered_df['Items Purchased']),
            satisfaction_numeric.corr(filtered_df['Average Rating']),
            satisfaction_numeric.corr(filtered_df['Days Since Last Purchase'])
        ]
    }).sort_values('Correlation', ascending=False)
    
    fig_correlations = px.bar(
        correlations,
        x='Factor',
        y='Correlation',
        title="Factors Correlated with Customer Satisfaction",
        color='Correlation',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig_correlations, use_container_width=True)
    
    # Satisfaction insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("💡 Satisfaction Insights")
    
    top_satisfaction_segment = filtered_df.groupby('Customer_Segment')['Satisfaction Level'].apply(lambda x: (x == 'Satisfied').mean()).idxmax()
    worst_satisfaction_segment = filtered_df.groupby('Customer_Segment')['Satisfaction Level'].apply(lambda x: (x == 'Satisfied').mean()).idxmin()
    
    st.write(f"🏆 **Most Satisfied Segment**: {top_satisfaction_segment}")
    st.write(f"⚠️ **Least Satisfied Segment**: {worst_satisfaction_segment}")
    st.write(f"⭐ **Average Customer Rating**: {avg_rating:.2f}/5.0")
    
    strongest_driver = correlations.iloc[0]
    st.write(f"🎯 **Key Satisfaction Driver**: {strongest_driver['Factor']} (correlation: {strongest_driver['Correlation']:.3f})")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.header("🔮 Predictive Analytics")
    
    # Model performance
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Model Accuracy (R²)",
            value=f"{model_r2:.3f}",
            help="R-squared score indicating model performance"
        )
    
    with col2:
        st.metric(
            label="Prediction Error (MAE)",
            value=f"${model_mae:.2f}",
            help="Mean Absolute Error in dollars"
        )
    
    with col3:
        st.metric(
            label="Features Used",
            value=len(feature_names),
            help="Number of features in the prediction model"
        )
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance for Spending Prediction",
        color='Importance',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Prediction tool
    st.subheader("🎯 Customer Spending Predictor")
    st.write("Adjust the parameters below to predict customer spending:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_age = st.slider("Age", 18, 80, 35)
        pred_items = st.slider("Items Purchased", 1, 20, 5)
        pred_rating = st.slider("Average Rating", 1.0, 5.0, 4.0, 0.1)
    
    with col2:
        pred_days = st.slider("Days Since Last Purchase", 1, 365, 30)
        pred_gender = st.selectbox("Gender", df['Gender'].unique())
        pred_city = st.selectbox("City", df['City'].unique())
    
    with col3:
        pred_membership = st.selectbox("Membership Type", df['Membership Type'].unique())
        pred_satisfaction = st.selectbox("Satisfaction Level", df['Satisfaction Level'].unique())
        pred_discount = st.checkbox("Discount Applied")
    
    # Make prediction
    if st.button("🚀 Predict Spending", type="primary"):
        # Encode categorical variables
        le_gender, le_city, le_membership, le_satisfaction = encoders
        
        try:
            gender_encoded = le_gender.transform([pred_gender])[0]
            city_encoded = le_city.transform([pred_city])[0]
            membership_encoded = le_membership.transform([pred_membership])[0]
            satisfaction_encoded = le_satisfaction.transform([pred_satisfaction])[0]
            discount_numeric = int(pred_discount)
            
            # Create prediction array
            prediction_input = np.array([[
                pred_age, pred_items, pred_rating, pred_days,
                gender_encoded, city_encoded, membership_encoded,
                satisfaction_encoded, discount_numeric
            ]])
            
            # Make prediction
            predicted_spend = model.predict(prediction_input)[0]
            
            # Display result
            st.success(f"💰 Predicted Customer Spending: ${predicted_spend:.2f}")
            
            # Add context
            avg_spend = df['Total Spend'].mean()
            if predicted_spend > avg_spend:
                st.info(f"📈 This is {((predicted_spend/avg_spend - 1) * 100):.1f}% above average spending (${avg_spend:.2f})")
            else:
                st.info(f"📉 This is {((1 - predicted_spend/avg_spend) * 100):.1f}% below average spending (${avg_spend:.2f})")
            
        except ValueError as e:
            st.error("Error in prediction. Please check your inputs.")
    
    # Customer lifetime value prediction
    st.subheader("📊 Customer Lifetime Value Analysis")
    
    # CLV estimation based on current data
    clv_data = filtered_df.groupby('Customer_Segment').agg({
        'Total Spend': 'mean',
        'Days Since Last Purchase': 'mean',
        'Items Purchased': 'mean'
    }).round(2)
    
    # Simple CLV calculation (annualized)
    clv_data['Estimated_Annual_CLV'] = (
        clv_data['Total Spend'] * (365 / clv_data['Days Since Last Purchase'].clip(lower=1))
    ).round(2)
    
    fig_clv = px.bar(
        x=clv_data.index,
        y=clv_data['Estimated_Annual_CLV'],
        title="Estimated Annual Customer Lifetime Value by Segment",
        color=clv_data['Estimated_Annual_CLV'],
        color_continuous_scale='viridis'
    )
    fig_clv.update_xaxes(title="Customer Segment")
    fig_clv.update_yaxes(title="Estimated Annual CLV ($)")
    st.plotly_chart(fig_clv, use_container_width=True)
    
    # Predictive insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("💡 Predictive Insights")
    
    most_important_feature = feature_importance.iloc[0]['Feature']
    highest_clv_segment = clv_data['Estimated_Annual_CLV'].idxmax()
    
    st.write(f"🎯 **Most Important Predictor**: {most_important_feature}")
    st.write(f"💎 **Highest CLV Segment**: {highest_clv_segment} (${clv_data.loc[highest_clv_segment, 'Estimated_Annual_CLV']:,.2f})")
    st.write(f"🎮 **Model Performance**: {model_r2:.1%} accuracy with ±${model_mae:.2f} error")
    
    st.write("**🚀 Recommendations for Revenue Growth:**")
    st.write("• Focus marketing spend on features that drive the highest predictions")
    st.write("• Develop retention strategies for high CLV segments")
    st.write("• Use predictive scores for personalized offers")
    st.write("• Monitor model performance and retrain with new data")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666;'>
        📊 E-commerce Customer Analytics Dashboard | Built with Streamlit | 
        Data Science & Analytics Project
    </div>
    """, 
    unsafe_allow_html=True
)

# Download processed data
if st.sidebar.button("📥 Download Processed Data"):
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="filtered_customer_data.csv",
        mime="text/csv"
    )

