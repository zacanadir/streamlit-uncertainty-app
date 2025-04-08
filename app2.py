import streamlit as st
import joblib
import pandas as pd
from preprocess import load_data
from sklearn.metrics import classification_report,accuracy_score
from model import get_train_test_data
from preprocess import get_preprocessor
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
from model2 import load_mc_dropout_model,predict_with_mc_dropout
import tensorflow as ts
from tensorflow.keras.activations import sigmoid

st.set_page_config(
    page_title="Uncertainty Explorer",
    page_icon="📊",
    layout="wide"
)


st.title("📱 Mobile Price Prediction App")
# 1. Introduction Section
st.header("Welcome to the Mobile Price Prediction App")
st.write(f"""This app predicts the price range of mobile devices based on their specifications. Initially, it uses machine learning to classify devices into one of four price categories:

    0 → 🪙 Lowest Price Level\n
    1 → 💵 Low-Mid Price Level\n
    2 → 💳 Mid-High Price Level\n
    3 → 💼 Highest Price Level\n

The goal of this app evolves from achieving static predictions to addressing the uncertainty inherent in the data. Using a sequential neural network model with the addition of Monte Carlo (MC) Dropout, we simulate epistemic uncertainty—the uncertainty due to our lack of knowledge about the model and data. This approach helps us understand the variability in predictions arising from data noise, model limitations, and the potential for overfitting.""")

# Load data
data = load_data()
X = data.drop(columns='price_range')
y = data['price_range']

# 2. Data Overview Section
st.header("📊 Data Overview")

# Define feature types
binary_features = ['Bluetooth', 'Dual Sim', '4G', '3G', 'Touch Screen', 'Wifi']
numeric_features = [col for col in X.columns if col not in binary_features]

# Donut chart for feature types
feature_type_counts = pd.Series({
    "Binary Features": len(binary_features),
    "Numeric Features": len(numeric_features)
})

fig = go.Figure(data=[go.Pie(
    labels=feature_type_counts.index,
    values=feature_type_counts.values,
    hole=0.4,
    marker=dict(colors=["#00cc96", "#636efa"]),
    textinfo="label+percent"
)])
fig.update_layout(title="Feature Types Distribution")
st.plotly_chart(fig, use_container_width=True)

# Display lists of features
with st.expander("🔍 View Binary & Numeric Features"):
    st.markdown(f"**🧮 Numeric Features ({len(numeric_features)}):**")
    st.write(", ".join(numeric_features))
    st.markdown(f"**🔘 Binary Features ({len(binary_features)}):**")
    st.write(", ".join(binary_features))


st.subheader("📐 Feature Summary Statistics")
summary_stats = X[numeric_features].describe().T[["min", "50%", "mean", "max"]]
summary_stats.rename(columns={"50%": "median"}, inplace=True)
st.dataframe(summary_stats.style.format(int), use_container_width=True)


st.subheader("📎 Feature Correlation Heatmap")
corr = X.corr()

# Get top 3 most correlated pairs (excluding 1.0 diagonal)
corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
top_corrs = corr_pairs.unstack().dropna().sort_values(ascending=False).head(3)

st.markdown("**🔥 Top 3 Most Correlated Feature Pairs:**")
for (f1, f2), val in top_corrs.items():
    st.write(f"- `{f1}` & `{f2}` → Correlation: **{val:.2f}**")

# Plot heatmap
fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_corr, use_container_width=True)


X_train, X_test, y_train, y_test = get_train_test_data(X, y)
# Load the pre-trained model
model = joblib.load("models/model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# 3. Model Training & Evaluation
st.header("💡 Model Training & Evaluation")
st.write("We use a Random Forest model trained on the dataset. Below are the performance metrics:")

# Displaying Accuracy, Precision, Recall, F1-Score
st.subheader("📊 Model Performance")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy**: {accuracy:.4f}")

# Classification Report with more readable format
classification_rep = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(classification_rep).transpose()
st.write("**Classification Report**")
left_col, center_col, right_col = st.columns([1, 3, 1])  # Wider center column
with center_col:
    st.write(df_report.style.format("{:.4f}").background_gradient(cmap="Blues"))

# Add Confusion Matrix with Plotly heatmap
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=[f"Class {i}" for i in range(4)],
    y=[f"Class {i}" for i in range(4)],
    colorscale='Viridis'
)
fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
st.plotly_chart(fig_cm)

from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

# Get the predicted probabilities for all classes
y_pred_proba = model.predict_proba(X_test)
# Initialize a figure for the ROC curves
fig_roc = go.Figure()
# Loop through each class and plot the ROC curve
for i in range(4):  # Since you have 4 classes (0, 1, 2, 3)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, i],pos_label=i)
    roc_auc = auc(fpr, tpr)

    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'Class {i} ROC curve (area = {roc_auc:.2f})'
        )
    )

# Update layout for the figure
fig_roc.update_layout(
    title="Receiver Operating Characteristic (ROC) Curve for All Classes",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    showlegend=True
)

# Display the plot
st.plotly_chart(fig_roc)

# 4. Model Robustness to Class Imbalance
st.header("🧭 Model Robustness to Class Imbalance")

#st.header("🎯 Precision Matters: Model vs. Imbalance")
st.write("Many real-world datasets are imbalanced. This section explores how well the model performs across all classes using precision-recall analysis.")

# Plot class distribution
st.subheader("🔢 Class Distribution")
class_counts = y.value_counts().sort_index()
st.bar_chart(class_counts)
st.write("This shows how frequent each price range is in the dataset. The model needs to be able to perform well across all of them.")

# Precision-Recall Curves
st.subheader("📉 Precision-Recall Curves")
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

# Binarize labels for multi-class precision-recall
n_classes = 4
y_proba = model.predict_proba(X_test)
y_test_binarized = label_binarize(y_test, classes=[0,1,2,3])
# Create interactive precision-recall plot
fig = go.Figure()
colors = ['red', 'orange', 'blue', 'green']
labels = ['Lowest', 'Low-Mid', 'Mid-High', 'Highest']

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_proba[:, i])
    ap = average_precision_score(y_test_binarized[:, i], y_proba[:, i])
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                             name=f"{labels[i]} (AP={ap:.2f})", line=dict(color=colors[i])))

fig.update_layout(title="Precision-Recall Curves for Each Price Level",
                  xaxis_title="Recall",
                  yaxis_title="Precision",
                  height=400,
                  width=700)
st.plotly_chart(fig)

st.write("These curves show how well the model identifies each price level, especially under imbalanced conditions. A higher area under the curve (average precision) indicates better performance.")

# 5. Model Interpretability
st.header("🔍 Model Interpretability")
st.write("SHAP values show how each feature contributes to the predictions made by the model.")

# Extract feature names after preprocessing and remove prefixes
clf_step = model.named_steps['clf']
preprocessor = get_preprocessor()
preprocessor.fit(X_train)
X_test_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
feature_names = [name.split('__')[-1] for name in feature_names]  # Strip 'scaler__' or 'remainder__'
feature_names_sorted = sorted(feature_names)

# Initialize the SHAP explainer
explainer = shap.TreeExplainer(clf_step)
sh_val_obj = explainer(X_test_transformed)
agg_shap_values = np.abs(sh_val_obj.values).mean(axis=2)  # Shape: (features, classes)
# Plot the aggregated SHAP feature importance across all classes

# Mean absolute SHAP value per feature
feature_importance = agg_shap_values.mean(axis=0)

# Sort features by importance
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importance = feature_importance[sorted_idx]

fig_shap = go.Figure(
    data=go.Bar(
        x=sorted_importance,
        y=sorted_features,
        orientation='h',
        marker=dict(color=sorted_importance, colorscale='Blues'),
    )
)

fig_shap.update_layout(
    title="🔑 Aggregated SHAP Feature Importance (Across All Classes)",
    xaxis_title="Mean |SHAP value|",
    yaxis_title="Feature",
    yaxis=dict(autorange='reversed'),  # Most important at the top
    height=600
)

st.plotly_chart(fig_shap)

# 6. User Prediction
st.header("📊 Try It Yourself: Predict Mobile Price Range")

user_input = {}

# First, create columns for numeric features
st.subheader("🔢 Numeric Inputs")
col1, col2 = st.columns(2)

# Add numeric inputs to col1
with col1:
    for col in numeric_features[:len(numeric_features)//2]:  # First half of numeric features
        val = st.number_input(f"{col}:", value=int(X[col].mean()))
        user_input[col] = val

# Add numeric inputs to col2
with col2:
    for col in numeric_features[len(numeric_features)//2:]:  # Second half of numeric features
        val = st.number_input(f"{col}:", value=int(X[col].mean()))
        user_input[col] = val

# Now, create columns for binary features
st.subheader("✅ Binary Features")
col1, col2 = st.columns(2)
# Add binary inputs to col1
with col1:
    for col in binary_features[:len(binary_features)//2]:  # First half of binary features
        val = st.checkbox(f"{col}", value=True)
        user_input[col] = 1 if val else 0

# Add binary inputs to col2
with col2:
    for col in binary_features[len(binary_features)//2:]:  # Second half of binary features
        val = st.checkbox(f"{col}", value=True)
        user_input[col] = 1 if val else 0

# Convert to DataFrame
user_num_df = pd.DataFrame([user_input], columns=numeric_features)
user_bin_df = pd.DataFrame([user_input], columns=binary_features)
frame = [user_num_df, user_bin_df]
user_df = pd.concat(frame, axis=1)
user_df = user_df[feature_names_sorted]  # Reorder according to the model
# Predict
price_labels = {
    0: "Lowest Price Level",
    1: "Low-Mid Price Level",
    2: "Mid-High Price Level",
    3: "Highest Price Level"
}


# Predict on button click
import numpy as np
from scipy.stats import entropy

import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.subheader("🔮 Get Predictions")
st.write("Metrics generated in the app, such as confidence and entropy (which measures uncertainty), offer insights into the model's certainty about its predictions. Confidence represents the predicted probability of the most likely price range, while entropy reflects the spread or uncertainty across all predicted classes. Lower confidence indicates a more uncertain prediction, and higher entropy suggests greater uncertainty about the model's choice.")


# --- Session State Initialization for Button Locking ---
for key in ["model1_clicked", "model2_clicked"]:
    if key not in st.session_state:
        st.session_state[key] = False



# --- Button Layout: Divide page into two sections for Buttons ---
col1, col2 = st.columns([1, 1])

# TODO: Button 1 - User Prediction - Static Case
with col1:
    if st.button("Predict Price Range - Static Tree Model"):
        st.session_state.model1_clicked = True
        st.session_state.model2_clicked = False
        st.empty()  # Remove any existing plots before new ones appear

        # Prediction logic
        prediction_proba = model.predict_proba(user_df)[0]
        prediction = np.argmax(prediction_proba)
        confidence = prediction_proba[prediction]
        uncertainty = entropy(prediction_proba)

        if confidence < 0.6:
            st.warning("⚠️ The model is not very confident in this prediction. Consider reviewing the input.")

        st.session_state.prediction = prediction
        st.session_state.prediction_proba = prediction_proba
        st.session_state.confidence = confidence
        st.session_state.uncertainty = uncertainty

# TODO: Button 2 - User Prediction - MC Dropout Model Button
with col2:
    if st.button("Predict Price Range - MC Dropout Model"):
        st.session_state.model2_clicked = True
        st.session_state.model1_clicked = False
        st.empty()  # Remove any existing plots before new ones appear

        # Load MC Dropout model
        model2 = load_mc_dropout_model('models/model2.h5')

        # Prepare input data
        #user_input_df = pd.DataFrame([user_input], columns=numeric_features + binary_features)
        #user_input_df = user_input_df[feature_names_sorted]

        # Prediction with MC Dropout
        mean_predictions, uncertainty = predict_with_mc_dropout(model2, user_df, n_iterations=50)

        prediction_mc = np.argmax(mean_predictions[0])
        confidence_mc = mean_predictions[0][prediction_mc]
        uncertainty_mc = entropy(mean_predictions[0])

        if confidence_mc < 0.6:
            st.warning("⚠️ The model is not very confident in this prediction. Consider reviewing the input.")

        st.session_state.prediction_mc = prediction_mc
        st.session_state.prediction_mc_proba = mean_predictions[0]
        st.session_state.confidence_mc = confidence_mc
        st.session_state.uncertainty_mc = uncertainty_mc
        st.session_state.model2=model2



# --- Model Active Status Badge ---
if st.session_state.model1_clicked:
    with col1:
        st.markdown(" **Active Model:** 🌳 Static Tree Model")
elif st.session_state.model2_clicked:
    with col2:
        st.markdown(" **Active Model:** 🤖 MC Dropout Model")


# Placeholder at the top where new content can appear
top_placeholder = st.empty()


# --- Feature Impact Visualization ---
st.subheader("🔍 Understand What Impacts the Price Level You're Seeing")

col3, col4 = st.columns([1, 1])

# TODO: Button 3 - Feature Impact Visualization - Static Case
with col3:
    if st.button("See Feature Impact for Static Prediction", disabled=st.session_state.model2_clicked):
        st.session_state.button_clicked = "button3"
        st.empty()  # Remove any existing plots before new ones appear

        if 'prediction' not in st.session_state:
            st.warning("Please click 'Predict Price Range - Static Tree Model' first to generate the prediction.")
        else:
            prediction = st.session_state.prediction
            sh_val_obj = explainer(user_df)
            values = sh_val_obj[0, :, prediction].values
            features = sh_val_obj.feature_names
            base_value = sh_val_obj.base_values[0][prediction]


            final_prediction = base_value + values.sum()

            #final_prediction = sigmoid(base_value + values.sum())  # for binary classification

            sorted_idx = np.argsort(np.abs(values))[::-1]
            values_sorted = values[sorted_idx]
            features_sorted = np.array(features)[sorted_idx]

            # Save to session state for plotting
            st.session_state.features_sorted = features_sorted
            st.session_state.values_sorted = values_sorted
            st.session_state.base_value = base_value
            st.session_state.final_prediction = final_prediction

# TODO: Button 4 - Feature Impact Visualization - MC Dropout Model
with col4:
    if st.button("See Feature Impact for MC Dropout Prediction", disabled=st.session_state.model1_clicked):
        st.session_state.button_clicked = "button4"
        st.empty()  # Remove any existing plots before new ones appear

        if 'prediction_mc' not in st.session_state:
            st.warning("Please click 'Predict Price Range - MC Dropout Model' first to generate the prediction.")
        else:
            prediction_class = st.session_state.prediction_mc
            mean_predictions = st.session_state.prediction_mc_proba

            # Use a small background dataset for SHAP
            X_background = X_train.sample(100, random_state=42)

            # Create the explainer
            explainer_mc = shap.GradientExplainer(st.session_state.model2, X_background)

            # Get SHAP values: this might return shape (1, n_features, n_classes)
            raw_shap = explainer_mc.shap_values(user_df.values)

            # If raw_shap is a single array instead of a list, split it manually
            if isinstance(raw_shap, np.ndarray):
                # Shape: (1, features, classes)
                shap_values_list = [raw_shap[0, :, i] for i in range(raw_shap.shape[2])]
            else:
                shap_values_list = raw_shap

            # Values from SHAP
            values = shap_values_list[prediction_class]

            # Feature names
            feature_names = user_df.columns.to_numpy()

            # Get predicted probability for class
            pred_probs = st.session_state.prediction_mc_proba  # from MC dropout
            final_prediction = pred_probs[prediction_class]



            # Estimate base value
            base_value = final_prediction - np.sum(values)

            # Sort for plotting
            sorted_idx = np.argsort(np.abs(values))[::-1]
            values_sorted = values[sorted_idx]
            features_sorted = feature_names[sorted_idx]

            # Store
            st.session_state.features_sorted_mc = features_sorted
            st.session_state.values_sorted_mc = values_sorted
            st.session_state.base_value_mc = base_value
            st.session_state.final_prediction_mc = final_prediction

with top_placeholder.container():
    with st.expander("📊 View Prediction Results", expanded=True):
        if st.session_state.model1_clicked == True and 'prediction' in st.session_state:

            st.success(f"📱 Predicted Price Range: {price_labels[st.session_state.prediction]}")
            st.markdown(f"**Confidence:** {st.session_state.confidence:.2f}")
            st.markdown(f"**Prediction Entropy (Uncertainty):** {st.session_state.uncertainty:.2f}")

            sorted_indices = np.argsort(st.session_state.prediction_proba)[::-1]
            sorted_probs = st.session_state.prediction_proba[sorted_indices]
            sorted_labels = [price_labels[i] for i in sorted_indices]
            colors = ["skyblue" if i != st.session_state.prediction else "crimson" for i in sorted_indices]

            fig_proba = go.Figure(
                data=[go.Bar(x=sorted_labels, y=sorted_probs, marker_color=colors)]
            )
            fig_proba.update_layout(
                yaxis_title="Probability",
                xaxis_title="Price Range Levels",
                title="🔢 Model Confidence Across All Price Levels - Static Prediction"
            )

            st.plotly_chart(fig_proba, use_container_width=True, key="fig_proba")

        elif st.session_state.model2_clicked == True and 'prediction_mc' in st.session_state:
            # Plot for MC Dropout Prediction
            # st.subheader("🔢 Class Probability Distribution (MC Dropout)")
            st.success(f"📱 Predicted Price Range: {price_labels[st.session_state.prediction_mc]}")
            st.markdown(f"**Confidence (Mean Probability):** {st.session_state.confidence_mc:.2f}")
            st.markdown(f"**Prediction Entropy (Uncertainty):** {st.session_state.uncertainty_mc:.2f}")

            sorted_indices = np.argsort(st.session_state.prediction_mc_proba)[::-1]
            sorted_probs = st.session_state.prediction_mc_proba[sorted_indices]
            sorted_labels = [price_labels[i] for i in sorted_indices]
            colors = ["skyblue" if i != st.session_state.prediction_mc else "crimson" for i in sorted_indices]

            fig_proba2 = go.Figure(
                data=[go.Bar(x=sorted_labels, y=sorted_probs, marker_color=colors)]
            )
            fig_proba2.update_layout(
                yaxis_title="Probability",
                xaxis_title="Price Range Levels",
                title="🔢 Model Confidence Across All Price Levels - MC Dropout Prediction"
            )

            st.plotly_chart(fig_proba2, use_container_width=True, key="fig_proba2")

    with st.expander("🧠 View SHAP Feature Impact", expanded=True):
        if st.session_state.model1_clicked and 'features_sorted' in st.session_state:


            features_sorted = st.session_state.features_sorted
            values_sorted = st.session_state.values_sorted
            base_value = st.session_state.base_value
            final_prediction = st.session_state.final_prediction

            fig = go.Figure(go.Waterfall(
                name="SHAP",
                orientation="v",
                x=["Base Value"] + features_sorted.tolist() + ["Final Prediction"],
                y=[base_value] + values_sorted.tolist() + [final_prediction - base_value - values_sorted.sum()],
                textposition="outside",
                connector={"line": {"color": "rgba(0,0,0,0.2)"}},
                increasing={"marker": {"color": "green"}},
                decreasing={"marker": {"color": "red"}},
                totals={"marker": {"color": "blue"}}
            ))

            fig.update_layout(
                title=f"↕️ Feature Impact for {price_labels[st.session_state.prediction]} - Static Prediction",
                yaxis_title="Model Output (Log-Odds or Score)",
                xaxis_title="Feature Contributions",
                showlegend=False,
                height=600,
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True, key="fig_shap")

            st.warning("The SHAP values show how each feature contributes to the model's prediction. However, due to the nature of tree-based models, the SHAP values might not match the exact final prediction. This is because SHAP values distribute feature importance based on game theory, which is a bit different from how the model calculates the prediction internally. So, while the SHAP values give a clear view of feature impacts, they won't always sum up to the final prediction itself.")

        elif st.session_state.model2_clicked and 'features_sorted_mc' in st.session_state:
            features_sorted = st.session_state.features_sorted_mc
            values_sorted = st.session_state.values_sorted_mc
            base_value = st.session_state.base_value_mc
            final_prediction = st.session_state.final_prediction_mc

            fig2 = go.Figure(go.Waterfall(
                name="SHAP",
                orientation="v",
                x=["Base Value"] + features_sorted.tolist() + ["Final Prediction"],
                y=[base_value] + values_sorted.tolist() + [final_prediction - base_value - values_sorted.sum()],
                textposition="outside",
                connector={"line": {"color": "rgba(0,0,0,0.2)"}},
                increasing={"marker": {"color": "green"}},
                decreasing={"marker": {"color": "red"}},
                totals={"marker": {"color": "blue"}}
            ))

            fig2.update_layout(
                title=f"↕️ Feature Impact for {price_labels[st.session_state.prediction_mc]} - MC Dropout Prediction",
                yaxis_title="Model Output (Log-Odds or Score)",
                xaxis_title="Feature Contributions",
                showlegend=False,
                height=600,
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig2, use_container_width=True, key="fig2_shap")
            st.warning("This SHAP chart explains one snapshot of the model with dropout active. Since predictions are averaged over many passes, exact probabilities may differ slightly.")

