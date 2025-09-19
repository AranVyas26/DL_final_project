from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
import joblib
import plots as plot_utils   # renamed to avoid conflict
import pandas as pd
import io
import os
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# ---------------- Model & Scaler ----------------
model = tf.keras.models.load_model("C:/Users/vyasa/Downloads/dl2_transformer.keras")
scaler = joblib.load("scaler.pkl")

failure_labels = {
    0: "No Failure",
    1: "TWF (Tool Wear Failure)",
    2: "HDF (Heat Dissipation Failure)",
    3: "PWF (Power Failure)",
    4: "OSF (Overstrain Failure)",
    5: "RNF (Random Failure)"
}

# ---------------- Detailed Failure Analysis ----------------
failure_analysis = {
    0: {
        "explanation": "No Failure detected. Machine is operating normally.",
        "causes": ["All parameters are within safe operating limits."],
        "prevention": ["Continue regular monitoring and maintenance."]
    },
    1: {
        "explanation": "Tool Wear Failure – degradation due to extended use.",
        "causes": [
            "High torque causing excessive wear",
            "Prolonged operation without replacement",
            "Inadequate lubrication"
        ],
        "prevention": [
            "Implement tool replacement schedules",
            "Ensure proper lubrication",
            "Avoid prolonged overloads"
        ]
    },
    2: {
        "explanation": "Heat Dissipation Failure – system unable to release heat effectively.",
        "causes": [
            "High air and process temperature",
            "Cooling system malfunction",
            "Blocked airflow or ventilation"
        ],
        "prevention": [
            "Improve cooling mechanisms",
            "Clean ventilation paths",
            "Monitor temperature sensors"
        ]
    },
    3: {
        "explanation": "Power Failure – abnormal power supply or motor issue.",
        "causes": [
            "Electrical fluctuations",
            "Motor overload",
            "Faulty power lines"
        ],
        "prevention": [
            "Use surge protectors",
            "Check motor load regularly",
            "Schedule power system inspections"
        ]
    },
    4: {
        "explanation": "Overstrain Failure – machine components overstressed.",
        "causes": [
            "Excessive torque",
            "Operating beyond rated load",
            "Weak structural parts"
        ],
        "prevention": [
            "Limit torque within safe range",
            "Strengthen machine components",
            "Monitor load conditions"
        ]
    },
    5: {
        "explanation": "Random Failure – unpredictable breakdown.",
        "causes": [
            "Sudden material defects",
            "Unforeseen operating condition",
            "Component fatigue"
        ],
        "prevention": [
            "Regular inspections",
            "Early fault detection systems",
            "Predictive maintenance"
        ]
    }
}

# ---------------- Generate Plots Once ----------------
plot_utils.generate_all_plots()

# ---------------- Ensure temp/report folders ----------------
os.makedirs("static/temp", exist_ok=True)
os.makedirs("static/reports", exist_ok=True)

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/plots")
def plots():
    return render_template("plots.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ---------------- AI Report ----------------
@app.route("/ai_report", methods=["GET", "POST"])
def ai_report():
    preview = None
    report_ready = False
    error = None
    report_path = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part"
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "No file selected"
            else:
                try:
                    df = pd.read_csv(file)
                    if "Predicted Failure" not in df.columns:
                        error = "Uploaded CSV must contain 'Predicted Failure' column."
                        return render_template("ai_report.html", error=error)
                    
                    # Save uploaded CSV temporarily
                    temp_csv = "static/temp/ai_report_input.csv"
                    df.to_csv(temp_csv, index=False)
                    
                    # Generate summary and chart
                    summary = df['Predicted Failure'].value_counts().to_dict()
                    plt.figure(figsize=(5,4))
                    sns.countplot(y='Predicted Failure', data=df, palette="Set2")
                    plt.title("Failure Type Counts")
                    plt.tight_layout()
                    chart_path = "static/reports/failure_chart.png"
                    plt.savefig(chart_path)
                    plt.close()

                    # Generate PDF report
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, "Machine Failure Prediction Report", 0, 1, 'C')
                    pdf.set_font("Arial", '', 10)
                    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
                    pdf.cell(0, 8, f"Total Records: {len(df)}", 0, 1)
                    pdf.cell(0, 8, "Model: dl2_transformer.keras", 0, 1)
                    pdf.ln(5)

                    # Summary Section
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, "Summary of Predictions", 0, 1)
                    pdf.set_font("Arial", '', 11)
                    for failure, count in summary.items():
                        pdf.cell(0, 6, f"{failure}: {count} machines", 0, 1)
                    pdf.ln(5)

                    # Add Chart
                    pdf.image(chart_path, x=35, w=140)
                    pdf.ln(10)

                    # Detailed Record Section
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, "Detailed Record Predictions", 0, 1)
                    pdf.set_font("Arial", '', 10)
                    for idx, row in df.iterrows():
                        pdf.multi_cell(0, 6, f"Record {idx+1}: Predicted Failure = {row['Predicted Failure']}, "
                                              f"Type={row['Type']}, Air Temp={row['Air temperature [K]']}, "
                                              f"Process Temp={row['Process temperature [K]']}, "
                                              f"Rotational Speed={row['Rotational speed [rpm]']}, "
                                              f"Torque={row['Torque [Nm]']}")
                        pdf.ln(1)

                    report_path = f"static/reports/machine_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    pdf.output(report_path)
                    report_ready = True

                    # Preview first 10 rows
                    preview = df.head(10).to_html(classes="table table-bordered table-striped", index=False)

                except Exception as e:
                    error = f"Error: {str(e)}"

    return render_template("ai_report.html", preview=preview, report_ready=report_ready, error=error)

# ---------------- Download AI Report ----------------
@app.route("/download_report")
def download_report():
    reports = sorted([f for f in os.listdir("static/reports") if f.endswith(".pdf")], reverse=True)
    if reports:
        return send_file(os.path.join("static/reports", reports[0]), as_attachment=True)
    else:
        return "No report available. Please generate the report first."

# ---------------- Bulk CSV ----------------
@app.route("/bulk", methods=["GET", "POST"])
def bulk():
    preview = None
    error = None
    temp_path = "static/temp/bulk_predictions.csv"

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part"
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "No file selected"
            else:
                try:
                    df = pd.read_csv(file)
                    required_cols = ["Type", "Air temperature [K]", "Process temperature [K]",
                                     "Rotational speed [rpm]", "Torque [Nm]"]
                    for col in required_cols:
                        if col not in df.columns:
                            error = f"Missing column: {col}"
                            return render_template("bulk.html", error=error)
                    
                    features = df[required_cols].values
                    scaled_features = scaler.transform(features)
                    predictions = model.predict(scaled_features)
                    predicted_classes = [failure_labels[int(np.argmax(p))] for p in predictions]
                    df["Predicted Failure"] = predicted_classes
                    df.to_csv(temp_path, index=False)
                    preview = df.head(10).to_html(classes="table table-bordered table-striped", index=False)
                except Exception as e:
                    error = f"Error: {str(e)}"

    return render_template("bulk.html", preview=preview, error=error)

@app.route("/download_bulk_csv")
def download_bulk_csv():
    temp_path = "static/temp/bulk_predictions.csv"
    if os.path.exists(temp_path):
        return send_file(temp_path, as_attachment=True)
    else:
        return "No file available. Please upload and predict first."

# ---------------- Other Routes ----------------
@app.route("/logging")
def logging_page():
    return render_template("logging.html")

@app.route("/user_pred")
def user_pred():
    return render_template("user_pred.html")

@app.route("/model_development")
def model_development():
    return render_template("model_development.html")

# ---------------- Single Prediction ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            type_val = int(request.form["type"])
            air_temp = float(request.form["air_temp"])
            process_temp = float(request.form["process_temp"])
            rot_speed = float(request.form["rot_speed"])
            torque = float(request.form["torque"])

            features = np.array([[type_val, air_temp, process_temp, rot_speed, torque]])
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)

            predicted_class = int(np.argmax(prediction, axis=1)[0])
            predicted_label = failure_labels[predicted_class]
            detailed = failure_analysis.get(predicted_class, None)

            return render_template(
                "user_pred.html",
                prediction=predicted_label,
                detailed=detailed
            )
        except Exception as e:
            return render_template("user_pred.html", prediction=f"Error: {str(e)}", detailed=None)
# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(debug=True)
