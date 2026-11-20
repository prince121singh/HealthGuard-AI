import html
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT_DIR / "reports"


def generate_html_report(
    patient_data,
    prediction_result,
    explanation_df=None,
    output_path=None
):
    """
    Generate a standalone HTML report.

    Returns:
        tuple: (report_file_path, html_content)
    """

    patient_data = patient_data or {}
    prediction_result = prediction_result or {}

    label = prediction_result.get(
        "label",
        "Prediction unavailable"
    )

    prediction = prediction_result.get("prediction", "N/A")
    probability = prediction_result.get("probability")

    if probability is not None:
        probability_text = f"{float(probability) * 100:.2f}%"
    else:
        probability_text = "Not available"

    def safe(value):
        return html.escape(str(value))

    # Patient input table
    input_rows = ""

    for feature, value in patient_data.items():
        input_rows += f"""
        <tr>
            <td>{safe(feature)}</td>
            <td>{safe(value)}</td>
        </tr>
        """

    if not input_rows:
        input_rows = """
        <tr>
            <td colspan="2">No input information available</td>
        </tr>
        """

    # Explainability table
    shap_rows = ""

    if isinstance(explanation_df, pd.DataFrame):
        for _, row in explanation_df.head(10).iterrows():

            feature = safe(row.get("feature", "Unknown"))
            value = safe(row.get("value", "N/A"))

            shap_value = float(row.get("shap_value", 0))

            shap_rows += f"""
            <tr>
                <td>{feature}</td>
                <td>{value}</td>
                <td>{shap_value:+.4f}</td>
            </tr>
            """

    if not shap_rows:
        shap_rows = """
        <tr>
            <td colspan="3">No explanation available</td>
        </tr>
        """

    generated_at = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )

    report_html = f"""
    <!DOCTYPE html>
    <html lang="en">

    <head>
        <meta charset="UTF-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

        <title>HealthGuard AI Report</title>

        <style>
            body {{
                margin: 0;
                padding: 35px 15px;
                background: #f1f5f9;
                color: #172033;
                font-family: Arial, sans-serif;
            }}

            .container {{
                max-width: 850px;
                margin: auto;
                background: white;
                padding: 35px;
                border-radius: 16px;
                box-shadow: 0 8px 30px rgba(0,0,0,.08);
            }}

            h1 {{
                color: #0891b2;
                margin-bottom: 5px;
            }}

            h2 {{
                margin-top: 30px;
                color: #164e63;
            }}

            .meta {{
                color: #64748b;
                font-size: 14px;
            }}

            .result {{
                background: #ecfeff;
                border-left: 5px solid #0891b2;
                padding: 18px;
                border-radius: 8px;
                margin-top: 25px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 12px;
            }}

            th, td {{
                padding: 12px;
                text-align: left;
                border: 1px solid #e2e8f0;
            }}

            th {{
                background: #f1f5f9;
            }}

            .warning {{
                margin-top: 35px;
                padding: 18px;
                background: #fffbeb;
                border-left: 5px solid #f59e0b;
                border-radius: 8px;
                line-height: 1.6;
            }}

            footer {{
                margin-top: 30px;
                text-align: center;
                color: #64748b;
                font-size: 12px;
            }}

            @media print {{
                body {{ background: white; }}
                .container {{ box-shadow: none; }}
            }}
        </style>
    </head>

    <body>

        <div class="container">

            <h1>HealthGuard AI</h1>

            <p class="meta">
                Machine Learning & Explainable AI Report
                <br>
                Generated: {safe(generated_at)}
            </p>

            <div class="result">

                <h2>Prediction Summary</h2>

                <p>
                    <strong>Model output:</strong>
                    {safe(prediction)}
                </p>

                <p>
                    <strong>Output label:</strong>
                    {safe(label)}
                </p>

                <p>
                    <strong>Model probability:</strong>
                    {safe(probability_text)}
                </p>

            </div>

            <h2>Input Information</h2>

            <table>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                </tr>

                {input_rows}
            </table>

            <h2>Explainability (SHAP)</h2>

            <table>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>SHAP Value</th>
                </tr>

                {shap_rows}
            </table>

            <div class="warning">

                <strong>Important Disclaimer</strong>

                <p>
                    HealthGuard AI is an educational machine
                    learning demonstration. It is not a medical
                    diagnostic system and does not provide
                    medical advice.
                </p>

                <p>
                    Model predictions and SHAP explanations
                    should not be interpreted as clinical
                    conclusions.
                </p>

            </div>

            <footer>
                HealthGuard AI | Educational Use Only
            </footer>

        </div>

    </body>
    </html>
    """

    if output_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = (
            REPORT_DIR / "healthguard_report.html"
        )
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    output_path.write_text(
        report_html,
        encoding="utf-8"
    )

    return str(output_path), report_html