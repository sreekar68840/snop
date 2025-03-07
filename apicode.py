import os
import re
import json
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from flask import Flask, request, jsonify
import threading
from flask_cors import CORS
from io import BytesIO
import base64
from PIL import Image
import plotly.io as pio
import time
 
# Load Environment Variables
load_dotenv()
 
# Snowflake Credentials
SNOWFLAKE_USER = "abhishek"
SNOWFLAKE_PASSWORD = "Ashok@8827396075"
SNOWFLAKE_ACCOUNT = "STB09500.us-west-2"
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
SNOWFLAKE_DATABASE = "SNOP_DB"
SNOWFLAKE_SCHEMA = "SNOP_SCHEMA"
 
def create_connection():
    try:
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        return conn
    except Exception as e:
        print(f"❌ Snowflake Connection Failed: {e}")
        raise
 
def get_snowflake_metadata(conn):
    metadata_query = """
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS;
    """
    try:
        cursor = conn.cursor()
        cursor.execute(metadata_query)
        metadata_rows = cursor.fetchall()
        cursor.close()
 
        if not metadata_rows:
            raise ValueError("⚠️ No metadata retrieved! Check database permissions or schema name.")
 
        metadata_df = pd.DataFrame(metadata_rows, columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])
        metadata_dict = (
            metadata_df.drop(columns=["TABLE_NAME"])
            .groupby(metadata_df["TABLE_NAME"], group_keys=False)
            .apply(lambda x: {col: dtype for col, dtype in zip(x["COLUMN_NAME"], x["DATA_TYPE"])}).
            to_dict()
        )
        return metadata_dict
    except Exception as e:
        print(f"❌ Error fetching metadata: {str(e)}")
        return None
 
def query_snowflake(conn, sql_query):
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        return pd.DataFrame(result, columns=columns)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})
 
def visual_generate(llm, query, data, response):
    try:
        prompt = f"""
        Give me Python code that can generate an insightful graph or plot on the given dataframe for the query based on response using Plotly.
        Dataframe: {data}.
        Query: {query}
        Response: {response}
 
        Follow below color schema for the plot:
        - Background: black (#2B2C2E)
        - Text and labels: white (#FFFFFF)
        - Bars/Lines: either #4BC884 or #22A6BD
        Also, add data labels.
 
        DO NOT include any additional text other than the Python code in the response.
        Save the plot at the end using: fig.write_image('graph.png', engine='kaleido')
 
        If it's not possible to generate a graph, return 'No graph generated' as a response.
        """
       
        code = llm.invoke(prompt).content
        if "No graph generated" in code:
            encoded_image = ''
        else:
            code = re.sub(r'```python', '', code)
            code = re.sub(r'`', '', code)
            exec(code)
            # plt.savefig('graph.png', dpi=300, bbox_inches='tight', facecolor='black')
            with open('graph.png', 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print("⚠️ Graph generation error:", e)
        encoded_image = ''
    return encoded_image
 
os.environ["OPENAI_API_KEY"] = "ed610003b55340b3a1d243008ef2c437"
os.environ["AZURE_ENDPOINT"] = "https://bg.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2024-08-01-preview"
os.environ["AZURE_DEPLOYMENT_NAME"] = "gpt-omni"
 
try:
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        temperature=0.7
    )
except Exception as e:
    print(f"❌ Failed to Initialize Azure OpenAI Model: {e}")
    raise
 
app = Flask(__name__)
 
@app.route('/getdata', methods=['POST'])
def query_api():
    data = request.get_json()
    user_question = data.get("user_question")
    if not user_question:
        return jsonify({"message": "No user_question provided", "result": {}}), 400
 
    conn = create_connection()
    snowflake_metadata = get_snowflake_metadata(conn)
    if not snowflake_metadata:
        conn.close()
        return jsonify({"message": "Metadata retrieval failed.", "result": {}}), 500
    with open("instructions.txt", "r", encoding="utf-8") as file:
        system_prompt = file.read().strip()
    metadata_prompt = f"{system_prompt}\n\nUser Question:\n{user_question}"
    try:
        llm_response = llm.invoke(metadata_prompt).content.strip()
        sql_match = re.search(r"```sql\n(.*?)\n```", llm_response, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            conn.close()
            return jsonify({"message": "LLM did not return a valid SQL query format.", "result": {}}), 500
    except Exception as e:
        conn.close()
        return jsonify({"message": f"Error generating SQL query: {str(e)}", "result": {}}), 500
   
    result_df = query_snowflake(conn, sql_query)
    explanation_prompt = f"Explain the meaning of the following query results in one sentence:\n{result_df.head(10).to_json()}"
    explanation_response = llm.invoke(explanation_prompt).content.strip()
    conn.close()
   
    result_list = result_df.to_dict(orient="records")
    graph_png = visual_generate(llm, sql_query, result_list, explanation_response)
    graph_png_url = f"data:image/png;base64,{graph_png}" if graph_png else ""
   
    return jsonify({
         "message": explanation_response,
         "result": result_list,
         "image": graph_png_url
    })

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000)
