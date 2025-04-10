import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import pandas as pd
from sqlalchemy import create_engine, MetaData, inspect
from fastapi.middleware.cors import CORSMiddleware
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Dict

app = FastAPI(
    title="Energy Analytics API",
    description="API for analyzing energy data using Groq AI with RAG",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
groq_client = Groq(api_key=os.getenv("API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Database configuration
DATABASE_SERVER = 'DESKTOP-CPE1NTJ'
DATABASE_NAME = 'SANDBOX_1107_CLONED_1BT'
conn_str = f'mssql+pyodbc://{DATABASE_SERVER}/{DATABASE_NAME}?driver=ODBC+Driver+17+for+SQL+Server'

class QueryRequest(BaseModel):
    user_query: str

# --- RAG Components ---
def get_dynamic_schema() -> List[Document]:
    """Retrieve schema information and convert to documents for vectorization"""
    engine = create_engine(conn_str)
    inspector = inspect(engine)
    documents = []
    
    for table_name in inspector.get_table_names(schema="MODEL_DATA"):
        columns = inspector.get_columns(table_name, schema="MODEL_DATA")
        column_info = "\n".join([f"- {col['name']} ({col['type']})" for col in columns])
        
        fks = inspector.get_foreign_keys(table_name, schema="MODEL_DATA")
        fk_info = "\n".join([f"-> Related to {fk['referred_table']}.{fk['referred_columns'][0]}" 
                          for fk in fks]) if fks else "No foreign keys"
        
        content = f"""
        Table: MODEL_DATA.{table_name}
        Columns:
        {column_info}
        Relationships:
        {fk_info}
        """
        documents.append(Document(
            page_content=content,
            metadata={"table": table_name, "type": "schema"}
        ))
    
    example_queries = [
        Document(
            page_content="Show top 5 power plants by summer capacity. Use MODEL_DATA.POWER_PLANT_SUMMARY_MV with SUMMER_CAPACITY.",
            metadata={"type": "example"}
        ),
        Document(
            page_content="Find power plants with capacity over 100MW. Use MODEL_DATA.POWER_PLANT_SUMMARY_MV table.",
            metadata={"type": "example"}
        )
    ]
    return documents + example_queries

schema_docs = get_dynamic_schema()
vector_store = FAISS.from_documents(schema_docs, embeddings)

def retrieve_rag_context(query: str, k: int = 3) -> str:
    relevant_docs = vector_store.similarity_search(query, k=k)
    context = []
    
    for doc in relevant_docs:
        if doc.metadata["type"] == "schema":
            context.append(f"Schema Context:\n{doc.page_content}")
        elif doc.metadata["type"] == "example":
            context.append(f"Example Query:\n{doc.page_content}")
    
    return "\n\n".join(context)

PROMPT_TEMPLATE = """
You are a SQL expert generating Microsoft SQL Server queries. Use the following context:

{context}

Guidelines:
- Generate ONLY the SQL query without any formatting or explanations
- Use WITH(NOLOCK) for all table accesses
- Always prefix tables with MODEL_DATA schema
- Use TOP instead of LIMIT
- Do NOT use backticks (`) or any special formatting
- Use proper SQL Server syntax

Catogry Columns:
- **SCENARIO** : Varchar(50), null, 'Base Case', 'High Gas Price', 'Low Gas Price'
- **CYCLE** : Varchar(50), null, 'DA', 'RT'
- **PERIOD** : Varchar(50), null, 'P', 'O'
- **POWER_PLANT_NAME** : VARCHAR(100),null, 'Power Plant 1' to 'Power Plant 100'
- **UNIT_TYPE** : Varchar(50), null, 'Wind', 'IC/GT', 'ES', 'CC'
- **PRIMARY_FUEL** : Varchar(50), null, 'Wind', 'NG'
- **DATA_TAG_ID** : BIGINT, null, 1,2,3 ('Base Case'='1', 'High Gas Price'='2', 'Low Gas Price'='3)
- *POWER_PLANT_TYPE* : VARCHAR(50), null, 'Terminal' , 'Nuclear' , 'Hydro , 'Solar' , 'Wind'
- **TRANSATION_CATEGORY** : VARCHAR(50), null, 'Sale', 'Purchase'
- **PRODUCT_TYPE** : VARCHAR(50), null, 'Energy', 'CO2', 'NG' 
- **FUEL_TYPE** : VARCHAR(50), null, 'PV' , 'Wind' , 'Gas' , 'Coal'
- **ZONE_ID** : VARCHAR(50), null, 'Zone 1', 'Zone 2', 'Zone 3'
User Query: {query}

Generate SQL:
"""

def clean_sql_response(sql: str) -> str:
    """Remove any formatting or special characters from SQL response"""
    # Remove code blocks and backticks
    sql = re.sub(r'```sql?\s*', '', sql, flags=re.IGNORECASE)
    sql = sql.replace('```', '')
    
    # Remove leading/trailing special characters
    sql = sql.strip('*#\n\t ')
    
    # Ensure proper termination
    if not sql.endswith(';'):
        sql += ';'
    return sql

def generate_sql_with_rag(query: str) -> str:
    context = retrieve_rag_context(query)
    full_prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            model="llama3-70b-8192",
            temperature=0.1,
            max_tokens=1024,
            stop=[";"]
        )
        raw_sql = response.choices[0].message.content.strip()
        return clean_sql_response(raw_sql)
    except Exception as e:
        raise HTTPException(500, f"SQL generation failed: {str(e)}")

def validate_sql(sql: str) -> bool:
    forbidden_patterns = [
        r"\b(DELETE|UPDATE|INSERT|DROP|ALTER|GRANT)\b",
        r"`",  # Prevent backticks
        r"--",  # Prevent SQL comments
        r"\/\*.*?\*\/"  # Prevent block comments
    ]
    return not any(re.search(p, sql, re.IGNORECASE) for p in forbidden_patterns)

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        sql = generate_sql_with_rag(request.user_query)
        print(f"Cleaned SQL: {sql}")
        
        if not validate_sql(sql):
            raise HTTPException(400, "Invalid SQL detected")
        
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
        
        return {
            "sql": sql,
            "results": df.to_dict(orient="records"),
            "stats": {
                "rows": len(df),
                "columns": list(df.columns),
                "sample": df.head(5).to_dict()
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Execution error: {str(e)}")
        raise HTTPException(500, f"Query execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)