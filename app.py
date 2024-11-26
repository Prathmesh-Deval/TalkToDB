import base64
import sqlite3
from pathlib import Path
from typing import Any
import gradio as gr
from fastapi import FastAPI
from gradio.themes.utils.colors import slate
from injector import inject, singleton
import pandas as pd
from transformers import AutoTokenizer
from llama_index.core import SQLDatabase, Settings
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, text, inspect
import os
import json

global llm
global embed
import os


UI_TAB_TITLE = "KKL PRIVATE GPT"
AVATAR_BOT = Path(r"static\logo.svg")
AVATAR_BOT2 = Path(r"static\img.ico")
MODES = ["Update Column Names", "Ingested Databases", "Start Chat"]


def messages_to_prompt1(messages):
    inst_buffer = []

    prompt = ""

    for message in messages:
        if message.role == 'system' or message.role == 'user':
            inst_buffer.append(str(message.content).strip())

        elif message.role == 'assistant':
            prompt += "[INST] " + "\n".join(inst_buffer) + " [/INST]"
            prompt += " " + str(message.content).strip() + "</s>"
            inst_buffer.clear()
        else:
            raise ValueError(f"Unknown message role {message.role}")

    if len(inst_buffer) > 0:
        prompt += "[INST] " + "\n".join(inst_buffer) + " [/INST]"


def completion_to_prompt1(completion):
    return "[INST] " + str(completion).strip() + " [/INST]"
#
def messages_to_prompt_description(messages):
    prompt = ""
    for message in messages:
        if 'role' not in message or 'content' not in message:
            raise ValueError("Each message must be a dictionary with 'role' and 'content' keys.")

        if message['role'] == 'system':
            prompt += f"<|system|>\n{message['content']}</s>\n"
        elif message['role'] == 'user':
            prompt += f"<|user|>\n{message['content']}</s>\n"
        elif message['role'] == 'assistant':
            prompt += f"<|assistant|>\n{message['content']}</s>\n"

    # Ensure we start with a system prompt if not already there
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # Add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt

def completion_to_prompt_description(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"



def initialize_llm():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    llm = LlamaCPP(
        model_path=r"models/Mistral-7B-Instruct-v0.3.Q8_0.gguf",
        temperature=0.1,
        max_new_tokens=1024,
        context_window=16348,  # max 32k
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": -1},
        messages_to_prompt=messages_to_prompt1,
        completion_to_prompt=completion_to_prompt1,
        verbose=False,
    )

    embed = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    Settings.llm = llm
    Settings.embed_model = embed
    Settings.tokenizer = tokenizer
    return llm, embed



llm, embed = initialize_llm()
# print("DIRRRR", dir(llm))


def initialize_llm2():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    llm2 = LlamaCPP(
        model_path=r"models/Mistral-7B-Instruct-v0.3.Q8_0.gguf",
        temperature=0.1,
        max_new_tokens=1024,
        context_window=16348,  # max 32k
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": -1},
        messages_to_prompt=messages_to_prompt1,
        completion_to_prompt=completion_to_prompt1,
        verbose=False,
    )

    embed2 = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    Settings.llm2 = llm2
    Settings.embed_model = embed2
    Settings.tokenizer = tokenizer
    return llm2, embed2


# Example usage in generate_table_description
def generate_table_description(df_output):
    columns = df_output['Edit Column Name'].tolist()
    # Create a prompt for the model
    prompt = f"""
    You are an AI assistant with expertise in nuclear engineering, nuclear science, and complex scientific terminologies. 
    I will provide you with a list of column names that pertain to a nuclear company dataset. 
    These column names may include scientific terms, nuclear terminology, or even terms in multiple languages such as English or German.

    Your task is to generate a brief 2-3 line description for these columns, explaining what each column represents in the context of the nuclear industry.
    Please make sure the description is clear, concise, and uses appropriate scientific language to explain the table.

    Columns: {columns}

    Please provide a short description for the dataset's columns.
    """

    # Now, create the message list as a list of dictionaries
    messages = [
        {"role": "system", "content": "You are an AI assistant with nuclear industry expertise."},
        {"role": "user", "content": prompt}
    ]

    # Convert messages to formatted prompt
    formatted_prompt = messages_to_prompt_description(messages)

    llm2, embed2 = initialize_llm2()

    # Get the response from the model
    response = llm2.complete(formatted_prompt)

    print("\n\n####RES:", response)
    return gr.update(value=response, visible=True)

def store_df_in_db(updated_df=None, filename="unknown", overwrite=False):
    global llm
    global embed
    global tables

    print("\n\n\n UPDATED DF :", updated_df, "\n\n\n")

    table_name = filename

    query = "SELECT name FROM sqlite_master WHERE type='table';"
    src = sqlite3.connect("DATABASE.db")

    cursor = src.cursor()
    cursor.execute(query)
    existing_tables = [row[0] for row in cursor.fetchall()]
    print("Table_LIST", existing_tables)
    cursor.close()


    if table_name in existing_tables and not overwrite:
        print(f"{table_name} : Table Already Exists!")
        src.close()

        return f"Table {table_name} already exists!"

    try:
        print("UPDATING @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        updated_df.to_sql(table_name, src, if_exists='replace' if overwrite else 'fail', index=False)

        # llm, embed = initialize_llm()

        tables = existing_tables if table_name in existing_tables else existing_tables + [table_name]

        print(f"Table {table_name} stored successfully.")

        src.close()

        return f"Table {table_name} updated successfully!" if overwrite else f"Table {table_name} stored successfully!"
    except Exception as e:
        print(f"Error while storing table {table_name}: {e}")
        return f"Error: {str(e)}"
    finally:
        overwrite = False
        src.close()


def ask_db(query, selected_table):
    try:
        print("ASK DB")
        # global engine
        global tables


        # src = sqlite3.connect("DATABASE.db")
        #
        # query2= "SELECT name FROM sqlite_master WHERE type='table';"
        #
        # cursor = src.cursor()
        # cursor.execute(query2)
        # tables = [row[0] for row in cursor.fetchall()]
        # print("Table_LIST", tables)
        # cursor.close()
        # src.close()
        selected_tables = list()
        selected_tables.append(selected_table)

        engine = create_engine('sqlite:///DATABASE.db')

        sql_database = SQLDatabase(engine, include_tables=selected_tables)

        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=selected_tables,
            llm=llm,
            embed_model=embed
        )
        print("Sending question to model")
        response = query_engine.query(query)
        print("Answer :", response)
        response_meta = {response.metadata['sql_query']}
        print("SQL QUERY USED: ", response_meta)
        engine.dispose()
        return response

    except Exception as e:
        return f"Error during query execution: {e}"


def process_file(file):
    global df
    global file_name
    print("ORIGINAL FILENAME", file.name)

    file_name = file.name.split("\\")[-1].split(".")[0]
    print("\nFileName-> ", file_name)

    if file.name.endswith('.csv'):
        df = pd.read_csv(file.name)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file.name)
    else:
        return "Unsupported file format. Please upload a CSV or Excel file."

    column_info = pd.DataFrame({
        'Existing Column Name': df.columns,
        "Edit Column Name": df.columns,
        'ADD Description': ['' for _ in df.columns]
    })
    gr.update(interactive=True)
    return gr.update(value=file_name, visible=True), gr.update(value=column_info, visible=True), gr.update(
        visible=True), gr.update(visible=True), gr.update(visible=True) , gr.update(visible=True)


def get_database_tables():
    if os.path.exists("DATABASE.db"):
        src = sqlite3.connect("DATABASE.db")
        cursor = src.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables1 = [row[0] for row in cursor.fetchall()]
        cursor.close()
        src.close()
        return tables1
    return []


def view_table_columns(table_name):
    print("Fetching table columns...")

    if os.path.exists("DATABASE.db"):
        src = sqlite3.connect("DATABASE.db")
        query = f"PRAGMA table_info({table_name});"
        df = pd.read_sql(query, src)

        if not df.empty:
            columns = df['name'].tolist()

            stats_query = f"SELECT * FROM {table_name}"
            data = pd.read_sql(stats_query, src)

            stats = {}

            for column in columns:
                column_data = data[column]

                numeric_column = pd.to_numeric(column_data, errors='coerce')

                stats[column] = {
                    "Column Name": column,
                    'Data Type': column_data.dtype,
                    'Null Values': column_data.isnull().sum(),
                    'Unique Count': column_data.nunique(),
                    'Max Value': numeric_column.max() if numeric_column.notnull().any() else None,
                    'Min Value': numeric_column.min() if numeric_column.notnull().any() else None
                }
            src.close()
            stats_df = pd.DataFrame(stats).T

            print("Stats DataFrame:\n", stats_df)

            return gr.update(value=stats_df, visible=True), gr.update(visible=True)

        else:
            return gr.update(value="No columns found in the table.", visible=True), gr.update(visible=False)

    return gr.update(value="Database not found.", visible=True), gr.update(visible=False)


def delete_table(table_name):
    try:
        if os.path.exists("DATABASE.db"):
            src = sqlite3.connect("DATABASE.db")  # cursor = connection.cursor()
            cursor = src.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
            src.commit()
            cursor.close()
            src.close()

            return f"Table '{table_name}' has been deleted successfully."
        return "No database found."
    except Exception as e:
        return f"Error deleting table '{table_name}': {e}"


def hide_data_frame():
    return gr.update(visible=False), gr.update(visible=False)


@singleton
class PrivateGptUi:
    @inject
    def __init__(self) -> None:
        self._system_prompt = "This is the system prompt."
        self._selected_filename = None
        self.updated_columns = []
        self.updated_descriptions = []
        self.is_file_loaded = False

    def _chat(self, message: str, history: list[list[str]], mode: str, selected_table: str, *_: Any) -> Any:
        print(f"Selected Table: {selected_table}")

        response = ask_db(message, selected_table)
        return response.text if hasattr(response, 'text') else str(response)

    def _set_current_mode(self, mode: str) -> Any:
        global db_connection, query_engine, df

        if mode == "Start Chat":
            if os.path.exists("DATABASE.db"):
                src = sqlite3.connect("DATABASE.db")

                query = "SELECT name FROM sqlite_master WHERE type='table';"
                cursor = src.cursor()
                cursor.execute(query)
                table_list = [row[0] for row in cursor.fetchall()]
                cursor.close()
                src.close()
                if len(table_list) == 0:
                    return [
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(interactive=False),
                        gr.update(value="No Tables in Database. Please upload a file to continue Chat.", visible=True),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),

                    ]
            else:
                return [
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    gr.update(value="No Tables in Database. Please upload a file to continue Chat.", visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),


                ]

        self._system_prompt = f"System prompt updated for mode: {mode}"

        if mode == "Update Column Names":
            self._system_prompt = f"Preprocess CSV and Ingest into Database."


            return [
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(placeholder=self._system_prompt, interactive=False, visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),

            ]
        elif mode == "Start Chat":
            # store_df_in_db(updated_df=df)
            updated_tables = get_database_tables()
            default_table = updated_tables[0] if updated_tables else "No Tables Available"
            return [
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(placeholder=self._system_prompt, interactive=True, visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(choices=updated_tables, value=default_table, visible=True),  # Update dropdown
                gr.update(visible=False)



            ]
        elif mode=="Ingested Databases":
            self._system_prompt = f"Database Details:"
            updated_tables = get_database_tables()

            return [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(placeholder=self._system_prompt, interactive=False, visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(choices=updated_tables, visible= True)

            ]


    def _save_column_updates(self, updated_df):
        global df
        original_columns = updated_df['Existing Column Name'].tolist()
        updated_columns = updated_df['Edit Column Name'].tolist()
        updated_descriptions = updated_df['ADD Description'].tolist()

        print("Existing Column Names:", original_columns)
        print("Updated Column Names:", updated_columns)
        print("Updated Descriptions:", updated_descriptions)

        df.columns = updated_columns

        print("Updated DataFrame:")
        print(df)

        return gr.update(value="Column Names Updated Successfully!", visible=True)

    def _build_ui_blocks(self) -> gr.Blocks:
        with gr.Blocks(
                title=UI_TAB_TITLE,
                theme=gr.themes.Soft(primary_hue=slate),
                css=""" 
                body, .gradio-container {
                        font-family: Arial, Helvetica, sans-serif;
                }
                .logo { 
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100px;
                    background-color: #00538C;

                }
                .logo img { 
                    height: 70%;
                    background-color: rgb(0, 83, 140); 
                    padding-right: 30px;

                }
                .header-ico {
                    height: 20px;
                    background-color: antiquewhite;
                    border-radius: 2px;
                    margin-right: 20px;
                }
                /* Custom CSS for the Save Button */
                .save-btn {
                    background-color: #28a745; /* Green */
                    color: white;
                    border-radius: 5px;
                    padding: 6px 12px; /* Small button */
                    font-size: 12px;
                    cursor: pointer;
                    width: auto; /* Make the button only as wide as its content */
                    margin-left: auto; /* Center the button horizontally within its container */

                }
                .save-btn:hover {
                    background-color: #218838; /* Darker green on hover */
                }
                .success-msg {
                    color: green;          /* Font color set to green */
                    font-size: 12px;       /* Font size set to 12px */
                    width: 20%;            /* Control the width of the textbox */
                    height: 50px;
                    margin-left: 80%;      /* Align the textbox towards the right (adjust as needed) */
                    text-align: center;    /* Center the text inside the box */

                }
                .ingest-btn {
                    background-color: rgb(0, 83, 140);/* blue */
                    color: white;
                    border-radius: 5px;
                    # padding: 6px 12px; /* Small button */
                    font-size: 12px;
                    cursor: pointer;
                    # width: auto; /* Make the button only as wide as its content */
                    margin-left: auto; /* Center the button horizontally within its container */

                }

                /* Style for the confirmation message */
                .confirmation-message textarea {
                    font-weight: bold;
                    border: none; /* Remove border */
                    background: transparent; /* Transparent background */
                    text-align: center; /* Center align */
                    width: 100%; /* Full width for better alignment */
                }

                .confirm-btn, .cancel-btn {
                    background-color: #28a745; /* Green */
                    color: white;
                    border-radius: 5px;
                    padding: 6px 12px; /* Reduced padding for smaller buttons */
                    font-size: 12px; /* Smaller font size */
                    cursor: pointer;
                    width: auto;  /* Auto width based on content */
                    margin-right: 10px; /* Small space between buttons */
                    display: inline-block; /* Display buttons next to each other */
                }

                .confirm-btn:hover {
                    background-color: #218838; /* Darker green on hover */
                }

                .cancel-btn {
                    background-color: #dc3545; /* Red */
                    color: white;
                    border-radius: 5px;
                    padding: 6px 12px; /* Reduced padding for smaller buttons */
                    font-size: 12px; /* Smaller font size */
                    cursor: pointer;
                    width: auto;  /* Auto width based on content */
                }

                .cancel-btn:hover {
                    background-color: #c82333; /* Darker red on hover */
                }

                /* Container for the buttons (to ensure they're in the same row) */
                .button-container {
                    display: flex;
                    justify-content: center; /* Center the buttons */
                    gap: 10px; /* Space between buttons */
                }
                
                .db-table-row {
                    display: flex;
                    justify-content: center; /* Center the content horizontally */
                    align-items: center;    /* Center the content vertically */
                    gap: 10px;              /* Spacing between items */
                    margin: 5px 0;          /* Spacing between rows */
                    height: 35px;           /* Reduce the row height */
                }
                
                /* Styling for the text box */
                .db-text-box {
                    width: 150px;           /* Adjust width to save space */
                    text-align: center;     /* Center-align text */
                }
                
                .db-text-box textarea {
                    height: 100px;           /* Adjust width to save space */
                    text-align: center;     /* Center-align text */
                }
                
                /* Styling for buttons */
                .db-confirm-btn {
                    background-color: #b8e994; /* Pastel green for "Update Columns" */
                    color: black;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 5px;
                    height: 30px; /* Reduce button height */
                    cursor: pointer;
                }
                
                .db-confirm-btn:hover {
                    background-color: #a5d6a7; /* Slightly darker green on hover */
                }
                
                .db-cancel-btn {
                    background-color: #f8a5a5; /* Pastel red for "Delete Table" */
                    color: black;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 5px;
                    height: 30px; /* Reduce button height */
                    cursor: pointer;
                }
                
                .db-cancel-btn:hover {
                    background-color: #f08686; /* Slightly darker red on hover */
                }
                .radio-buttons {
                    # display: flex;
                    # align-items: center;
                    # justify-content: space-evenly;
                }
                
                .radio-buttons label {
                    # padding: 10px 20px;
                    # font-size: 16px;
                    cursor: pointer;
                    # transition: all 0.3s ease;
                    # position: relative;
                    # border-radius: 5px 5px 0 0;  /* Rounded corners on the top */
                }

                
                .radio-buttons label:hover
                    background-color: rgba(0, 83, 140, 0.6);  /* Light hover effect */
                }
                
                # /* Underline effect */
                # .radio-buttons input[type="radio"]:checked + label::after {
                #     content: '';
                #     position: absolute;
                #     bottom: 0;
                #     left: 0;
                #     width: 100%;
                #     height: 2px;
                #     background-color: white; /* Underline color */
                #     border-radius: 1px;
                # }
                
                .kkllogo img{
                    background-color: rgb(0, 83, 140); ;
                }
                
                .ingested_message {
                    color: rgb(0,83,140) !important;

                }
                
                .ingested_message input {
                    font-weight: bold;
                    color: rgb(0,83,140) !important;
                    background-color: #e9ecef; /* Light gray background */
                    border: 1px solid #adb5bd; /* Subtle gray border */
                    border-radius: 5px; /* Rounded corners for elegance */
                    padding: 10px; /* Add spacing for better readability */
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
                    font-size: 13px; /* Standard font size */
                    text-align: center;
                }
            """
        ) as blocks:
            # avatar_byte = AVATAR_BOT.read_bytes()
            # f_base64 = f"data:image/png;base64,{base64.b64encode(avatar_byte).decode('utf-8')}"

            # Read the SVG file and encode it to base64
            with open("static/KKL_Logo.svg", "rb") as svg_file:
                svg_byte = svg_file.read()
                f_base64 = f"data:image/svg+xml;base64,{base64.b64encode(svg_byte).decode('utf-8')}"


            gr.HTML(f"""
                <div class='logo'>
                    <img class='header-ico' src='{f_base64}' />
                    <h1 style="color: white;">KKL PRIVATE GPT</h1>  <!-- Title with white color -->
                </div>
            """)

            with gr.Column(scale=3):

                mode = gr.Radio(MODES, show_label=False, value="Update Column Names", interactive=True, elem_classes=["radio-buttons"])
                explanation_mode = gr.Textbox(
                    placeholder="Preprocess CSV and Ingest into Database.",
                    show_label=False,
                    max_lines=3,
                    interactive=False,
                )

                ingest_button = gr.Button("Ingest File", elem_classes=["ingest-btn"], visible=False)
                ingest_message = gr.Textbox(
                    label="Status",
                    show_label=False,
                    max_lines=1,
                    interactive=False,
                    elem_classes = ["ingested_message"]
                )

                def save_file_description(file_name: str, description: str) -> None:
                    json_file_path = "table_description.json"

                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r") as file:
                            data = json.load(file)
                    else:
                        data = {}

                    data[file_name] = description

                    with open(json_file_path, "w") as file:
                        json.dump(data, file, indent=4)

                    print(f"Saved description for file: {file_name}")


                def handle_ingest_file(filename,table_description):
                    ingest_message = gr.update(value="Processing... Please wait.", visible=True)
                    save_file_description(filename, table_description)

                    status = store_df_in_db(updated_df=df, filename=filename)
                    print(status)

                    if "already exists" in status:
                        return (
                            gr.update(visible=True),
                            gr.update(value=status),
                            gr.update(visible=True),
                        )
                    else:
                        return (
                            gr.update(visible=False),
                            gr.update(value=status),
                            gr.update(visible=True),
                        )

                def handle_confirm(filename,table_description):
                    confirmation_message = gr.update(value="Processing... Please wait.", visible=True)

                    save_file_description(filename, table_description)

                    status = store_df_in_db(updated_df=df, filename=filename, overwrite=True)

                    return (
                        gr.update(visible=False),
                        gr.update(value="File Ingested Succesfully!", interactive=False),
                        gr.update(visible=True)
                    )

                def handle_cancel():
                    return (
                        gr.update(visible=False),
                        gr.update(value="File is not Ingested. Rename the table if you want to store it with new name or overwrite it.")
                    )

                with gr.Column(visible=True) as csv_mode:
                    with gr.Row():
                        with gr.Column(visible=False) as confirmation_dialog:
                            confirmation_message = gr.Textbox(
                                value=f"Provided Table Name already exists. Do you want to overwrite?",
                                show_label=False,
                                interactive=False,
                                elem_classes=["confirmation-message"],
                            )
                            with gr.Row():
                                confirm_button = gr.Button("Confirm", elem_classes=["confirm-btn"])
                                cancel_button = gr.Button("Cancel", elem_classes=["cancel-btn"])
                        file_input = gr.File(label="Upload CSV/Excel File")
                        file_name_textbox = gr.Textbox(value="No file uploaded", label="Table Name", interactive=True,
                                                       visible=False)
                    table_description = gr.Textbox(value="", placeholder="Enter a description for the table or Click on Generate description to generate using GenAI.", interactive=True,
                                                   visible=False,show_label=False)
                    with gr.Row():
                        save_button = gr.Button("Save Updated Columns", visible=False, elem_classes=["save-btn"])
                        generate_description= gr.Button("Generate Description ", visible=False, elem_classes=["save-btn"])


                    df_output = gr.Dataframe(type="pandas",
                                             interactive=True, column_widths=[200, 200, 300], visible=False)

                file_input.upload(process_file, inputs=file_input,
                                  outputs=[file_name_textbox, df_output, save_button, ingest_button,table_description,generate_description])

                save_button.click(self._save_column_updates, inputs=df_output, outputs=ingest_message)


                generate_description.click(generate_table_description, inputs=df_output, outputs=table_description)


                file_input.upload(self._on_file_uploaded, inputs=file_input, outputs=[mode])




                confirm_button.click(handle_confirm, inputs=[file_name_textbox,table_description],
                                     outputs=[confirmation_dialog, ingest_message])
                cancel_button.click(handle_cancel, inputs=[], outputs=[confirmation_dialog, ingest_message])

                ingest_button.click(
                    handle_ingest_file,
                    inputs=[file_name_textbox,table_description],
                    outputs=[confirmation_dialog, ingest_message, ingest_message],
                )

                with gr.Column(visible=False) as database_mode:
                    # List all tables in a dropdown or radio button
                    table_selector = gr.Radio(
                        choices=get_database_tables(),  # Populate with current tables
                        label="Select a Table",
                        interactive=True,
                    )

                    # Buttons to act on the selected table
                    view_button = gr.Button(value="View Columns", elem_classes=["db-confirm-btn"])
                    delete_button = gr.Button(value="Delete Table", elem_classes=["db-cancel-btn"])

                    # Status and output
                    status_message = gr.Textbox(label="Status", interactive=False, visible=True)
                    tables_output = gr.Dataframe(label="Table Details", interactive=False, visible=False)

                    # Button to close the table details view
                    close_button = gr.Button(value="Close", elem_classes=["close-btn"], visible=False)

                    def handle_view_columns(selected_table):
                        if not selected_table:
                            return gr.update(value="Please select a table first.", visible=True)

                        return view_table_columns(selected_table)

                    def handle_delete_table(selected_table):
                        if not selected_table:
                            return gr.update(value="Please select a table first.", visible=True)

                        # Perform the delete action
                        status = delete_table(selected_table)

                        # Refresh the table selector
                        updated_tables = get_database_tables()
                        return gr.update(choices=updated_tables), gr.update(value=status)

                    view_button.click(
                        handle_view_columns,
                        inputs=table_selector,
                        outputs=[tables_output, status_message],
                    )

                    delete_button.click(
                        handle_delete_table,
                        inputs=table_selector,
                        outputs=[table_selector, status_message],
                    )

                close_button.click(hide_data_frame, outputs=[tables_output, close_button])

                with gr.Column(visible=False) as chat_mode:

                    tables1 = get_database_tables()
                    default_table = tables1[0] if tables1 else "No Tables Available"

                    # Dropdown for selecting a database table
                    table_dropdown = gr.Dropdown(
                        choices=tables1,
                        value=default_table,
                        label="Select Table To Start Chat",
                        interactive=True,
                    )

                    # Function to handle table changes and update selected table in _chat
                    def update_table(selected_table):
                        # Update the selected table globally or pass it to _chat dynamically
                        self._selected_table = selected_table  # Store it in the class instance for later access

                    # Trigger table update when dropdown changes
                    table_dropdown.change(
                        update_table,
                        inputs=[table_dropdown],
                        outputs=[],
                    )


                    label_text = f"LLM: Mistral:7b-instruct-v0.3-q8_0"
                    _ = gr.ChatInterface(
                        self._chat,
                        chatbot=gr.Chatbot(
                            label=label_text,
                            show_copy_button=True,
                            elem_id="chatbot",
                            render=False,
                            avatar_images=(None, AVATAR_BOT2),
                        ),
                        additional_inputs=[mode,table_dropdown],
                    )

                mode.change(self._set_current_mode, inputs=mode,
                            outputs=[csv_mode, chat_mode, database_mode, explanation_mode, ingest_message, ingest_button, table_dropdown, table_selector])


        return blocks

    def _on_file_uploaded(self, file: gr.File) -> gr.update:
        self.is_file_loaded = True
        return gr.update(interactive=True)

    def get_ui_blocks(self) -> gr.Blocks:
        return self._build_ui_blocks()

    # def mount_in_app(self, app: FastAPI, path: str) -> None:
    #     blocks = self.get_ui_blocks()
    #     blocks.queue()
    #     gr.mount_gradio_app(app, blocks, path=path, favicon_path=AVATAR_BOT)


if __name__ == "__main__":
    ui = PrivateGptUi()
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False)
