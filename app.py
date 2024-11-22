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
from sqlalchemy import create_engine, text
import os


global df

UI_TAB_TITLE = "KKL PRIVATE GPT"
AVATAR_BOT = Path(r"static\KKL_Logo.svg")
AVATAR_BOT2 = Path(r"static\img.ico")
MODES = ["Update Column Names", "Start Chat"]


def messages_to_prompt(messages):
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


def completion_to_prompt(completion):
    return "[INST] " + str(completion).strip() + " [/INST]"


def initialize_llm():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    llm = LlamaCPP(
        model_path="models/Mistral-7B-Instruct-v0.3.Q8_0.gguf",
        temperature=0.1,
        max_new_tokens=1024,
        context_window=16348,  # max 32k
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": -1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False,
    )

    embed = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    Settings.llm = llm
    Settings.embed_model = embed
    Settings.tokenizer = tokenizer
    return llm, embed


def store_df_in_db(updated_df=None):
    # global query_engine
    # global engine
    global llm
    global embed
    print("\n\n\n UPDATED DF :", updated_df,"\n\n\n")

    db_file = 'data22.db'
    if os.path.exists(db_file):
        print(f"Deleting existing database file: {db_file}")
        os.remove(db_file)


    src = sqlite3.connect(':memory:')
    dst = sqlite3.connect('data22.db')

    updated_df.to_sql("sample_table", src)

    llm, embed = initialize_llm()

    src.backup(dst)
    dst.close()
    src.close()


def ask_db(query):
    try:
        global engine

        engine = create_engine('sqlite:///data22.db')

        sql_database = SQLDatabase(engine, include_tables=["sample_table"])

        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=["sample_table"],
            llm=llm,
            embed_model=embed
        )
        print("Done.\n")
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
    return gr.update(value=column_info, visible=True), gr.update(visible=True)  # Show the button here

@singleton
class PrivateGptUi:
    @inject
    def __init__(self) -> None:
        self._system_prompt = "This is the system prompt."
        self._selected_filename = None
        self.updated_columns = []
        self.updated_descriptions = []
        self.is_file_loaded = False

    def _chat(self, message: str, history: list[list[str]], mode: str, *_: Any) -> Any:
        response = ask_db(message)
        return response.text if hasattr(response, 'text') else str(response)

    def _set_current_mode(self, mode: str) -> Any:
        global db_connection, query_engine, df

        self._system_prompt = f"System prompt updated for mode: {mode}"

        # When the mode is "Update Column Names", close the DB connection and reset the session
        if mode == "Update Column Names":
            if engine:
                engine.dispose()
                print("Session variables reset.")

            return [
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(placeholder=self._system_prompt, interactive=True)
            ]
        else:
            store_df_in_db(updated_df=df)

            return [
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(placeholder=self._system_prompt, interactive=True)
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

        return "DF UPDATE SUCCESS"

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
                    height: 60%; 
                    border-radius: 8px;
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

            """
        ) as blocks:
            avatar_byte = AVATAR_BOT.read_bytes()
            f_base64 = f"data:image/png;base64,{base64.b64encode(avatar_byte).decode('utf-8')}"
            gr.HTML(f"""
                <div class='logo'>
                    <img class='header-ico' src='{f_base64}'>
                    <h1 style="color: white;">KKL PRIVATE GPT</h1>  <!-- Title with white color -->
                </div>
            """)
            # with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                mode = gr.Radio(MODES, label="Mode", value="Update Column Names", interactive=False)
                explanation_mode = gr.Textbox(
                    placeholder="Get contextualized answers from selected files.",
                    show_label=False,
                    max_lines=3,
                    interactive=False,
                )

                with gr.Column(visible=True) as csv_mode:
                    file_input = gr.File(label="Upload CSV/Excel File")
                    save_button = gr.Button("Save Updated Columns", visible= False,elem_classes=["save-btn"], )

                    df_output = gr.Dataframe(type="pandas",
                                             interactive=True,column_widths=[200, 200,300], visible=False)

                file_input.upload(process_file, inputs=file_input, outputs=[df_output, save_button])

                save_button.click(self._save_column_updates, inputs=df_output)

                file_input.upload(self._on_file_uploaded, inputs=file_input, outputs=[mode])

                with gr.Column(visible=False) as chat_mode:
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
                        additional_inputs=[mode],
                    )

            mode.change(self._set_current_mode, inputs=mode, outputs=[csv_mode, chat_mode, explanation_mode])

        return blocks

    def _on_file_uploaded(self, file: gr.File) -> gr.update:
        self.is_file_loaded = True
        return gr.update(interactive=True)

    def get_ui_blocks(self) -> gr.Blocks:
        return self._build_ui_blocks()

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        gr.mount_gradio_app(app, blocks, path=path, favicon_path=AVATAR_BOT)


if __name__ == "__main__":
    ui = PrivateGptUi()
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False)
