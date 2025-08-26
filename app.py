import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()


def get_system_prompt() -> str:
    return (
        "You are a patient, expert Python programming tutor. "
        "Answer questions clearly and concisely, provide runnable code examples when useful, "
        "and explain reasoning step by step only when requested or helpful. "
        "Prefer Pythonic, idiomatic solutions. When showing code, include minimal examples and "
        "brief comments. If the question is ambiguous, ask clarifying questions."
    )


def build_messages(history, system_prompt: str):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    return messages


def generate_response(messages, model: str = "gpt-4", temperature: float = 0.2, max_tokens: int = 800) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def render_chat(history):
    for msg in history:
        role = "assistant" if msg["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(msg["content"])


def main():
    st.set_page_config(page_title="Python Tutor Chatbot", page_icon="🐍", layout="centered")
    st.title("🐍 Python Programming Tutor")
    st.caption(
        "Ask Python questions, get helpful, concise answers with runnable examples. "
        "Powered by OpenAI. Set your OPENAI_API_KEY environment variable before running."
    )

    # Session state initialization
    if "history" not in st.session_state:
        st.session_state.history = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = get_system_prompt()

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox(
            "Model",
            options=["gpt-4", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4 is more capable; gpt-3.5-turbo is faster and cheaper.",
        )
        temperature = st.slider(
            "Creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Lower values make answers more focused and deterministic.",
        )
        max_tokens = st.slider(
            "Max tokens in reply",
            min_value=128,
            max_value=2000,
            value=800,
            step=64,
            help="Upper bound on the response length.",
        )
        with st.expander("System behavior (advanced)"):
            system_prompt_text = st.text_area(
                "System prompt",
                value=st.session_state.system_prompt,
                height=180,
                help="Controls the assistant's behavior and tone.",
            )
            if st.button("Update system prompt"):
                st.session_state.system_prompt = system_prompt_text
                st.success("System prompt updated.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear chat"):
                st.session_state.history = []
                st.success("Chat cleared.")
        with col2:
            example_clicked = st.button("Load examples")

    # Load some example Q&A into the chat
    if example_clicked:
        examples = [
            {"role": "user", "content": "How do I read a CSV file into a pandas DataFrame?"},
            {"role": "assistant", "content": "Use pandas.read_csv:\n\n```python\nimport pandas as pd\n\ndf = pd.read_csv('data.csv')\nprint(df.head())\n```\nYou can pass parameters like `sep`, `dtype`, and `usecols` to control parsing."},
            {"role": "user", "content": "What is the difference between a list and a tuple?"},
            {"role": "assistant", "content": "Lists are mutable and typically used for homogeneous collections you need to modify. Tuples are immutable, hashable when containing hashable items, and suitable for fixed collections or dict keys:\n\n- list: `[]`, mutable\n- tuple: `()`, immutable"},
        ]
        st.session_state.history.extend(examples)

    # Render chat history
    if st.session_state.history:
        render_chat(st.session_state.history)

    # Chat input
    user_input = st.chat_input("Ask a Python question...")
    if user_input:
        # Append user message
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    messages = build_messages(st.session_state.history, st.session_state.system_prompt)
                    reply = generate_response(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    reply = f"Sorry, I ran into an error: {e}"

            st.markdown(reply)
        # Append assistant message
        st.session_state.history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()