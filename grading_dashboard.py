import streamlit as st
import pandas as pd
import re


def get_longest_sentences(transcript):
    # Initialize variables to hold the current longest sentences and their lengths
    longest_doctor_sentence = longest_patient_sentence = ""
    longest_doctor_length = longest_patient_length = 0

    # Split the transcript into parts, keeping the speaker tags
    parts = re.split(r"(\[doctor\]|\[patient\])", transcript)

    # Initialize a variable to remember the last speaker
    last_speaker = None

    for part in parts:
        if part in ["[doctor]", "[patient]"]:
            last_speaker = part  # Update the last speaker
        else:
            # If this part is a dialogue, split it into sentences and find the longest one
            sentences = re.split(r"\.|\?|\!", part)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:  # Check if the sentence is not empty
                    if (
                        last_speaker == "[doctor]"
                        and len(sentence) > longest_doctor_length
                    ):
                        longest_doctor_sentence = sentence
                        longest_doctor_length = len(sentence)
                    elif (
                        last_speaker == "[patient]"
                        and len(sentence) > longest_patient_length
                    ):
                        longest_patient_sentence = sentence
                        longest_patient_length = len(sentence)

    return longest_doctor_sentence, longest_patient_sentence


def get_questions(transcript, speaker):
    """
    Extract questions based on interrogative patterns from the dialogue of a specific speaker.

    Parameters:
    - transcript: The full conversation as a string.
    - speaker: "doctor" or "patient" to specify whose questions to find.

    Returns:
    A list of sentences that are likely to be questions asked by the specified speaker.
    """
    # Patterns that typically start a question in English
    question_patterns = [
        "can you",
        "what",
        "how",
        "do you",
        "have you",
        "is it",
        "are you",
        "did you",
        "would you",
        "could you",
        "should you",
        "will you",
    ]

    # Compile a single regular expression pattern that matches any of the question patterns
    pattern = re.compile("|".join(question_patterns), re.IGNORECASE)

    # Split the transcript into chunks by speaker
    chunks = re.split(r"\[(doctor|patient)\]", transcript)

    questions = []
    for i, chunk in enumerate(chunks):
        if chunks[i - 1].lower().strip().startswith(speaker) and pattern.search(chunk):
            print(chunk)
            sentences = re.split(r"(?<=\.)\s*", chunk)
            for sentence in sentences:
                if pattern.search(sentence):
                    questions.append(sentence.strip())

    return questions


# Title
st.title("Grading Dashboard")


# Load dataframe
@st.cache_data
def load_data():
    # Make sure to update the path to where your CSV file is located
    return pd.read_csv("responses/all.csv")


df = load_data()

# Sort and group by ID
df.sort_values(by="id", inplace=True)
grouped_df = df.groupby("id")
all_ids = list(df["id"].unique())

# State to keep track of current id index
if "current_id" not in st.session_state:
    st.session_state["current_id"] = df["id"].unique()[0]


# Function to navigate to the next or previous ID
def navigate_id(direction):
    current_index = all_ids.index(st.session_state["current_id"])
    if direction == "next" and current_index < len(all_ids) - 1:
        st.session_state["current_id"] = all_ids[current_index + 1]
    elif direction == "prev" and current_index > 0:
        st.session_state["current_id"] = all_ids[current_index - 1]


# Displaying data for the current id
current_group = grouped_df.get_group(st.session_state["current_id"])

st.write(f"### ID: {st.session_state['current_id']}")

# Show the transcript
transcript = current_group["transcript"].values[0]

# Compute the longest sentence in the transcript
longest_doctor_sentence, longest_patient_sentence = get_longest_sentences(transcript)

# Compute the first quest patient asked
patient_questions = get_questions(transcript, "patient")
doctor_questions = get_questions(transcript, "doctor")
question_answers = {
    "What is the first question the patient asked the doctor?": (
        patient_questions[0] if patient_questions else None
    ),
    "What is the second question the patient asked the doctor?": (
        patient_questions[1] if len(patient_questions) > 1 else None
    ),
    "What is the last question the doctor asked the patient?": (
        doctor_questions[-1] if doctor_questions else None
    ),
    "What is the second to last question the doctor asked the patient?": (
        doctor_questions[-2] if len(doctor_questions) > 1 else None
    ),
    "What is the longest sentence the patient said?": longest_patient_sentence,
}

with st.sidebar:
    st.write("### Transcript")
    st.write(transcript)


# Iterate through each question in the current ID group
for _, row in current_group.iterrows():
    st.write(f"#### Question: {row['question']}")

    # Show answer if in the question_answers dictionary
    if row["question"] in question_answers:
        with st.expander("Show Possible Answer"):
            st.write(question_answers[row["question"]])

    # Show Answer textarea
    st.text_area(
        "Answer",
        placeholder="Type your answer here",
        height=100,
        key=f"{row['id']}_{row['question']}-answer",
    )

    # Initialize an empty DataFrame for grading
    grading_df = pd.DataFrame(columns=["Model", "Result", "Rating"])

    models = [
        "gemma_result",
        "mistral_result",
        "gpt4_result",
        "opus_result",
        "gpt3.5_result",
        "sonnet_result",
    ]
    for model in models:
        # Construct a new row DataFrame to concatenate
        new_row = pd.DataFrame(
            {
                "Model": [
                    model.replace("_", " ").title().replace("Result", "").strip()
                ],
                "Result": [row[model]],
                "Rating": None,
            }
        )
        grading_df = pd.concat([grading_df, new_row], ignore_index=True)

    # Display the DataFrame for grading
    st.data_editor(
        grading_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rating": st.column_config.NumberColumn(
                "Your rating",
                help="How much do you like this response (1-3)? 1 is wrong, 2 is correct, 3 is correct and well-written.",
                min_value=1,
                max_value=3,
                step=1,
                format="%d ⭐",
            ),
        },
    )

# Navigation buttons
col1, col2 = st.columns(2)
with col1:
    st.button("Previous ID", on_click=navigate_id, args=("prev",))
with col2:
    st.button("Next ID", on_click=navigate_id, args=("next",))
