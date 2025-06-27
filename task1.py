"""
TEXT SUMMARIZATION TOOL - NLP Internship Project
Author: [Your Name]
Date: 2025-06-20
Description: Summarizes lengthy articles using HuggingFace transformers.
"""

import os
import textwrap
from transformers import pipeline

# -------------------------------
# Utility Functions
# -------------------------------

def load_summarizer(model_name="facebook/bart-large-cnn"):
    """
    Loads a summarization model from Hugging Face transformers.

    Parameters:
        model_name (str): The name of the pre-trained model.

    Returns:
        pipeline: A Hugging Face summarization pipeline.
    """
    print(f"Loading summarizer model: {model_name}")
    summarizer_pipeline = pipeline("summarization", model=model_name)
    return summarizer_pipeline

def read_text_input():
    """
    Reads multiline user input until an empty line is entered.

    Returns:
        str: The full concatenated text input.
    """
    print("\nEnter your text (Press Enter twice to finish):\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return " ".join(lines).strip()

def summarize_article(summarizer, article_text, max_length=130, min_length=30):
    """
    Summarizes a given article using the summarizer pipeline.

    Parameters:
        summarizer (pipeline): The Hugging Face summarization pipeline.
        article_text (str): The original article text.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: The summarized text.
    """
    print("Summarizing your text. Please wait...\n")
    summary = summarizer(article_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def save_to_file(summary_text, original_text):
    """
    Saves the original and summarized text to a file.

    Parameters:
        summary_text (str): The generated summary.
        original_text (str): The original input text.
    """
    filename = "summary_output.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("ORIGINAL TEXT:\n")
        f.write(original_text + "\n\n")
        f.write("SUMMARY:\n")
        f.write(summary_text)
    print(f"Summary saved to: {filename}")

def print_wrapped(title, content, width=90):
    """
    Nicely formats and prints long text with a title.

    Parameters:
        title (str): Section title.
        content (str): Text content to wrap.
        width (int): Line width for wrapping.
    """
    print(f"\n{title}")
    print("-" * len(title))
    print(textwrap.fill(content, width=width))

# -------------------------------
# Main Flow
# -------------------------------

def main():
    """
    Main function that controls the application flow.
    """
    print("=" * 70)
    print("TEXT SUMMARIZATION TOOL USING NLP".center(70))
    print("=" * 70)

    # Load summarization model
    model_choice = "facebook/bart-large-cnn"
    summarizer = load_summarizer(model_choice)

    # Input method selection
    print("\nChoose input method:")
    print("1. Paste text manually")
    print("2. Load from a .txt file")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        text = read_text_input()
    elif choice == "2":
        file_path = input("Enter path to your .txt file: ").strip()
        if not os.path.isfile(file_path):
            print("Error: File not found!")
            return
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("Error: Invalid choice.")
        return

    if len(text.split()) < 40:
        print("Warning: The text is too short. Please input at least 40 words.")
        return

    # Generate summary
    summary = summarize_article(summarizer, text)

    # Display results
    print_wrapped("ORIGINAL TEXT", text)
    print_wrapped("GENERATED SUMMARY", summary)

    # Ask to save the summary
    save_opt = input("\nDo you want to save the summary to a file? (y/n): ").strip().lower()
    if save_opt == "y":
        save_to_file(summary, text)

    print("\nDone! Thank you for using the summarization tool.")

# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    main()
