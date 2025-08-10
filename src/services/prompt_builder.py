def build_prompt(quiz_type, category, difficulty):
    """
    Crea il prompt per l'LLM.
    """
    return (
        f"You are a quiz generator for children. Generate a single Italian quiz of type '{quiz_type}' "
        f"about the category '{category}', suitable for the given level. The difficulty value must be the number {difficulty}. "
        f"Use only the provided sources. The quiz must follow strictly one of the following JSON schemas:\n\n"

        f"▶ For 'quiz':\n"
        f"{{\"type\": \"quiz\", \"category\": ..., \"question\": ..., \"difficulty\": ..., \"options\": [...], \"answer\": ...}}\n\n"

        f"▶ For 'matching':\n"
        f"{{\"type\": \"matching\", \"category\": ..., \"question\": ..., \"difficulty\": ..., \"pairs\": [{{\"left\": ..., \"right\": ...}}]}}\n\n"

        f"▶ For 'memory':\n"
        f"{{\"type\": \"memory\", \"category\": ..., \"question\": ..., \"difficulty\": ..., \"pairs\": [{{\"front\": ..., \"back\": ...}}]}}\n\n"

        f"▶ For 'sorting':\n"
        f"{{\"type\": \"sorting\", \"category\": ..., \"question\": ..., \"difficulty\": ..., \"items\": [...], \"solution\": [...]}}\n\n"

        f"Only output the raw JSON object. No extra text. Only return a single valid JSON object. "
        f"Use double quotes around all keys and string values. Do not use single quotes. "
        f"Always respond in **ITALIANO**.\n"
        f"Return an empty JSON (i.e. '{{}}') if the context is insufficient."
    )
