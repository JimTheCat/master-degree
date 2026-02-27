"""
Data Manager for Text Annotations
Handles loading/saving annotations and static data
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, List
from config import TEXTS_FILE, CATEGORIES_FILE


# ============================================================================
# CACHED DATA LOADERS
# ============================================================================

@st.cache_data
def load_texts() -> pd.DataFrame:
    """
    Load texts from file.

    Format: Each line is "ID TEXT_CONTENT"
    Returns: DataFrame with columns ['id', 'tekst']
    """
    rows = []
    with open(TEXTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                rows.append({
                    "id": parts[0],
                    "tekst": parts[1] if len(parts) == 2 else ""
                })
    return pd.DataFrame(rows)


@st.cache_data
def load_categories() -> Dict[str, List[str]]:
    """
    Load categories from JSON file.

    Returns: Dictionary with category groups (emocje, techniki_retoryczne)
    """
    with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# ANNOTATIONS MANAGER
# ============================================================================

class AnnotationsManager:
    """
    Manages annotation storage and retrieval.
    Stores annotations as {text_id: {"emocje": [...], "techniki_retoryczne": [...]}}
    """

    @staticmethod
    def load_from_csv(filepath: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Load annotations from CSV file with JSON format.

        Args:
            filepath: Path to CSV file (format: id;json_data)

        Returns:
            Dictionary mapping text_id to dict with category groups
            e.g., {"1": {"emocje": [...], "techniki_retoryczne": [...]}}
        """
        if not os.path.exists(filepath):
            return {}

        try:
            annotations = {}
            with open(filepath, 'r', encoding='utf-8') as f:
                # Skip header line
                next(f, None)

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Split on first semicolon only
                    parts = line.split(';', 1)
                    if len(parts) == 2:
                        text_id, json_str = parts
                        try:
                            # Parse JSON data
                            data = json.loads(json_str)
                            annotations[text_id] = data
                        except json.JSONDecodeError:
                            # Fallback for old format or empty data
                            annotations[text_id] = {"emocje": [], "techniki_retoryczne": []}

            return annotations

        except Exception as e:
            st.error(f"Error loading annotations from {filepath}: {e}")
            return {}

    @staticmethod
    def save_to_csv(annotations: Dict[str, Dict[str, List[str]]], filepath: str) -> bool:
        """
        Save annotations to CSV file with JSON format.

        Args:
            annotations: Dictionary mapping text_id to dict with category groups
            filepath: Path where CSV should be saved

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write header
                f.write("id;kategorie\n")

                # Write each annotation as JSON (including empty categories)
                for text_id, cats_dict in annotations.items():
                    # Handle None or empty values - save with default structure
                    if cats_dict is None:
                        cats_dict = {"emocje": [], "techniki_retoryczne": []}

                    # Remove 'annotated' flag if present (legacy cleanup)
                    cats_to_save = {k: v for k, v in cats_dict.items() if k != "annotated"}

                    json_str = json.dumps(cats_to_save, ensure_ascii=False)
                    f.write(f"{text_id};{json_str}\n")

            return True

        except Exception as e:
            st.error(f"Error saving annotations to {filepath}: {e}")
            return False

    @staticmethod
    def find_first_unannotated(texts_df: pd.DataFrame,
                               annotations: Dict[str, Dict[str, List[str]]]) -> int:
        """
        Find the index of the first text without annotations.

        Args:
            texts_df: DataFrame of texts
            annotations: Current annotations dictionary

        Returns:
            Index of first unannotated text, or 0 if all are annotated
        """
        for i, row in texts_df.iterrows():
            text_id = str(row["id"])
            # Text is annotated if it exists in annotations dictionary
            if text_id not in annotations:
                return i
        return 0

    @staticmethod
    def count_annotated(annotations: Dict[str, Dict[str, List[str]]]) -> int:
        """
        Count how many texts have been annotated (with or without categories).

        Args:
            annotations: Annotations dictionary

        Returns:
            Number of texts with annotations
        """
        # Each entry in dictionary means text was annotated
        return len(annotations)
