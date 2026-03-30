import re
import pandas as pd
import requests
import io
import streamlit as st


def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


def get_sheet_names(uploaded_file):
    """Return the list of sheet names in an Excel file."""
    xls = pd.ExcelFile(uploaded_file)
    return xls.sheet_names


def load_excel(uploaded_file, sheet_name=None):
    """Load a specific sheet from an Excel file. Defaults to first sheet."""
    if sheet_name is None:
        sheet_name = 0
    return pd.read_excel(uploaded_file, sheet_name=sheet_name)


def load_google_sheet(url: str) -> pd.DataFrame:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("Invalid Google Sheets URL. Expected format: https://docs.google.com/spreadsheets/d/<SHEET_ID>/...")

    sheet_id = match.group(1)

    gid = "0"
    gid_match = re.search(r"gid=(\d+)", url)
    if gid_match:
        gid = gid_match.group(1)

    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    response = requests.get(export_url, timeout=30)
    response.raise_for_status()

    return pd.read_csv(io.StringIO(response.text))
