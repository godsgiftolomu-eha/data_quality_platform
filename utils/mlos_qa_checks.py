import pandas as pd
import uuid
import re


def _find_col(df, candidates):
    """Find a column name case-insensitively from a list of candidates."""
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        if name.lower() in df_cols_lower:
            return df_cols_lower[name.lower()]
    return None


def _col(df, candidates):
    """Return the series for a column found by candidate names, or None."""
    col_name = _find_col(df, candidates)
    if col_name is not None:
        return df[col_name]
    return None


# ── Individual checks ─────────────────────────────────────────────────────


def check_wards_have_takeoffpoint(mlos_df, tp_df):
    """Rule 1: All wards must have a takeoffpoint."""
    issues = []
    ward_col_mlos = _find_col(mlos_df, ["ward_code", "wardcode", "ward code"])
    ward_col_tp = _find_col(tp_df, ["wardcode", "ward_code", "ward code"])

    if ward_col_mlos is None:
        issues.append({"rule": 1, "severity": "ERROR",
                        "message": "MLoS table missing 'ward_code' column"})
        return issues
    if ward_col_tp is None:
        issues.append({"rule": 1, "severity": "ERROR",
                        "message": "Takeoffpoint table missing 'wardcode' column"})
        return issues

    mlos_wards = set(mlos_df[ward_col_mlos].dropna().astype(str).str.strip().unique())
    tp_wards = set(tp_df[ward_col_tp].dropna().astype(str).str.strip().unique())
    missing = mlos_wards - tp_wards

    if missing:
        issues.append({
            "rule": 1, "severity": "ERROR",
            "message": f"{len(missing)} ward(s) in MLoS have no takeoffpoint: {', '.join(sorted(missing))}",
        })
    return issues


def check_takeoffpoint_name_match(mlos_df, tp_df):
    """Rule 2: takeoffpoint in MLoS = name in takeoffpoint table."""
    issues = []
    tp_mlos = _find_col(mlos_df, ["takeoffpoint", "take_off_point", "takeoff_point"])
    tp_name = _find_col(tp_df, ["name", "takeoffpoint_name", "tp_name"])

    if tp_mlos is None:
        issues.append({"rule": 2, "severity": "ERROR",
                        "message": "MLoS table missing 'takeoffpoint' column"})
        return issues
    if tp_name is None:
        issues.append({"rule": 2, "severity": "ERROR",
                        "message": "Takeoffpoint table missing 'name' column"})
        return issues

    valid_names = set(tp_df[tp_name].dropna().astype(str).str.strip().unique())
    mlos_vals = mlos_df[tp_mlos].astype(str).str.strip()
    bad_mask = ~mlos_vals.isin(valid_names) & mlos_vals.notna() & (mlos_vals != "nan")
    bad_rows = mlos_df.index[bad_mask].tolist()

    if bad_rows:
        bad_vals = mlos_vals[bad_mask].unique().tolist()
        issues.append({
            "rule": 2, "severity": "ERROR",
            "message": f"{len(bad_rows)} row(s) have takeoffpoint names not found in takeoffpoint table",
            "rows": bad_rows,
            "invalid_values": bad_vals[:20],
        })
    return issues


def check_takeoffpoint_code_match(mlos_df, tp_df):
    """Rule 3: takeoffpoint_code in MLoS = code in takeoffpoint table."""
    issues = []
    code_mlos = _find_col(mlos_df, ["takeoffpoint_code", "tp_code", "takeoff_point_code"])
    code_tp = _find_col(tp_df, ["code", "takeoffpoint_code", "tp_code"])

    if code_mlos is None:
        issues.append({"rule": 3, "severity": "ERROR",
                        "message": "MLoS table missing 'takeoffpoint_code' column"})
        return issues
    if code_tp is None:
        issues.append({"rule": 3, "severity": "ERROR",
                        "message": "Takeoffpoint table missing 'code' column"})
        return issues

    valid_codes = set(tp_df[code_tp].dropna().astype(str).str.strip().unique())
    mlos_vals = mlos_df[code_mlos].astype(str).str.strip()
    bad_mask = ~mlos_vals.isin(valid_codes) & mlos_vals.notna() & (mlos_vals != "nan")
    bad_rows = mlos_df.index[bad_mask].tolist()

    if bad_rows:
        bad_vals = mlos_vals[bad_mask].unique().tolist()
        issues.append({
            "rule": 3, "severity": "ERROR",
            "message": f"{len(bad_rows)} row(s) have takeoffpoint_code not found in takeoffpoint table",
            "rows": bad_rows,
            "invalid_values": bad_vals[:20],
        })
    return issues


def check_ward_code_match(mlos_df, tp_df):
    """Rule 4: ward_code in MLoS = wardcode in takeoffpoint table."""
    issues = []
    ward_mlos = _find_col(mlos_df, ["ward_code", "wardcode", "ward code"])
    ward_tp = _find_col(tp_df, ["wardcode", "ward_code", "ward code"])

    if ward_mlos is None:
        issues.append({"rule": 4, "severity": "ERROR",
                        "message": "MLoS table missing 'ward_code' column"})
        return issues
    if ward_tp is None:
        issues.append({"rule": 4, "severity": "ERROR",
                        "message": "Takeoffpoint table missing 'wardcode' column"})
        return issues

    valid_wards = set(tp_df[ward_tp].dropna().astype(str).str.strip().unique())
    mlos_vals = mlos_df[ward_mlos].astype(str).str.strip()
    bad_mask = ~mlos_vals.isin(valid_wards) & mlos_vals.notna() & (mlos_vals != "nan")
    bad_rows = mlos_df.index[bad_mask].tolist()

    if bad_rows:
        bad_vals = mlos_vals[bad_mask].unique().tolist()
        issues.append({
            "rule": 4, "severity": "ERROR",
            "message": f"{len(bad_rows)} row(s) have ward_code not found in takeoffpoint table",
            "rows": bad_rows,
            "invalid_values": bad_vals[:20],
        })
    return issues


def check_security_compromised(mlos_df):
    """Rule 5: security_compromised must be Y or N, no nulls."""
    issues = []
    col = _find_col(mlos_df, ["security_compromised", "security compromised", "securitycompromised"])
    if col is None:
        issues.append({"rule": 5, "severity": "ERROR",
                        "message": "Missing 'security_compromised' column"})
        return issues

    series = mlos_df[col]
    nulls = series.isna().sum()
    if nulls:
        issues.append({"rule": 5, "severity": "ERROR",
                        "message": f"{nulls} null value(s) in security_compromised",
                        "rows": mlos_df.index[series.isna()].tolist()})

    vals = series.dropna().astype(str).str.strip().str.upper()
    bad_mask = ~vals.isin(["Y", "N"])
    bad_rows = vals.index[bad_mask].tolist()
    if bad_rows:
        issues.append({
            "rule": 5, "severity": "ERROR",
            "message": f"{len(bad_rows)} row(s) have invalid security_compromised (must be Y or N)",
            "rows": bad_rows,
            "invalid_values": vals[bad_mask].unique().tolist()[:20],
        })
    return issues


def check_accessibility_status(mlos_df):
    """Rule 6 & 7: accessibility_status must be valid; Partially Accessible / Inaccessible must have reason."""
    issues = []
    col = _find_col(mlos_df, ["accessibility_status", "accessibility status", "accessibilitystatus"])
    reason_col = _find_col(mlos_df, [
        "reason", "accessibility_reason", "inaccessibility_reason",
        "inaccessible_reason", "reason_for_inaccessibility",
    ])

    if col is None:
        issues.append({"rule": 6, "severity": "ERROR",
                        "message": "Missing 'accessibility_status' column"})
        return issues

    valid = {"fully accessible", "partially accessible", "inaccessible"}
    vals = mlos_df[col].astype(str).str.strip().str.lower()
    bad_mask = ~vals.isin(valid) & (vals != "nan")
    bad_rows = vals.index[bad_mask].tolist()

    if bad_rows:
        issues.append({
            "rule": 6, "severity": "ERROR",
            "message": f"{len(bad_rows)} row(s) have invalid accessibility_status "
                       f"(must be Fully Accessible, Partially Accessible, or Inaccessible)",
            "rows": bad_rows,
            "invalid_values": mlos_df[col].iloc[bad_rows].unique().tolist()[:20],
        })

    null_status = mlos_df[col].isna().sum()
    if null_status:
        issues.append({"rule": 6, "severity": "ERROR",
                        "message": f"{null_status} null value(s) in accessibility_status",
                        "rows": mlos_df.index[mlos_df[col].isna()].tolist()})

    # Rule 7: reason required for Partially Accessible / Inaccessible
    needs_reason = vals.isin(["partially accessible", "inaccessible"])
    if needs_reason.any():
        if reason_col is None:
            issues.append({"rule": 7, "severity": "ERROR",
                            "message": "No 'reason' column found but Partially Accessible / Inaccessible rows exist"})
        else:
            reason_vals = mlos_df[reason_col]
            missing_reason = needs_reason & (reason_vals.isna() | (reason_vals.astype(str).str.strip() == ""))
            bad_rows = mlos_df.index[missing_reason].tolist()
            if bad_rows:
                issues.append({
                    "rule": 7, "severity": "ERROR",
                    "message": f"{len(bad_rows)} row(s) are Partially Accessible / Inaccessible but have no reason",
                    "rows": bad_rows,
                })
    return issues


def check_habitation_status(mlos_df):
    """Rule 8: habitation_status must be Abandoned, Migrated, Inhabited, or Partially Inhabited."""
    issues = []
    col = _find_col(mlos_df, ["habitation_status", "habitation status", "habitationstatus"])
    if col is None:
        issues.append({"rule": 8, "severity": "ERROR",
                        "message": "Missing 'habitation_status' column"})
        return issues

    valid = {"abandoned", "migrated", "inhabited", "partially inhabited"}
    vals = mlos_df[col].astype(str).str.strip().str.lower()
    bad_mask = ~vals.isin(valid) & (vals != "nan")
    bad_rows = vals.index[bad_mask].tolist()

    if bad_rows:
        issues.append({
            "rule": 8, "severity": "ERROR",
            "message": f"{len(bad_rows)} row(s) have invalid habitation_status",
            "rows": bad_rows,
            "invalid_values": mlos_df[col].iloc[bad_rows].unique().tolist()[:20],
        })

    null_count = mlos_df[col].isna().sum()
    if null_count:
        issues.append({"rule": 8, "severity": "ERROR",
                        "message": f"{null_count} null value(s) in habitation_status",
                        "rows": mlos_df.index[mlos_df[col].isna()].tolist()})
    return issues


def check_population_not_null(mlos_df):
    """Rule 9: set_pop, target_pop, no_of_houses, noncpliant must not be null."""
    issues = []
    pop_fields = {
        "set_pop": ["set_pop", "setpop", "set pop", "settlement_population"],
        "target_pop": ["target_pop", "targetpop", "target pop", "target_population"],
        "no_of_houses": ["no_of_houses", "noofhouses", "no of houses", "num_houses"],
        "noncpliant": ["noncpliant", "noncompliant", "non_compliant", "non compliant"],
    }

    for field_label, candidates in pop_fields.items():
        col = _find_col(mlos_df, candidates)
        if col is None:
            issues.append({"rule": 9, "severity": "ERROR",
                            "message": f"Missing '{field_label}' column"})
            continue

        nulls = mlos_df[col].isna().sum()
        if nulls:
            issues.append({
                "rule": 9, "severity": "ERROR",
                "message": f"{nulls} null value(s) in {field_label}",
                "rows": mlos_df.index[mlos_df[col].isna()].tolist(),
            })
    return issues


def check_team_code(mlos_df):
    """Rule 10: team_code must be a number or NA."""
    issues = []
    col = _find_col(mlos_df, ["team_code", "teamcode", "team code"])
    if col is None:
        issues.append({"rule": 10, "severity": "ERROR",
                        "message": "Missing 'team_code' column"})
        return issues

    vals = mlos_df[col].astype(str).str.strip().str.upper()
    # Valid: numeric value or "NA"
    valid_mask = vals.str.match(r"^\d+(\.\d+)?$") | (vals == "NA") | (vals == "NAN")
    bad_rows = mlos_df.index[~valid_mask].tolist()

    if bad_rows:
        issues.append({
            "rule": 10, "severity": "ERROR",
            "message": f"{len(bad_rows)} row(s) have invalid team_code (must be a number or NA)",
            "rows": bad_rows,
            "invalid_values": vals[~valid_mask].unique().tolist()[:20],
        })
    return issues


def check_day_of_activity(mlos_df):
    """Rule 11: day_of_activity must be one of the valid day combinations or NA."""
    issues = []
    col = _find_col(mlos_df, ["day_of_activity", "dayofactivity", "day of activity",
                               "day_activity", "dayactivity"])
    if col is None:
        issues.append({"rule": 11, "severity": "ERROR",
                        "message": "Missing 'day_of_activity' column"})
        return issues

    valid = {
        "1", "1_2", "1_2_3", "1_2_3_4",
        "2", "2_3", "2_3_4",
        "3", "3_4",
        "4",
        "NA",
    }

    vals = mlos_df[col].astype(str).str.strip().str.upper()
    bad_mask = ~vals.isin(valid) & (vals != "NAN")
    bad_rows = mlos_df.index[bad_mask].tolist()

    if bad_rows:
        issues.append({
            "rule": 11, "severity": "ERROR",
            "message": f"{len(bad_rows)} row(s) have invalid day_of_activity",
            "rows": bad_rows,
            "invalid_values": vals[bad_mask].unique().tolist()[:20],
        })
    return issues


def check_urban_rural_scattered(mlos_df):
    """Rule 12: urban, rural, scattered = Y or N (no null).
    A settlement can't be both urban and rural.
    Urban can't be scattered.
    """
    issues = []
    urban_col = _find_col(mlos_df, ["urban"])
    rural_col = _find_col(mlos_df, ["rural"])
    scattered_col = _find_col(mlos_df, ["scattered"])

    for label, col_name in [("urban", urban_col), ("rural", rural_col), ("scattered", scattered_col)]:
        if col_name is None:
            issues.append({"rule": 12, "severity": "ERROR",
                            "message": f"Missing '{label}' column"})
            continue

        series = mlos_df[col_name]
        nulls = series.isna().sum()
        if nulls:
            issues.append({
                "rule": 12, "severity": "ERROR",
                "message": f"{nulls} null value(s) in {label}",
                "rows": mlos_df.index[series.isna()].tolist(),
            })

        vals = series.dropna().astype(str).str.strip().str.upper()
        bad_mask = ~vals.isin(["Y", "N"])
        bad_rows = vals.index[bad_mask].tolist()
        if bad_rows:
            issues.append({
                "rule": 12, "severity": "ERROR",
                "message": f"{len(bad_rows)} row(s) have invalid '{label}' value (must be Y or N)",
                "rows": bad_rows,
            })

    # Rule 12a: Can't be both urban=Y and rural=Y
    if urban_col and rural_col:
        u = mlos_df[urban_col].astype(str).str.strip().str.upper()
        r = mlos_df[rural_col].astype(str).str.strip().str.upper()
        both = (u == "Y") & (r == "Y")
        if both.any():
            issues.append({
                "rule": "12a", "severity": "ERROR",
                "message": f"{both.sum()} row(s) are both urban=Y and rural=Y (invalid)",
                "rows": mlos_df.index[both].tolist(),
            })

    # Rule 12b: Urban can't be scattered
    if urban_col and scattered_col:
        u = mlos_df[urban_col].astype(str).str.strip().str.upper()
        s = mlos_df[scattered_col].astype(str).str.strip().str.upper()
        urban_scattered = (u == "Y") & (s == "Y")
        if urban_scattered.any():
            issues.append({
                "rule": "12b", "severity": "ERROR",
                "message": f"{urban_scattered.sum()} row(s) are urban=Y and scattered=Y (urban can't be scattered)",
                "rows": mlos_df.index[urban_scattered].tolist(),
            })
    return issues


def check_profile_fields(mlos_df):
    """Rule 13: Remaining profile columns must be Y, N, or NA."""
    issues = []
    # These are the columns handled by other specific rules - exclude them
    known_cols_lower = {
        "takeoffpoint", "take_off_point", "takeoff_point",
        "takeoffpoint_code", "tp_code", "takeoff_point_code",
        "ward_code", "wardcode",
        "security_compromised", "securitycompromised",
        "accessibility_status", "accessibilitystatus",
        "reason", "accessibility_reason", "inaccessibility_reason",
        "inaccessible_reason", "reason_for_inaccessibility",
        "habitation_status", "habitationstatus",
        "set_pop", "setpop", "settlement_population",
        "target_pop", "targetpop", "target_population",
        "no_of_houses", "noofhouses", "num_houses",
        "noncpliant", "noncompliant", "non_compliant",
        "team_code", "teamcode",
        "day_of_activity", "dayofactivity", "day_activity",
        "urban", "rural", "scattered",
        "source", "editer", "editor",
        "globalid", "global_id",
        "fc_globalid", "fc_global_id",
        "settlementarea_globlid", "settlementarea_globalid", "settlement_area_globalid",
        # Common non-profile columns
        "objectid", "settlement", "settlement_name", "settlementname",
        "state", "state_code", "statecode",
        "lga", "lga_code", "lgacode",
        "ward", "ward_name", "wardname",
        "latitude", "longitude", "lat", "lon", "x", "y",
        "shape", "geom", "geometry",
    }

    profile_cols = []
    for c in mlos_df.columns:
        if c.lower().strip() not in known_cols_lower:
            # Check if the column values look like a Y/N profile column
            vals = mlos_df[c].dropna().astype(str).str.strip().str.upper()
            unique_vals = set(vals.unique())
            # If most values are Y/N/NA, treat as profile column
            if unique_vals and unique_vals.issubset({"Y", "N", "NA", "NAN", ""}):
                profile_cols.append(c)
            elif len(unique_vals) > 0 and len(unique_vals - {"Y", "N", "NA", "NAN", ""}) <= len(unique_vals) * 0.1:
                profile_cols.append(c)

    for col in profile_cols:
        vals = mlos_df[col].astype(str).str.strip().str.upper()
        valid_mask = vals.isin(["Y", "N", "NA", "NAN"])
        bad_rows = mlos_df.index[~valid_mask].tolist()
        if bad_rows:
            issues.append({
                "rule": 13, "severity": "WARNING",
                "message": f"Profile column '{col}': {len(bad_rows)} row(s) have values other than Y, N, or NA",
                "rows": bad_rows,
                "invalid_values": vals[~valid_mask].unique().tolist()[:20],
            })
    return issues


def check_source(mlos_df):
    """Rule 14: source must be 'MLoS'."""
    issues = []
    col = _find_col(mlos_df, ["source"])
    if col is None:
        issues.append({"rule": 14, "severity": "ERROR",
                        "message": "Missing 'source' column"})
        return issues

    vals = mlos_df[col].astype(str).str.strip()
    bad_mask = vals.str.lower() != "mlos"
    bad_rows = mlos_df.index[bad_mask].tolist()

    if bad_rows:
        issues.append({
            "rule": 14, "severity": "ERROR",
            "message": f"{len(bad_rows)} row(s) have source != 'MLoS'",
            "rows": bad_rows,
            "invalid_values": vals[bad_mask].unique().tolist()[:20],
        })
    return issues


def check_editer(mlos_df):
    """Rule 15: editer must be in firstname.surname format, all lowercase."""
    issues = []
    col = _find_col(mlos_df, ["editer", "editor"])
    if col is None:
        issues.append({"rule": 15, "severity": "ERROR",
                        "message": "Missing 'editer' column"})
        return issues

    pattern = re.compile(r"^[a-z]+\.[a-z]+$")
    vals = mlos_df[col].astype(str).str.strip()
    bad_mask = ~vals.apply(lambda v: bool(pattern.match(v)))
    bad_rows = mlos_df.index[bad_mask].tolist()

    if bad_rows:
        issues.append({
            "rule": 15, "severity": "ERROR",
            "message": f"{len(bad_rows)} row(s) have invalid editer format (must be firstname.surname, all lowercase)",
            "rows": bad_rows,
            "invalid_values": vals[bad_mask].unique().tolist()[:20],
        })
    return issues


def _is_valid_uuid(val):
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, AttributeError):
        return False


def check_globalid(mlos_df):
    """Rule 16: globalid must be a valid UUID."""
    issues = []
    col = _find_col(mlos_df, ["globalid", "global_id"])
    if col is None:
        issues.append({"rule": 16, "severity": "ERROR",
                        "message": "Missing 'globalid' column"})
        return issues

    vals = mlos_df[col].astype(str).str.strip()
    bad_mask = ~vals.apply(_is_valid_uuid)
    bad_rows = mlos_df.index[bad_mask].tolist()

    if bad_rows:
        issues.append({
            "rule": 16, "severity": "ERROR",
            "message": f"{len(bad_rows)} row(s) have invalid globalid (must be a valid UUID)",
            "rows": bad_rows,
        })
    return issues


def check_fc_globalid(mlos_df):
    """Rule 17: fc_globalid must be a valid UUID (generate new if missing)."""
    issues = []
    col = _find_col(mlos_df, ["fc_globalid", "fc_global_id"])
    if col is None:
        issues.append({"rule": 17, "severity": "ERROR",
                        "message": "Missing 'fc_globalid' column"})
        return issues

    vals = mlos_df[col].astype(str).str.strip()
    bad_mask = ~vals.apply(_is_valid_uuid)
    bad_rows = mlos_df.index[bad_mask].tolist()

    if bad_rows:
        issues.append({
            "rule": 17, "severity": "WARNING",
            "message": f"{len(bad_rows)} row(s) have invalid/missing fc_globalid (will generate new UUIDs on fix)",
            "rows": bad_rows,
        })
    return issues


def check_settlementarea_globalid(mlos_df):
    """Rule 18: settlementarea_globlid must be a valid UUID (generate new if missing). Field planned for removal."""
    issues = []
    col = _find_col(mlos_df, ["settlementarea_globlid", "settlementarea_globalid",
                               "settlement_area_globalid"])
    if col is None:
        issues.append({"rule": 18, "severity": "INFO",
                        "message": "Missing 'settlementarea_globlid' column (planned for removal, not critical)"})
        return issues

    vals = mlos_df[col].astype(str).str.strip()
    bad_mask = ~vals.apply(_is_valid_uuid)
    bad_rows = mlos_df.index[bad_mask].tolist()

    if bad_rows:
        issues.append({
            "rule": 18, "severity": "WARNING",
            "message": f"{len(bad_rows)} row(s) have invalid/missing settlementarea_globlid "
                       f"(will generate new UUIDs on fix). Note: this field is planned for removal.",
            "rows": bad_rows,
        })
    return issues


# ── Auto-fix helpers ──────────────────────────────────────────────────────


def fix_generate_uuids(df, col_candidates):
    """Generate new UUIDs for rows with missing/invalid UUIDs."""
    col = _find_col(df, col_candidates)
    if col is None:
        return df
    df = df.copy()
    vals = df[col].astype(str).str.strip()
    bad_mask = ~vals.apply(_is_valid_uuid)
    for idx in df.index[bad_mask]:
        df.at[idx, col] = str(uuid.uuid4())
    return df


# ── Master runner ─────────────────────────────────────────────────────────


def run_all_mlos_checks(mlos_df, tp_df=None):
    """Run all MLoS QA checks. Returns a list of issue dicts.

    Each issue has keys: rule, severity, message, and optionally rows, invalid_values.
    """
    all_issues = []

    # Cross-table checks (rules 1-4) require takeoffpoint table
    if tp_df is not None:
        all_issues.extend(check_wards_have_takeoffpoint(mlos_df, tp_df))
        all_issues.extend(check_takeoffpoint_name_match(mlos_df, tp_df))
        all_issues.extend(check_takeoffpoint_code_match(mlos_df, tp_df))
        all_issues.extend(check_ward_code_match(mlos_df, tp_df))
    else:
        all_issues.append({
            "rule": "1-4", "severity": "SKIPPED",
            "message": "Cross-table checks skipped: no takeoffpoint table uploaded",
        })

    # Single-table checks (rules 5-18)
    all_issues.extend(check_security_compromised(mlos_df))
    all_issues.extend(check_accessibility_status(mlos_df))
    all_issues.extend(check_habitation_status(mlos_df))
    all_issues.extend(check_population_not_null(mlos_df))
    all_issues.extend(check_team_code(mlos_df))
    all_issues.extend(check_day_of_activity(mlos_df))
    all_issues.extend(check_urban_rural_scattered(mlos_df))
    all_issues.extend(check_profile_fields(mlos_df))
    all_issues.extend(check_source(mlos_df))
    all_issues.extend(check_editer(mlos_df))
    all_issues.extend(check_globalid(mlos_df))
    all_issues.extend(check_fc_globalid(mlos_df))
    all_issues.extend(check_settlementarea_globalid(mlos_df))

    return all_issues


def issues_to_dataframe(issues):
    """Convert issues list to a summary DataFrame for display."""
    if not issues:
        return pd.DataFrame(columns=["Rule", "Severity", "Message", "Affected Rows"])
    rows = []
    for issue in issues:
        rows.append({
            "Rule": str(issue.get("rule", "")),
            "Severity": issue.get("severity", ""),
            "Message": issue.get("message", ""),
            "Affected Rows": len(issue.get("rows", [])) if "rows" in issue else "-",
        })
    return pd.DataFrame(rows)
