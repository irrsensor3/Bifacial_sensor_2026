start_date = earliest
    if not start_date or not end_date or start_date > end_date:
        st.caption("Pick a valid range.")
        return st.session_state.get(f"_{key_prefix}_df"), st.session_state.get(f"_{key_prefix}_label")

    period_files = resolve_period_files(
        available_files,
        start_date,
        end_date,
    )
