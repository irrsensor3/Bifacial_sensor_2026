# Setup after the security fix

## 1. Generate your admin password (required — admin login is disabled until you do)

Run this locally, with your own password in place of `YOUR-PASSWORD`:

```bash
python -c "import hashlib,secrets; s=secrets.token_hex(16); print('ADMIN_SALT =', repr(s)); print('ADMIN_PASSWORD_HASH =', repr(hashlib.pbkdf2_hmac('sha256', b'YOUR-PASSWORD', s.encode(), 200_000).hex()))"
```

It prints two lines. Paste them into `.streamlit/secrets.toml` alongside the
Supabase values:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
ADMIN_SALT = "…"
ADMIN_PASSWORD_HASH = "…"
```

On Streamlit Community Cloud, put the same four lines in the app's **Settings →
Secrets** box instead. Never commit `secrets.toml`.

If the two admin values are missing the app still runs — guests can use it
normally — and admin login shows a message telling you what to add.

## 2. Change the password itself

The old password was `admin123`, published in the repo. Assume it is known.
Pick a new one when you generate the hash above.

## 3. Rotate your Supabase key

`SUPABASE_KEY` was read from secrets, so it may never have been committed — but
check your commit history. If it ever appeared in a tracked file, rotate it in
the Supabase dashboard. Removing it in a later commit does not remove it from
history.

Also confirm it is the **anon** key, not `service_role`. The service role key
bypasses Row Level Security and must never reach a browser.

---

# What changed in ui_sections.py

## Security

| Before | After |
|---|---|
| Password `admin123` hardcoded | Read from `st.secrets` |
| Salt hardcoded in a public repo | Per-install, in secrets |
| Single SHA-256 pass | PBKDF2, 200,000 rounds |
| `==` comparison | `hmac.compare_digest` (constant time) |
| Unlimited guesses | 5 attempts, then a 5-minute lockout |

A single SHA-256 of a short password falls to a laptop in seconds. PBKDF2 at
200k rounds makes each guess cost real time, and the lockout stops the online
attack entirely.

## Accessibility

- **Contrast**: `--ink-muted` was `#6B7280` on `#F1F4F9` = 4.3:1, below the
  4.5:1 floor for body text. Now `#55606E`, which clears AA.
- **Focus rings**: the old rule used `:focus`, which fires on mouse clicks too,
  so the ring appeared constantly and people learned to ignore it. Now
  `:focus-visible`, amber, on every interactive control — visible against the
  blue UI.
- **Reduced motion**: honoured. This matters here because Live Monitoring
  auto-refreshes and can move the page under someone who did not ask it to.
- **Font fallback**: the theme imported Inter from Google Fonts and named
  `'Inter', sans-serif` everywhere. Where Google Fonts is blocked, the whole app
  fell back to the browser default. A full stack is now named.

## A silent bug in the theme

The old CSS forced **every** button to the same blue:

```css
.stButton > button { background-color: var(--blue); color: #FFFFFF; }
```

That cancelled Streamlit's own primary/secondary distinction, so
`type="primary"` had no visible effect anywhere in the app. Buttons are now
styled per variant: primary filled, secondary outlined. The difference survives
greyscale and colour blindness, because it is weight and border rather than hue.

## Efficiency

- **`get_forced_sensors()` was uncached** — one Supabase round trip on *every*
  rerun. With 24 toggle buttons across two pages, every click paid for one. Now
  cached for 3s and invalidated immediately on write, so toggles still feel
  instant.
- **Matplotlib figures were never closed.** Every rerun built new ones and left
  them open, leaking memory for as long as the app stayed up. Added
  `close_figures()` — call it after `st.pyplot()`.

## Robustness

- **Missing secrets** raised a bare `KeyError` at import, shown as a stack
  trace. Now a message naming the missing key and where to add it.
- **Every Supabase read** could throw and blank the page. All three fetchers now
  route through `_safe_query()`, which warns and returns empty.
- **`pd.to_datetime`** on alerts had no `errors="coerce"`; one malformed
  timestamp took the page down.

## Copy

Button labels now say what happens: "Sign in as admin", "Continue as guest".
Errors say what to do rather than only what failed.

---

# Changes in the pages

## Admin_Controls.py

- **Reboot and shutdown now need confirmation.** They fired on a single click,
  so one mis-click took the logger offline with no way to cancel. Now two steps
  with an explicit warning about what each does.
- **Both commands are wrapped in try/except.** A failed write used to raise a
  stack trace over the page, leaving you unsure whether the command was sent.
- **"Test Supabase" wrote `command="hello"` into `pi_commands`** — the same row
  the Pi polls for reboot and shutdown. Writing junk into a live command channel
  to check connectivity is asking for trouble. It now reads `pi_settings`
  instead, and tells you if the `id = 1` row is missing (which would silently
  break every sensor toggle).
- Sensor toggles: state is in the label (`5 · ON`) and the button variant, not
  a red or green dot. Red for "normal" was backwards anyway — red reads as a
  fault.

## app.py

- **`user_role.capitalize()` crashed on `None`**, which happens with a stale tab
  open across a redeploy.
- **A missing `BifacialGrid.jpeg` took the whole app down**, on every page,
  because the image renders before navigation. Now caught.
- **The overview strip is collapsible** and collapsed on Live Monitoring, which
  already shows the same data in more detail. It costs two Supabase reads per
  page load and pushed each page's real content below the fold.
- **Gauge values are also stated as text.** A Plotly gauge is unreadable to a
  screen reader and hard to read a number off.
- **Two bare `except Exception` blocks removed.** Their fallbacks could never
  run, and they turned real errors into a plausible-looking zero.
- Pages reordered: Live Monitoring first, since that is what people open on
  arrival. The first entry is also the landing page.

## Live_Monitoring.py

- **`f"{val:.1f}"` raised on any non-numeric reading** and printed `nan` for a
  missing one. Now coerced, with "no reading" shown instead.
- **A tz-naive `created_at` raised `TypeError`** when subtracted from a
  tz-aware "now", blanking the sub-zero section. Normalised to UTC.
- **`created_at.strftime()`** assumed a parsed datetime; same crash as on the
  Panel Array page.
- **Auto-refresh is now opt-out.** It reruns the whole script every 5–60s,
  refetching from Supabase and resetting scroll — which fights anyone reading a
  chart or filling in the append controls. Off means a "Refresh now" button.
- Meter status says "OK" or "Fault" rather than relying on a coloured dot.

## Data_and_Reports.py

- **Both reports were generated on every rerun**, before either download button
  was clicked. Moving a slider silently rendered a Word document *and* a PDF,
  embedding matplotlib images into each. Now you pick a format and press
  "Build report".
- Report building is wrapped in try/except with a spinner.
- **Figures released** via `close_figures()`.
- Row and column counts shown as metrics rather than two lines of prose.

## Irradiance_Tracker.py

- **`index=len(months) - 1` raised `IndexError`** when no file had a
  recognisable month. Both the year and month lists are now checked.
- **Figures released** — this page builds one per sensor, so with 24 sensors it
  leaked the fastest.
- Charts are laid out two per row. Twenty-four stacked single-file meant a very
  long scroll and no way to compare distributions.

## requirements.txt

Every package was unpinned, so a new Streamlit release could break the deployed
app without anyone touching the code. Now pinned to major versions.

---

# Not done

- **Live_Monitoring.py is 381 lines** and does five separate jobs. It works, but
  it is the file most likely to break next. Worth splitting.
- **No tests.** The crashes fixed above would all have been caught by a handful
  of unit tests on the formatting and timestamp helpers.
- **The two data paths still disagree.** Live Monitoring and Panel Array read
  Supabase; Data & Reports and Irradiance Tracker read CSVs from Drive. Nothing
  reconciles them.
- **None of this is tested against a running app** — there is no Streamlit or
  Supabase in my environment. Every file parses and no import is broken, but
  please click through each page after deploying.
