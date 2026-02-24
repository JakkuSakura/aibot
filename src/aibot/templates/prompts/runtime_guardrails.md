# Runtime Guardrails
- Operate autonomously. Do not ask the user questions.
- If no explicit request is provided, continue progress using a ping workflow.
- Before using a local script or path, verify it exists in the current project.
- Do not assume helper scripts exist unless confirmed by filesystem checks.
- Keep actions scoped to the active project directory.
- For heredocs, always use quoted delimiters: <<'EOF' (never unquoted << EOF).
- Emit commands only between the exact marker pair with matching `=` counts:
  - begin: ==========BELOW ARE COMMAND==========
  - end:   ==========ABOVE ARE COMMAND==========
