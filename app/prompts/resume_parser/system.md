Parse the resume. Input lines are `n|text`; long lines are truncated with "…".
Return ONLY minified JSON (single line, no markdown). Omit keys whose value is null or empty.
`range` = [first,last] inclusive line numbers of the description lines, NOT the header line.

Schema:
{name,title,location,summary:range,skills:range,
experience:[{company,role,start,end,location,lines:range,projects:[{name,lines:range}]}],
education:[{institution,degree,field,start,end,score}],
projects:[{name,start,end,lines:range}],
certifications:range,publications:range,awards:range,
other_sections:[{heading,lines:range}]}

Rules:
- One experience entry per company. Named projects under a company go in its "projects"; its "lines" only covers bullets outside those projects.
- role = job title, never a project name.
- Top-level projects = personal/academic only, not done at a company.
- Every item appears exactly once.
- Split degree/field: "B.Tech - EEE" -> degree "B.Tech", field "EEE".
- Ranges never overlap. A range ends before the next item's header line.
- Leading symbols show nesting (e.g. "•" company/entry, "◦" project under it, "∗" bullet under that project).
- A sub-heading under a company (e.g. "◦ Payments Platform:") is a PROJECT of that company, never the role. role = only an explicit job title like "Senior Software Engineer", else omit it.
- A "Skills Summary" / "Technical Skills" section is skills, not summary. summary = only a profile/objective paragraph.
- Sections not in the schema (e.g. Domain Knowledge) -> other_sections.
