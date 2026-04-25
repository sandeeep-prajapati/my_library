# Skill: Add Chapters According To Syllabus

## Goal
Generate lesson files from a syllabus file and keep the writing style consistent with existing lessons in the same folder.

Primary target for this skill:
- Any `*/syllabus.md` file
- Output in the same folder as the syllabus
- Lesson files like `01.md`, `02.md`, ..., `NN.md`

---

## Behavior
When this skill is used, do the following in order:

1. Read the syllabus file and extract each numbered lesson item (`### 1.`, `### 2.`, etc.).
2. Detect the destination folder from the syllabus path.
3. Inspect existing lessons in that folder (at least first 1-2 files) to copy tone, formatting, and depth.
4. Create missing lesson files using zero-padded numbering (`01.md`, `02.md`, ...).
5. If a lesson file exists but is empty, fill it.
6. If a lesson file exists with content, do not overwrite unless explicitly asked.

---

## Writing Style Rules
Follow the same style as existing lesson files in the target folder (for example `01.md`, `02.md` if present):

- Conversational but technical.
- Deep internals first, then practical implementation.
- Use clear sections and progressive breakdown.
- Include real class/function/module names relevant to the target technology.
- Include code blocks where useful.
- Explain "why" and "how", not just definitions.
- Keep each lesson self-contained.

Use this lesson structure:

1. Short intro (what and why)
2. Core concept breakdown
3. Internal flow / architecture
4. Practical implementation examples
5. Common mistakes
6. Best practices for enterprise projects
7. Quick recap

---

## File Naming Rules
- Use two-digit numbers: `01.md` to `99.md`.
- Keep lesson files directly inside course folder unless user asks for subfolders.
- Syllabus stays as `syllabus.md`.

---

## Content Safety Rules
- Do not invent framework/library APIs that do not exist.
- Prefer stable, production-safe patterns.
- Do not suggest editing vendor code directly.
- Prefer extension points provided by the target framework/platform.

---

## Output Checklist
Before finishing, verify:

- All syllabus items have matching lesson files.
- No empty lesson file remains.
- Numbering and file names are correct.
- Style is consistent with existing lessons in the same folder.
- Existing non-empty user-written files were not overwritten unintentionally.
