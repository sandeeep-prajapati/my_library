Here are all 15 prompts in plain text:

---

**1. Scaffold my entire project from scratch**
You are a senior engineer. Set up a production-ready [Next.js / Python FastAPI] project with the ideal folder structure, ESLint, Prettier, env config, and a README. Explain every architectural decision as you go.

---

**2. Turn me into a VS Code power user**
Suggest the best VS Code extensions, keybindings, and workspace settings for vibe coding in [language/stack]. Give me a settings.json and keybindings.json I can paste directly into VS Code.

---

**3. Build a reusable skill module**
Create a skill called [skill-name] that does [X]. It should be a self-contained module with a clear interface, typed input/output schema, proper error handling, and a usage example. Explain how to plug it into a larger system.

---

**4. Extract a skill from existing code**
Look at this code: [paste code]. Identify logic that can be extracted into a reusable, composable skill. Refactor it, give it a clear name, define its input/output contract, and show me a before/after comparison.

---

**5. Design an automated workflow end-to-end**
Design a workflow that takes [input] and produces [output]. Break it into steps, identify which parts can run in parallel, handle errors gracefully, and generate working code plus a Mermaid diagram of the full flow.

---

**6. Build a multi-step data pipeline**
Create a data pipeline that fetches from [source], transforms it with [logic], and outputs to [destination]. Add structured logging, automatic retries with backoff, and a --dry-run mode that previews changes without writing.

---

**7. Automate my repetitive dev tasks**
I keep doing [task] manually every time [trigger]. Write a script that automates this completely. Hook it into my package.json scripts or .vscode/tasks.json so I can run it with one command.

---

**8. Build an AI agent with tool use**
Build an AI agent that achieves [goal]. Give it these tools: [tool1, tool2, tool3]. Implement a ReAct loop — reason, act, observe, repeat. Show the full system prompt, tool definitions, and the core loop logic.

---

**9. Create a coding agent that writes & runs code**
Build a coding agent that takes a plain-English task, writes code to solve it, executes it in a sandboxed environment, checks the output, and self-corrects until it passes. Show the full think → code → run → verify → fix loop.

---

**10. Orchestrate multiple specialized agents**
I need a planner agent that breaks down [goal] and delegates to specialist sub-agents: [agent1], [agent2]. Show the orchestration logic, how they communicate (message schema), and how the planner merges results into a final output.

---

**11. Debug this like a 10x engineer**
Here's a bug: [describe or paste code]. Don't just fix it — reason through what is wrong, why it happens, what the correct fix is, and how to prevent this class of bug. Then add a regression test that would have caught this.

---

**12. Do a full code review on my PR**
Review this code: [paste]. Check for correctness, edge cases, performance bottlenecks, security issues, readability, and test coverage. Give specific line-level feedback with suggested rewrites, then a summary verdict.

---

**13. Write a skill that calls an external API**
Build a skill that wraps the [API name] API. Handle authentication, rate limiting, pagination, and all error states. Return typed results. Include a mock mode that returns fixture data for local testing without hitting the real API.

---

**14. Generate tests for my entire module**
Here's my module: [paste code]. Write a complete test suite using [Jest/pytest]. Include unit tests for every function, integration tests for the main happy path, and edge case tests for tricky inputs. Aim for 100% branch coverage.

---

**15. Build a self-improving prompt agent**
Create a meta-prompt agent: takes a task description, generates a prompt to solve it, evaluates output quality on a 1-10 rubric, identifies weaknesses, rewrites the prompt to address them, and loops until quality score exceeds a threshold. Show the full loop with scoring logic.