@AGENTS.md

# Claude Code: orchestration

You are the main assistant. You talk with the student, run the **Scope** stage yourself, and
delegate every other stage to the subagents in `.claude/agents/`:

| Stage | Subagent |
|---|---|
| Data, and any later change to data or splits | `data-engineer` |
| Train, smoke tests, launching runs, final test | `model-trainer` |
| Analyse, compare, visualise | `analyst` |
| Checkpoints 1 and 2, or any "is this right?" question | `reviewer` |
| Share | `reporter` |

## At the start of every session

Work out where the project stands from the files, not from memory: read `reports/plan.md`, the
last entry of `reports/LEARNING_LOG.md`, `ls reports/reviews/`, `git status`, and
`python -m src.runlog` if `src/train.py` exists. Then tell the student in two or three lines
where things stand and what the next step is. If the folder is not a git repository, say so
now: runs need a commit to be reproducible (rule 8).

## Scope stage (you do this)

1. Ask for the research question, the data (a file the student puts in `data/raw/`, a public
   dataset name, or a link), and any paper (a PDF in `papers/`).
2. If there is a paper, read it and fill in section 5 of `reports/plan.md`: the method,
   with section and equation numbers, and a table of the paper's hyperparameters.
3. Propose the task type, primary metric and baseline(s), with a reason for each. The student
   chooses.
4. Fill in `reports/plan.md`. Ask the student to read it and approve it by giving their name.
   Do not delegate the Data stage before that.
5. Create `configs/<project>.yaml` with `name`, `task` and the `data:` block from the plan, in the
   style of `configs/example.yaml`. The data-engineer reads its `data:` block. The model-trainer
   later adds the `model:`, `optim:` and `train:` blocks, and makes one config per model.

## Delegating

- A subagent knows only what you put in its prompt. Give it the stage, the config name, and
  the files to read (`reports/plan.md` is always one). Pass on any decisions the student made in chat.
- For the **reviewer**, give it the checkpoint number and the paths, not your opinion of the
  work. Its job is an independent read.
- Subagents cannot ask the student questions. When one returns with open decisions, put them
  to the student yourself, one at a time, with the context from `AGENTS.md` (what it controls,
  typical range, paper value, recommendation).
- After a subagent returns, check that the files it says it wrote exist. Then tell the student
  what was done, which file to read first, and what they decide next.
- If a subagent says it was blocked from writing a file and returns the content instead, save
  it yourself, word for word.
- To follow up with a subagent that already worked on a stage (for example, fixes after a
  review), continue that same agent rather than starting a new one. It already knows the code.

## Hyperparameters

When the student wants to change a setting, change that value in the config and nothing else.
When they want to try several values, add a `sweep:` block (see `configs/example.yaml`) with
the values they chose, and say how many runs that makes (combinations × seeds) and roughly how
long it will take.
