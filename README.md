# AI research assistant for PyTorch + PyG projects

You bring the **research question, the data and the decisions**. A small team of AI agents
writes, runs and explains the code. You learn PyTorch and PyTorch Geometric by reading what
they build, choosing the hyperparameters, and judging the results.

```
 Scope ──► Data ──► ✓1 ──► Train ──► Analyse ──► ✓2 ──► Final test ──► Share
 you+AI    data-    re-    model-    analyst     re-    model-         reporter
           engineer viewer trainer               viewer trainer
```

| Agent | What it does |
|---|---|
| **data-engineer** | Cleans your data, builds PyTorch/PyG datasets and loaders, and makes fixed train/val/test splits |
| **model-trainer** | Builds a baseline and your main model (from your paper, if you give one), trains them, and runs the final test |
| **analyst** | Compares models across seeds and makes figures and the results write-up |
| **reviewer** | Independent check at ✓1 (data, leakage) and ✓2 (evaluation, claims). Changes nothing; reports problems |
| **reporter** | Packages everything so others can reproduce it: README, one-command script, report draft, model card |

## Your job

1. **Ask the question.** Say what you want to find out. The AI helps you turn it into `reports/plan.md`; you approve it.
2. **Bring the data and context.** Put files in `data/raw/` (or name a public dataset). Put papers to follow in `papers/`.
3. **Choose the hyperparameters.** Each experiment has a file in `configs/` where every setting is explained. You pick the values. The agents never change them without asking.
4. **Launch runs** on Colab or CCAST when the agents tell you a run is ready.
5. **Read `reports/LEARNING_LOG.md`** after each stage, with the code open. Ask about anything unclear.
6. **Decide** what to try next, when the final test runs, and what gets shared.

You never have to write code. Reading it is how you learn.

## Rules the agents follow (call them out if they don't)

- The **test set stays locked** until Checkpoint 2 passes, and the final test runs **once**.
- A **simple baseline** runs before the main model.
- Every comparison uses **≥ 3 seeds**, reported as mean ± std.
- **Your config values** are changed only when you say so.
- **No run is deleted**; results say how many things were tried.

---

## Getting started

1. Make your own copy: on GitHub, **Use this template → Create a new repository**, then clone it.
   If you copy the folder instead, run `git init` in it. Every run records the code version
   (git commit), and the agents will ask you to commit before real runs and before the final test.
2. Install Python 3.11+, then PyTorch for your machine ([pytorch.org](https://pytorch.org/get-started/locally/)), then:
   ```bash
   python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Open the project in your AI tool (below) and say something like:
   > *Let's start. My question is … My data is in data/raw/… I'd like to follow papers/smith2024.pdf.*

### Claude Code (full multi-agent setup)

```bash
cd my-project
claude
```
Start Claude **inside the project folder**. That's how it finds `CLAUDE.md` and the agents in
`.claude/agents/`. It works through the stages and calls each agent itself. To read PDFs in
`papers/`, Claude Code needs poppler: `brew install poppler` (macOS) or
`sudo apt install poppler-utils` (Linux/WSL).

**What to expect:** in a trial on Cora (small data, laptop CPU), the whole pipeline, from plan to
packaged report, took about an hour of agent time, with a handful of decisions from the student.
Training itself took under a minute. Larger data or more experiments take longer, and long sessions
can run into your Claude plan's usage limits, so work one stage per sitting if needed. A
complete worked example is in the course materials (`trial-cora`).

### Other AI coding tools (Cursor, GitHub Copilot agent mode, OpenAI Codex, Gemini CLI, …)

Most of these read `AGENTS.md` automatically. If yours doesn't, start each chat with
*"Read AGENTS.md first."* Then work **one stage per chat** and name the role:
> *Act as the data-engineer described in .claude/agents/data-engineer.md and do the Data stage.*

For checkpoints, **open a new chat** so the reviewer isn't checking its own work:
> *Act as the reviewer in .claude/agents/reviewer.md and run Checkpoint 1.*

If your tool can't read PDFs, paste the relevant sections of the paper.

### Chat-only assistants (ChatGPT, Claude.ai, Gemini on the web)

These work but take more copying. Each chat, upload `AGENTS.md`, the role file,
`reports/plan.md` and the files the stage needs. Then save the code it gives you into the right
files and run the commands yourself.

---

## Where training runs

Every run, wherever it happens, writes a folder `runs/<run_id>/` with its config, environment,
per-epoch history and metrics. That's what the analyst compares. `python -m src.runlog` lists all runs.

### Laptop
Smoke tests (1 epoch, a few seconds) and small datasets. The agents run these for you.

### Google Colab (default for real runs)
1. Recommended: put your project folder in Google Drive and install
   [Google Drive for desktop](https://www.google.com/drive/download/) so it syncs to your laptop.
   (Or push the project to GitHub and use the notebook's `github` mode.)
2. Open `notebooks/colab_runner.ipynb` in Colab (File → Upload notebook, or open it from GitHub).
3. Runtime → Change runtime type → **T4 GPU**.
4. In step 1 of the notebook, set the project path and the `CONFIG` and `EXTRA_ARGS` the agent gave you. Then Runtime → Run all.
5. In Drive mode the results sync back to `runs/` automatically. In GitHub mode, unzip the downloaded `new_runs.zip` in your project folder.

*Colab with a private repo:* use Drive mode, or put a GitHub token in Colab's Secrets panel and
clone with `https://<token>@github.com/you/repo.git`. Never paste a token into the notebook itself.

### NDSU CCAST (optional, for long or large runs)

**One-time setup**

1. Get a CCAST account and project from your instructor. You'll need your **project account
   name** (it looks like `x-ccast-prj-…`) and the **login host name**.
2. On your laptop, add this to `~/.ssh/config`:
   ```
   Host ccast
       HostName <CCAST login host>
       User <your NDSU username>
       ControlMaster auto
       ControlPath ~/.ssh/cm-%r@%h:%p
       ControlPersist 8h
   ```
   `ControlMaster` keeps one connection open after you log in once (including any two-factor
   prompt). Your AI agent can then run `ssh ccast …` commands without needing your password.
   Set up an SSH key (`ssh-keygen`, then `ssh-copy-id ccast`) if CCAST allows it.
3. Log in once (`ssh ccast`) and create the Python environment:
   ```bash
   module avail                      # find the conda/python module name
   module load anaconda3             # or whatever module avail shows
   conda create -n nn python=3.11 -y
   conda activate nn
   # pick the CUDA build matching CCAST's GPUs/driver: https://pytorch.org/get-started/locally/
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
4. In `scripts/ccast/job.slurm`, replace `x-ccast-prj-CHANGE_ME` with your project account,
   and check the partition and module lines against CCAST's documentation.
5. On CCAST, after your first `push` (below): `cd projects/<project> && conda activate nn && pip install -r requirements.txt`.

**Every time** (from your laptop, in the project folder; the agents can do this for you):
```bash
scripts/ccast/ccast.sh submit configs/my_experiment.yaml   # copies code + data, submits the job
scripts/ccast/ccast.sh status                               # is it running?
scripts/ccast/ccast.sh log                                  # latest output
scripts/ccast/ccast.sh pull                                 # bring finished runs/ back
```
Windows: use WSL, since `rsync` isn't available in plain PowerShell.

Alternatively, connect with VS Code **Remote-SSH** to CCAST and run your AI tool there, if CCAST
allows it. Never train on the login node; always use `sbatch`.

---

## What's in this folder

| Path | What it is |
|---|---|
| `AGENTS.md` | Rules every AI tool follows (roles, pipeline, validity rules) |
| `CLAUDE.md` | Claude Code orchestration (loads `AGENTS.md`) |
| `.claude/agents/` | The five agents' instructions |
| `configs/` | **Your** hyperparameter choices, one file per experiment |
| `data/raw/`, `papers/` | **Your** data and papers |
| `reports/plan.md` | **Your** research plan |
| `reports/LEARNING_LOG.md` | What the agents built and the concepts behind it |
| `src/runlog.py` | Records every run the same way (provided) |
| `src/*.py` | Code the agents write |
| `runs/`, `figures/`, `reports/` | Results |
| `notebooks/colab_runner.ipynb` | Runs training on Colab |
| `scripts/ccast/` | Runs training on CCAST |

## License

The template is under the MIT License (`LICENSE`). Your project built from it is yours. If you
publish it, keep the license notice for the template files, and choose a license for your own work.
