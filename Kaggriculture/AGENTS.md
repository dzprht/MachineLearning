# Agent Instructions

## 1. Sources of truth

Before doing any non-trivial work, read:

1. `README.md`
2. The task-author context file in the repository root (`context`, `context.md`, `CONTEXT.md`, or equivalent)
3. `PROJECT_MAP.md`, if it exists
4. `IDEAS.md`, if it exists

The task-author context file is read-only unless the user explicitly asks to modify it.

`README.md` and the task-author context define the problem and constraints.

`PROJECT_MAP.md` and `IDEAS.md` are working memory maintained for future agents.

---

## 2. Understand before changing

Before modifying code:

1. Understand the relevant part of the current implementation.
2. Search for existing functions, classes, utilities, configs, and abstractions that already solve part of the problem.
3. Prefer semantic code navigation with Serena when available:

   * inspect symbol overviews
   * find symbols
   * find references
   * inspect callers/callees where useful
4. Read entire large files only when necessary.

Do not start implementing from assumptions about how the repository is structured.

---

## 3. Minimal-change policy

Make the smallest change that properly tests or implements the requested idea.

Do not:

* refactor unrelated code
* rewrite working code merely because another implementation looks cleaner
* create abstractions for hypothetical future needs
* introduce wrappers that are used only once without a concrete benefit
* add compatibility layers that are not required
* create duplicate implementations of existing functionality
* add configuration options that are not currently needed
* introduce additional dependencies when the standard library or existing dependencies are sufficient
* keep an old and a new implementation side-by-side unless comparison is explicitly required
* create generic helper modules merely to shorten existing code

Prefer:

* extending existing code
* reusing existing abstractions
* local changes
* deleting code that becomes obsolete
* simple explicit implementations over speculative generality

A larger diff requires a concrete reason.

---

## 4. RL experiment workflow

This repository is used to solve an RL task.

Treat changes to the solution as experiments whenever their effect on performance is uncertain.

Before implementing an experimental idea:

1. Read `IDEAS.md`.
2. Check whether the idea:

   * already exists
   * was already tested
   * was rejected
   * is currently being tested
   * overlaps with another idea

Do not unknowingly repeat an experiment.

### User ideas

When the user explicitly gives an idea to test, add it to `IDEAS.md` as `APPROVED` unless the user says they only want to record/discuss it.

### Agent ideas

If you think of a potentially useful experiment:

1. Search `IDEAS.md` first.
2. If the idea is already present, use the existing entry.
3. If it is genuinely new, add it as `PROPOSED`.
4. Explain the idea briefly to the user.
5. Do NOT implement or run it until the user approves it.

Do not generate a large backlog of speculative ideas. Add an idea only when there is a concrete reason it may improve the current solution.

### Approved experiments

When starting an approved experiment:

1. Change its status to `TESTING`.
2. Record what is being changed.
3. Record what metric/evaluation determines whether it helped.
4. Keep the experiment as isolated as reasonably possible.

After evaluation:

1. Record the result.
2. Record the relevant metric(s).
3. Record the conclusion.
4. Change the status to:

   * `TESTED: KEEP`
   * `TESTED: REJECT`
   * `TESTED: INCONCLUSIVE`

Never delete failed ideas from `IDEAS.md`.
Failed experiments are useful information for future agents.

---

## 5. RL evaluation discipline

RL results may be noisy.

When practical:

* compare experiments against the current baseline
* use the same evaluation protocol for baseline and variant
* use the same seed set when comparing variants
* prefer multiple evaluation seeds over conclusions from one stochastic run
* change one meaningful factor at a time
* record relevant hyperparameters
* distinguish training reward from the actual target/evaluation metric

Do not claim an improvement from a noisy or incomparable run.

Do not quietly change the evaluation procedure in a way that makes a new idea look better.

If an experiment is expensive to run and the user has not explicitly approved the cost, ask before launching it.

---

## 6. IDEAS.md is experimental memory

`IDEAS.md` must remain concise enough that a new agent can read it at the beginning of a session.

Each idea should contain:

* status
* hypothesis
* proposed change
* reason
* evaluation method
* result, when available
* conclusion, when available

Do not fill it with long implementation details.

Use source-code references or filenames instead.

---

## 7. Maintain PROJECT_MAP.md

`PROJECT_MAP.md` is a concise semantic map of the current solution.

If it does not exist, create it after you understand enough of the repository.

It should describe things such as:

* important entry points
* important modules
* important classes/functions
* configuration
* environment / reward logic
* agent / policy implementation
* training loop
* evaluation code
* data flow
* where important hyperparameters live

Example structure:

```text
train.py
└── main training entry point
    ├── creates environment
    ├── creates agent
    └── starts training loop

src/agent.py
└── Agent
    ├── select_action()
    └── update()

src/env.py
└── Environment wrapper
    ├── reset()
    └── step()

evaluate.py
└── deterministic evaluation of trained policy
```

This is a semantic map, not a dump of every file in the repository.

Do not document trivial files.

When responsibilities, important symbols, or data flow change, update `PROJECT_MAP.md`.

Treat `PROJECT_MAP.md` as an index, not absolute ground truth. Verify relevant information against the source before editing code.

---

## 8. External libraries

When behavior, syntax, configuration, or API of a third-party library is uncertain, use Context7 when available.

Use Context7 especially for:

* library APIs
* version-specific behavior
* configuration
* deprecated functionality
* framework conventions

Do not invent library methods or parameters from memory when documentation can be checked.

Do not use Context7 unnecessarily for code that is entirely internal to this repository.

---

## 9. Serena usage

When Serena is available, prefer it for understanding and navigating non-trivial code.

Prefer symbolic operations such as:

* symbol overview
* find symbol
* find references
* semantic refactoring

over repeatedly reading entire files or performing broad text searches.

Use ordinary text/file tools when they are simpler for:

* tiny files
* configuration files
* Markdown
* exact text searches
* non-code assets

Do not use Serena merely for the sake of using Serena.

---

## 10. Before finishing a change

Before considering a task complete:

1. Run the relevant tests/checks.
2. Verify that unrelated behavior was not changed.
3. Check whether `PROJECT_MAP.md` needs updating.
4. Check whether `IDEAS.md` needs updating.
5. Remove temporary/debugging code.
6. Remove dead code made obsolete by the change.
7. Briefly report:

   * what changed
   * what was tested
   * result
   * remaining uncertainty

Do not leave experimental code in the repository without recording its outcome.
