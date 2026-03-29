# Feature Plan Template

Use this template to plan any new feature. Copy it, fill in the placeholders,
and follow the conventions at the bottom.

---

## Feature: [SHORT FEATURE NAME]

### Project Goal

[5 sentence description of the overall feature outcome.]

- [key deliverable or behavior]
- [key deliverable or behavior]
- [key deliverable or behavior]
- [key deliverable or behavior]
- [key constraint: platform, format, runtime, etc.]

---

### Step [N]: [Step title — verb phrase describing what this step produces]

**Goal:**
[One sentence describing what this step achieves and why it matters.]

**Tasks:**

- [Verb-led task: what to build, implement, define, or decide.]
- [Verb-led task.]
- [Verb-led task.]
- [Decision point: "Decide [X] and document the rationale."]

**Tests:**

- [Happy path test with specific input.]
- [Happy path test with different input shape/size.]
- [Edge case: empty/zero/missing input.]
- [Edge case: malformed or invalid input.]
- [Boundary condition test.]
- [Verification test: numeric check, size check, or structural assertion.]

**Expected Results:**

- [Measurable pass criterion.]
- [Measurable pass criterion.]
- [Measurable pass criterion.]

**Artifacts Produced:**

- [File or module created or modified.]
- [Spec document if applicable.]

**Dependencies/Prerequisites:**

- [What must exist before this step can start.]
- [Which earlier step's output this step consumes.]

**Risks/Failure Modes:**

- [Risk and its mitigation.]
- [Risk and its mitigation.]

---

### Step [N+1]: [Next step title]

**Goal:**
[One sentence.]

**Tasks:**

- [...]

**Tests:**

- [...]

**Expected Results:**

- [...]

**Artifacts Produced:**

- [...]

**Dependencies/Prerequisites:**

- [...]

**Risks/Failure Modes:**

- [...]

---

*(Repeat the step block for each additional step.)*

---

## Plan Conventions

### Naming

- Steps are numbered sequentially starting at 1.
- Step titles start with a verb (Define, Build, Implement, Add, Validate,
  Integrate, Tune, Create).

### Ordering Principles

- Contracts and specs come before implementation.
- Environment setup comes before code that depends on it.
- Components are built bottom-up: smallest isolated unit first.
- Integration comes after all components exist independently.
- Validation and verification come after the thing being validated exists.
- Performance tuning comes after correctness is proven.
- Failure recovery and hardening come after the happy path works.
- Acceptance testing is always the final step.

### Step Structure Rules

- **Goal:** one sentence, states what and why.
- **Tasks:** concrete and actionable; each starts with a verb.
- **Tests:** specific inputs, procedures, or scenarios; not vague.
- **Expected Results:** measurable pass/fail criteria; not aspirational.

### Decision Tasks

- When a step requires a design choice, include it as a task:
  "Decide [X] and document the rationale."
- The decision should be resolved within the step, not deferred.

### Test Categories to Consider

- Happy path (normal input, expected output).
- Edge cases (empty, zero, minimal, maximal, boundary).
- Invalid input (wrong type, wrong shape, corrupt data).
- Failure modes (missing dependency, disk full, OOM).
- Determinism (same input produces same output across runs).
- Integration (output feeds correctly into the next step's input).

### Artifact Tracking

- Each step should name the files it creates or modifies.
- Specs and contracts are artifacts too.

### Dependency Tracking

- Each step should name which earlier steps it depends on.
- Steps with no upstream dependency can potentially be parallelized.
