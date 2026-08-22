# ARC-AGI-2 Level 3 Skills



## Symbolic interpretation

- Skill ID: `level3-symbolic`
- Level: 3
- Family: `symbolic-interpretation`
- Training examples: 0

### Detection cues

- Shapes behave as tokens, instructions, or role labels rather than decoration

### Invariants

- Meaning must remain consistent across demonstrations

### Strategy

1. List plausible semantic roles for each repeated symbol.
2. Test each role by predicting all demonstration outputs.
3. Prefer the role assignment with the shortest exact program.

### Failure modes

- Treating every symbol as a geometric pattern
- Fixating on symmetry alone

## Contextual rule application

- Skill ID: `level3-context`
- Level: 3
- Family: `contextual-rule`
- Training examples: 0

### Detection cues

- The same local pattern changes differently depending on position or neighboring objects

### Invariants

- The branch condition must be observable in the input

### Strategy

1. Identify contexts that partition objects into roles.
2. Write the branch predicate before writing either transformation.
3. Verify both branches independently, then verify their composition.

### Failure modes

- Using a global rule where a conditional rule is required
