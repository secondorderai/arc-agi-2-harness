# ARC-AGI-2 Level 1 Skills



## Consistent color mapping

- Skill ID: `learned-color-mapping`
- Level: 1
- Family: `color-mapping`
- Training examples: 5

### Detection cues

- Input and output shapes match and each source color has one target color

### Invariants

- The mapping is consistent across every cell and pair

### Strategy

1. Infer a global source-to-target color map and preserve unmapped colors.

### Failure modes

- Using position-dependent recoloring without evidence

### Program templates

```json
[
  {
    "args": {
      "mapping": {
        "1": 5,
        "2": 6,
        "3": 4,
        "4": 3,
        "5": 1,
        "6": 2,
        "8": 9,
        "9": 8
      }
    },
    "op": "recolor",
    "steps": []
  },
  {
    "args": {
      "mapping": {
        "0": 0,
        "2": 4,
        "3": 6,
        "4": 0,
        "6": 0
      }
    },
    "op": "recolor",
    "steps": []
  },
  {
    "args": {
      "mapping": {
        "6": 2,
        "7": 7
      }
    },
    "op": "recolor",
    "steps": []
  },
  {
    "args": {
      "mapping": {
        "1": 1,
        "7": 5,
        "8": 8
      }
    },
    "op": "recolor",
    "steps": []
  },
  {
    "args": {
      "mapping": {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 8,
        "6": 6,
        "7": 7,
        "8": 5,
        "9": 9
      }
    },
    "op": "recolor",
    "steps": []
  }
]
```

## Rigid spatial transformation

- Skill ID: `learned-spatial-transform`
- Level: 1
- Family: `spatial-transform`
- Training examples: 7

### Detection cues

- Colors and object shapes are preserved while coordinates change

### Invariants

- Color counts and object topology remain constant

### Strategy

1. Test rotations, reflections, and transpose; verify exactly.

### Failure modes

- Confusing horizontal and vertical reflection

### Program templates

```json
[
  {
    "args": {
      "turns": 2
    },
    "op": "rotate",
    "steps": []
  },
  {
    "args": {
      "axis": "horizontal"
    },
    "op": "flip",
    "steps": []
  },
  {
    "args": {
      "axis": "vertical"
    },
    "op": "flip",
    "steps": []
  },
  {
    "args": {},
    "op": "transpose",
    "steps": []
  },
  {
    "args": {
      "turns": 3
    },
    "op": "rotate",
    "steps": []
  }
]
```
