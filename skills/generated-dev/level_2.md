# ARC-AGI-2 Level 2 Skills



## Object extraction and selection

- Skill ID: `learned-object-selection`
- Level: 2
- Family: `object-selection`
- Training examples: 9

### Detection cues

- Output is a crop or one connected component from a larger scene

### Invariants

- Selected object cells preserve their relative coordinates

### Strategy

1. Segment components, compare roles, select by a stable property, then crop.

### Failure modes

- Assuming the largest object is always selected

### Program templates

```json
[
  {
    "args": {
      "background": 0
    },
    "op": "crop_background",
    "steps": []
  },
  {
    "args": {
      "background": 0,
      "criterion": "largest"
    },
    "op": "select_component",
    "steps": []
  },
  {
    "args": {
      "background": 0,
      "criterion": "smallest"
    },
    "op": "select_component",
    "steps": []
  },
  {
    "args": {},
    "op": "compose",
    "steps": [
      {
        "args": {
          "background": 2,
          "criterion": "smallest"
        },
        "op": "select_component",
        "steps": []
      },
      {
        "args": {
          "mapping": {
            "0": 2,
            "3": 3,
            "5": 8
          }
        },
        "op": "recolor",
        "steps": []
      }
    ]
  },
  {
    "args": {},
    "op": "compose",
    "steps": [
      {
        "args": {
          "background": 1
        },
        "op": "crop_background",
        "steps": []
      },
      {
        "args": {
          "mapping": {
            "1": 0,
            "2": 2,
            "3": 3,
            "5": 5,
            "6": 6
          }
        },
        "op": "recolor",
        "steps": []
      }
    ]
  },
  {
    "args": {},
    "op": "compose",
    "steps": [
      {
        "args": {
          "background": 5,
          "criterion": "largest"
        },
        "op": "select_component",
        "steps": []
      },
      {
        "args": {
          "mapping": {
            "1": 4,
            "5": 5,
            "8": 1
          }
        },
        "op": "recolor",
        "steps": []
      }
    ]
  },
  {
    "args": {},
    "op": "compose",
    "steps": [
      {
        "args": {
          "background": 5,
          "criterion": "smallest"
        },
        "op": "select_component",
        "steps": []
      },
      {
        "args": {
          "mapping": {
            "1": 2,
            "2": 4,
            "4": 6,
            "6": 8
          }
        },
        "op": "recolor",
        "steps": []
      }
    ]
  }
]
```

## Scaling, tiling, and framing

- Skill ID: `learned-size-and-repetition`
- Level: 2
- Family: `size-and-repetition`
- Training examples: 2

### Detection cues

- Output dimensions are a simple multiple or padded form of the input

### Invariants

- Repeated copies preserve cell order

### Strategy

1. Compare dimension ratios, distinguish pixel scaling from whole-grid tiling.

### Failure modes

- Confusing scale with tile

### Program templates

```json
[
  {
    "args": {
      "columns": 3,
      "rows": 3
    },
    "op": "scale",
    "steps": []
  },
  {
    "args": {
      "columns": 2,
      "rows": 2
    },
    "op": "scale",
    "steps": []
  }
]
```

## Compositional transformation

- Skill ID: `level2-composition`
- Level: 2
- Family: `composition`
- Training examples: 0

### Detection cues

- No single primitive fits every demonstration
- Output preserves several input relations

### Invariants

- Apply one ordered program to every demonstration
- Do not special-case a grid

### Strategy

1. Describe objects and relations before proposing transformations.
2. Compose the smallest spatial, color, selection, or repetition operations.
3. Execute the full program on every demonstration and revise from exact diffs.

### Failure modes

- Correct operations in the wrong order
- A program that fits only one pair
