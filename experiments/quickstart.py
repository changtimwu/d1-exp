"""Run the two examples from the mlx-vlm d1 README: a text state and an image state."""

import json

from mlx_vlm import load, predict

MODEL = "LiquidAI/d1-3B"

model, processor = load(MODEL)

result = predict(model, processor, "I was charged twice, please refund one.", {
    "refund": {"type": "bool", "instructions": "Is the customer asking for a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges, refunds", "technical": "App faults"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Can wait", "Today", "Blocking the customer now"],
    },
})
print(json.dumps(result, indent=2, default=str))

photo = predict(model, processor, None, {
    "cats": {"type": "choice", "instructions": "How many cats are there?",
             "criteria": ["one", "two", "more"]},
}, images=["http://images.cocodataset.org/val2017/000000039769.jpg"])
print(json.dumps(photo, indent=2, default=str))
