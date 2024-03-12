import treeppl

import json

@treeppl.constructor("QTNode")
class QTNode(treeppl.Tree.Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.age = float(self.age)

    def __repr__(self):
        return f"QTNode(left={self.left!r}, right={self.right!r}, age={self.age})"

    def to_dict(self):
        return {"type": "QTNode", "age": self.age, "left": self.left.to_dict() if self.left else None, "right": self.right.to_dict() if self.right else None}

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

@treeppl.constructor("QTLeaf")
class QTLeaf(treeppl.Tree.Leaf):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.age = float(self.age)
        self.index = int(self.index)
        self.sequence = str(self.sequence)
        self.stateSequence = self.construct_state_sequence(self.sequence)
        self.character = str(self.character)
        self.characterState = int(self.characterState)

    def __repr__(self):
        return (f"QTNode(age={self.age}, index={self.index}, sequence={self.sequence}, "
                f"stateSequence={self.stateSequence}, character={self.character}, "
                f"characterState={self.characterState})")

    def construct_state_sequence(self, sequence):
        mapping = {'t': 0, 'T': 0, 'c': 1, 'C': 1, 'a': 2, 'A': 2, 'g': 3, 'G': 3, '-': -1} # Or? '-': None
        return [mapping.get(letter, -1) for letter in sequence]

    def to_dict(self):
        return {"type": "QTLeaf", "age": self.age, "index": self.index, "sequence": self.sequence, "stateSequence": self.stateSequence, "character": self.character, "characterState": self.characterState}

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

