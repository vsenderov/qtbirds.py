import treeppl
from typing import Optional, Union, List
import json
import numpy as np

@treeppl.constructor("QTNode")
class QTNode(treeppl.Tree.Node):
    age: float
    left: Optional[Union['QTNode', 'QTLeaf']] = None
    right: Optional[Union['QTNode', 'QTLeaf']] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.age = float(self.age)

    def __repr__(self):
        return f"QTNode(left={self.left!r}, right={self.right!r}, age={self.age})"

    def to_dict(self):
        return {"type": "QTNode", "age": self.age, "left": self.left.to_dict() if self.left else None, "right": self.right.to_dict() if self.right else None}

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    @property
    def sequence_length(self) -> int:
        """
        Traverse the tree depth-first until a leaf is reached and return the length
        of the sequence or stateSequence at the first encountered leaf.

        :return: The length of the sequence or stateSequence at the first leaf encountered
        """
        # always traverse left
        return self.left.sequence_length


@treeppl.constructor("QTLeaf")
class QTLeaf(treeppl.Tree.Leaf):
    age: float
    index: int
    sequence: str
    stateSequence: list = []
    character: str
    characterState: int
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.age = float(self.age)
        self.index = int(self.index)
        self.sequence = str(self.sequence)
        self.stateSequence = self.construct_state_sequence(self.sequence)
        self.character = str(self.character)
        self.characterState = int(self.characterState)

    def __repr__(self):
        return (f"QTLeaf(age={self.age}, index={self.index}, sequence={self.sequence}, "
                f"stateSequence={self.stateSequence}, character={self.character}, "
                f"characterState={self.characterState})")

    def construct_state_sequence(self, sequence):
        mapping = {'t': 0, 'T': 0, 'c': 1, 'C': 1, 'a': 2, 'A': 2, 'g': 3, 'G': 3, '-': -1} # Or? '-': None
        return [mapping.get(letter, -1) for letter in sequence]

    def to_dict(self):
        return {"type": "QTLeaf", "age": self.age, "index": self.index, "sequence": self.sequence, "stateSequence": self.stateSequence, "character": self.character, "characterState": self.characterState}

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
    
    def get_message(self, state):
        """ Converts a state into a 4-dimensional vector. """
        messages = [
            np.array([-1., -1., -1., -1.]),  # encoded for missing data
            np.array([1., 0., 0., 0.]),
            np.array([0., 1., 0., 0.]),
            np.array([0., 0., 1., 0.]),
            np.array([0., 0., 0., 1.])
        ]
        return messages[state + 1] if state >= 0 else messages[0]
    
    @property
    def sequence_length(self) -> int:
        return len(self.sequence)

    @property
    def messageList(self):
        """ Converts the stateSequence into a list of 4-dimensional message vectors. """
        return list(map(self.get_message, self.stateSequence))
    
    def get_char_message(self, message_vectors):
        """ Retrieve the message vector for the character state from the provided list. """
        if 0 <= self.characterState < len(message_vectors):
            return message_vectors[self.characterState]
        else:
            return [-1., -1., -1., -1.]  # Return a default vector for out of range or missing data



