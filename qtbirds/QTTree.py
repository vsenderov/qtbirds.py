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
    def messageList(self):
        """ Converts the stateSequence into a list of 4-dimensional message vectors. """
        return list(map(self.get_message, self.stateSequence))
    
    def get_char_message(self, message_vectors):
        """ Retrieve the message vector for the character state from the provided list. """
        if 0 <= self.characterState < len(message_vectors):
            return message_vectors[self.characterState]
        else:
            return [-1., -1., -1., -1.]  # Return a default vector for out of range or missing data


class AugmentedQTNode(QTNode):
    s_jumps_left: List[int]
    s_jumps_right: List[int]

    def __init__(self, age, left=None, right=None, s_jumps_left=None, s_jumps_right=None):
        super().__init__(age=age, left=left, right=right)
        self.s_jumps_left = s_jumps_left if s_jumps_left is not None else []
        self.s_jumps_right = s_jumps_right if s_jumps_right is not None else []

    def __repr__(self):
        return (f"AugmentedQTNode(age={self.age}, left={self.left!r}, right={self.right!r}, "
                f"s_jumps_left={self.s_jumps_left}, s_jumps_right={self.s_jumps_right})")

    def to_dict(self):
        return {
            "type": "AugmentedQTNode",
            "age": self.age,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
            "s_jumps_left": self.s_jumps_left,
            "s_jumps_right": self.s_jumps_right
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    
def find_sequence_length(tree: Union[QTNode, QTLeaf]) -> int:
    """
    Traverse the tree depth-first until a leaf is reached and return the length
    of the sequence or stateSequence at the first encountered leaf.

    :param tree: The root of the tree or subtree, which can be either a QTNode or QTLeaf
    :return: The length of the sequence or stateSequence at the first leaf encountered
    """
    if isinstance(tree, QTLeaf):
        # If the node is a leaf, return the length of the sequence
        return len(tree.sequence)
    elif isinstance(tree, QTNode):
        # If the node is not a leaf, recursively search the left child first (if it exists), 
        # otherwise search the right child
        if tree.left is not None:
            return find_sequence_length(tree.left)
        elif tree.right is not None:
            return find_sequence_length(tree.right)
    # If neither condition is met, return 0 indicating no sequence found
    return 0


class ComputedQTNode:
    """
    TODO should probably just derive from AugmentedQTNode
    """
    age: float                                # A float representing the age of the node
    messageList: List[np.ndarray]             # A list of n-dimensional numpy arrays representing message vectors
    charMessage: np.ndarray                   # A numpy array representing the character message vector
    left: Union['QTLeaf', 'ComputedQTNode']  # Left child node
    right: Union['QTLeaf', 'ComputedQTNode'] # Right child node
    s_jumps_left: List[int]                   # List of integers for left jumps
    s_jumps_right: List[int]                  # List of integers for right jumps

    def __init__(self, age, messageList, charMessage, left=None, right=None, s_jumps_left=None, s_jumps_right=None):
        self.age = float(age)
        self.messageList = [np.array(vector) for vector in messageList]
        self.charMessage = np.array(charMessage)
        self.left = left
        self.right = right
        self.s_jumps_left = s_jumps_left if s_jumps_left is not None else []
        self.s_jumps_right = s_jumps_right if s_jumps_right is not None else []

    def __repr__(self):
        return (f"ComputedQTNode(age={self.age}, messageList={self.messageList}, "
                f"charMessage={self.charMessage}, left={self.left}, right={self.right}, "
                f"s_jumps_left={self.s_jumps_left}, s_jumps_right={self.s_jumps_right})")

    def to_dict(self):
        # Convert numpy arrays to lists for JSON serialization and handle left/right
        node_dict = {
            "type": "ComputedQTNode",
            "age": self.age,
            "messageList": [list(vector) for vector in self.messageList],
            "charMessage": list(self.charMessage),
            "s_jumps_left": self.s_jumps_left,
            "s_jumps_right": self.s_jumps_right
        }
        if self.left:
            node_dict["left"] = self.left.to_dict()
        if self.right:
            node_dict["right"] = self.right.to_dict()
        return node_dict

    def to_json(self):
        import json
        return json.dumps(self.to_dict(), indent=4)


class PartiallyComputedQTNode(AugmentedQTNode):
    left: Union['QTLeaf', 'ComputedQTNode']
    right: Union['QTLeaf', 'ComputedQTNode']

    def __init__(self, age, left, right, s_jumps_left=None, s_jumps_right=None):
        super().__init__(age=age, left=left, right=right, s_jumps_left=s_jumps_left, s_jumps_right=s_jumps_right)

    def __repr__(self):
        return (f"PartiallyComputedQTNode(age={self.age}, left={self.left!r}, right={self.right!r}, "
                f"s_jumps_left={self.s_jumps_left}, s_jumps_right={self.s_jumps_right})")

    def to_dict(self):
        return {
            "type": "PartiallyComputedQTNode",
            "age": self.age,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
            "s_jumps_left": self.s_jumps_left,
            "s_jumps_right": self.s_jumps_right
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
