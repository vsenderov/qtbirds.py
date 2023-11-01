import treeppl

@treeppl.constructor("QTNode")
class QTNode(treeppl.Tree.Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.age = float(self.age)

    def __repr__(self):
        return f"QTNode(left={self.left!r}, right={self.right!r}, age={self.age})"

@treeppl.constructor("QTLeaf")
class QTLeaf(treeppl.Tree.Leaf):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.age = float(self.age)
        self.index = int(self.index)
        self.sequence = str(self.sequence)
        self.stateSequence = [int(item) for item in self.stateSequence]
        self.character = str(self.character)
        self.characterState = int(self.characterState)

    def __repr__(self):
        return (f"QTNode(age={self.age}, index={self.index}, sequence={self.sequence}, "
                f"stateSequence={self.stateSequence}, character={self.character}, "
                f"characterState={self.characterState})")
