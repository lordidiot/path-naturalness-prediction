from tqdm import tqdm

class BaseVertexFeature:
    def calculate(self, vertex: str) -> float:
        raise NotImplementedError
    
    def calculate_batch(self, vertices: list[str]) -> list[float]:
        return [self.calculate(vertex) for vertex in tqdm(vertices)]

bidirectional_relations = set([
    "RelatedTo", "Synonym", "Antonym", "DistinctFrom", "LocatedNear", "SimilarTo",
    "EtymologicallyRelatedTo",
])

unidirectional_relations = set([
    "FormOf", "IsA", "PartOf", "HasA", "UsedFor", "CapableOf", "AtLocation", "Causes",
    "HasSubevent", "HasFirstSubevent", "HasLastSubevent", "HasPrerequisite", "HasProperty",
    "MotivatedByGoal", "ObstructedBy", "Desires", "CreatedBy", "DerivedFrom", "SymbolOf",
    "DefinedAs", "MannerOf", "HasContext", "EtymologicallyDerivedFrom", "CausesDesire",
    "MadeOf", "ReceivesAction", "ExternalURL",
])
