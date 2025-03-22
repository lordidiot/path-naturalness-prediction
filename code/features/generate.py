import pickle

from .base_feature import BaseVertexFeature

def run_vertex_feature_on_original(vertex_feature: BaseVertexFeature,
                                   data_path: str,
                                   out: str) -> None:
    """
    Generates data for a vertex feature on the original dataset.
    The generated data is in pickle format and has the following type:

    ```
    dict[str, list[list[float]]]
    ```

    It is a dictionary, with each key being an ID of a path
    in the data path file, appended by the direction, `f` or `r`.

    The value is the value of the feature as calculated by
    `vertex_feature.calculate_batch(path)`, where `path`
    is the path with the corresponding ID and direction.

    Parameters
    ---
    vertex_feature: BaseVertexFeature
        The vertex feature to generate data for.
    data_path: str
        The path to the original dataset. Example: `../data/science/paths.pkl`
    out: str
        The path to save the generated data. Example: `../data/science/features/v_freq.pkl`
    """
    with open(data_path, "rb") as f:
        paths = pickle.load(f)
    data: dict[str, list[list[float]]] = dict()
    for key, path in paths.items():
        for direction in ['forward', 'reverse']:
            path_string = path[direction]['short']
            words = [item.lower() for i, item in enumerate(path_string.split(" ")) if i % 2 == 0]
            values = vertex_feature.calculate_batch(words)
            data[key + direction[0]] = values
    with open(out, "wb") as f:
        pickle.dump(data, f)
