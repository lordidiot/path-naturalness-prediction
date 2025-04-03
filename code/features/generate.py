import pickle
from tqdm import tqdm
import logging

from .path_types import Path
from .base_feature import BaseVertexFeature, BaseEdgeFeature

def _get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(f"{__name__}.log")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def enumerate_vertices(data_path: str) -> set[str]:
    """
    Enumerates all the vertices in the dataset.

    Parameters
    ---
    data_path: str
        The path to the original dataset. Example: `../data/science/paths.pkl`
    """
    with open(data_path, "rb") as f:
        paths = pickle.load(f)
    vertices = set()
    for path in paths.values():
        path_string = path['forward']['short']
        vertices.update(item.lower() for i, item in enumerate(path_string.split(" ")) if i % 2 == 0)
    return vertices

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

    Note that this function skips the paths that ran into
    an exception during the call to `vertex_feature.calculate_batch(path)`.
    Check the error log `features.generate.log` to see the exception details.

    Parameters
    ---
    vertex_feature: BaseVertexFeature
        The vertex feature to generate data for.
    data_path: str
        The path to the original dataset. Example: `../data/science/paths.pkl`
    out: str
        The path to save the generated data. Example: `../data/science/features/v_freq.pkl`
    """
    logger = _get_logger()
    with open(data_path, "rb") as f:
        paths = pickle.load(f)
    data: dict[str, list[list[float]]] = dict()
    keys = list(paths.keys())
    for key in tqdm(keys):
        path = paths[key]
        for direction in ['forward', 'reverse']:
            path_string = path[direction]['short']
            words = [item.lower() for i, item in enumerate(path_string.split(" ")) if i % 2 == 0]
            try:
                values = vertex_feature.calculate_batch(words)
            except Exception as e:
                logger.error(f"Error at key: {key}")
                logger.error(e, exc_info=True)
                continue
            data[key + direction[0]] = values
    with open(out, "wb") as f:
        pickle.dump(data, f)

def run_edge_feature_on_original(edge_feature: BaseEdgeFeature,
                                 data_path: str,
                                 out: str) -> None:
    """
    Generates data for an edge feature on the original dataset.
    The generated data is in pickle format and has the following type:

    ```
    dict[str, list[list[float]]]
    ```

    It is a dictionary, with each key being an ID of a path
    in the data path file, appended by the direction, `f` or `r`.

    The value is the value of the feature as calculated by
    `edge_feature.calculate_batch(path)`, where `path`
    is the path with the corresponding ID and direction.

    Note that this function skips the paths that ran into
    an exception during the call to `edge_feature.calculate_batch(path)`.
    Check the error log `features.generate.log` to see the exception details.

    Parameters
    ---
    edge_feature: BaseEdgeFeature
        The edge feature to generate data for.
    data_path: str
        The path to the original dataset. Example: `../data/science/paths.pkl`
    out: str
        The path to save the generated data. Example: `../data/science/features/e_dir.pkl`
    """
    logger = _get_logger()
    with open(data_path, "rb") as f:
        paths = pickle.load(f)
    data: dict[str, list[list[float]]] = dict()
    keys = list(paths.keys())
    for key in tqdm(keys):
        path = paths[key]
        for direction in ['forward', 'reverse']:
            path_string = path[direction]['short']
            items = path_string.split(" ")
            edges = [(items[i], items[i + 1], items[i + 2]) for i in range(0, len(items) - 2, 2)]
            try:
                values = edge_feature.calculate_batch(edges)
            except Exception as e:
                logger.info(f"Error at key: {key}")
                logger.info(e, exc_info=True)
                continue
            data[key + direction[0]] = values
    with open(out, "wb") as f:
        pickle.dump(data, f)

def load_fixed_endpoints(filename) -> dict[tuple[str, str], list[Path]]:
    with open(filename, 'rb') as f:
        data: dict[tuple[str, str], list[Path]] = pickle.load(f)
        return data

def run_vertex_feature_on_fixed_endpoints(vertex_feature: BaseVertexFeature,
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

    Note that this function skips the paths that ran into
    an exception during the call to `vertex_feature.calculate_batch(path)`.
    Check the error log `features.generate.log` to see the exception details.

    Parameters
    ---
    vertex_feature: BaseVertexFeature
        The vertex feature to generate data for.
    data_path: str
        The path to the original dataset. Example: `../data/science/paths.pkl`
    out: str
        The path to save the generated data. Example: `../data/science/features/v_freq.pkl`
    """
    logger = _get_logger()
    data: dict[str, list[list[float]]] = dict()
    dataset = load_fixed_endpoints(data_path)
    for _, paths in tqdm(dataset.items()):
        for path in paths:
            isReverseList = [False, True]
            for isReverse in isReverseList:
                path_string = path.short(isReverse)
                words = [item.lower() for i, item in enumerate(path_string.split(" ")) if i % 2 == 0]
                try:
                    values = vertex_feature.calculate_batch(words)
                except Exception as e:
                    logger.error(f"Error at key: {path.id}")
                    logger.error(e, exc_info=True)
                    continue
                data[path.id + 'r' if isReverse else 'f'] = values
    with open(out, "wb") as f:
        pickle.dump(data, f)

def run_edge_feature_on_fixed_endpoints(edge_feature: BaseEdgeFeature,
                                 data_path: str,
                                 out: str) -> None:
    """
    Generates data for an edge feature on the original dataset.
    The generated data is in pickle format and has the following type:

    ```
    dict[str, list[list[float]]]
    ```

    It is a dictionary, with each key being an ID of a path
    in the data path file, appended by the direction, `f` or `r`.

    The value is the value of the feature as calculated by
    `edge_feature.calculate_batch(path)`, where `path`
    is the path with the corresponding ID and direction.

    Note that this function skips the paths that ran into
    an exception during the call to `edge_feature.calculate_batch(path)`.
    Check the error log `features.generate.log` to see the exception details.

    Parameters
    ---
    edge_feature: BaseEdgeFeature
        The edge feature to generate data for.
    data_path: str
        The path to the original dataset. Example: `../data/science/paths.pkl`
    out: str
        The path to save the generated data. Example: `../data/science/features/e_dir.pkl`
    """
    logger = _get_logger()
    data: dict[str, list[list[float]]] = dict()
    dataset = load_fixed_endpoints(data_path)
    for _, paths in tqdm(dataset.items()):
        for path in paths:
            isReverseList = [False, True]
            for isReverse in isReverseList:
                path_string = path.short(isReverse)
            items = path_string.split(" ")
            edges = [(items[i], items[i + 1], items[i + 2]) for i in range(0, len(items) - 2, 2)]
            try:
                values = edge_feature.calculate_batch(edges)
            except Exception as e:
                logger.info(f"Error at key: {path.id}")
                logger.info(e, exc_info=True)
                continue
            data[path.id + 'r' if isReverse else 'f'] = values
    with open(out, "wb") as f:
        pickle.dump(data, f)
