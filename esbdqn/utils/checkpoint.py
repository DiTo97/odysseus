import dill as pickle
import glob
import numpy as np
import pathlib
import typing as t

# Custom imports
from dqn_zoo import parts


def attr2dict(state: parts.AttributeDict) \
             -> t.Dict:
    def get_state_rec(v: t.Any) -> t.Any:
        if getattr(v, 'get_state', None):
            return v.get_state()

        if type(v) is dict:
            return {j: get_state_rec(x)
                    for j, x in v.items()}
        
        # Only if primitive
        return v

    return {k: get_state_rec(v)
            for k, v in state.items()}


def dict2attr(dict_state: t.Dict) \
             -> parts.AttributeDict:
    state = parts.AttributeDict()

    for k, v in dict_state.items():
        state[k] = v

    return state


class PickleCheckpoint(parts.NullCheckpoint):
    """
    Pickle checkpoint for training on ODySSEUS. It will store:
        - Training iteration;
        - Deterministic RNG state;
        - Pick-up agent network weights;
        - Drop-off agent network weights;
        - CSV file writer.
    """
    def __init__(self,
                 models_dir_path: pathlib.Path,
                 model_name: str):
        super().__init__()

        self._models_dir_path = models_dir_path
        self._model_name = model_name

    def save(self) -> None:
        pathlib.Path.mkdir(self._models_dir_path,
                           parents=True, exist_ok=True)

        with open(str(self._models_dir_path / self._model_name)
                  + '-' + str(self.state.iteration - 1)
                  + '.pkl', 'wb') as f:
            pickle.dump(attr2dict(self.state), f)

    def can_be_restored(self) -> bool:
        return len(glob.glob(str(self._models_dir_path) + '/'
                             + self._model_name + '-*')) > 0

    def restore(self) -> None:
        cached_models_paths = glob.glob(str(self._models_dir_path) + '/'
                            + self._model_name + '-*')

        cached_models_iters = [int(m.split('.')[-2].split('-')[-1])
                               for m in cached_models_paths]

        last_model_path = cached_models_paths[
            np.argmax(cached_models_iters)]

        with open(last_model_path, 'rb') as f:
            self.state = dict2attr(pickle.load(f))
