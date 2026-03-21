from environments.base import SubsetEnv

# Profiling using `line_profiler`
import sys
maybe_profile=lambda x:x
if "kernprof" in sys.modules: maybe_profile=lambda x:profile(x)

class AlgHirschEnv(SubsetEnv):
    def __init__(self, *,
        num_envs: int,
        n: int,
        d: int,
        device: str,
        max_len: int = 128
    ) -> None:
        self._n = n
        self._d = d
        super().__init__(
            num_envs    = num_envs,
            max_len     = max_len,
            dim         = self._sgraph.num_generators,
            device      = device)

