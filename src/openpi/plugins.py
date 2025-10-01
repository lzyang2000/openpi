from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class PiSearchPathPlugin(SearchPathPlugin):
    """A Hydra search path plugin for Pi.

    This appends every config under `configs` to the search path. This allows us to use mhw's CLI even with configs
    defined only in the `Pi` package.
    """

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Manipulate the search path to include Pi configurations."""
        search_path.append("pi", "pkg://openpi/configs/mhw")
