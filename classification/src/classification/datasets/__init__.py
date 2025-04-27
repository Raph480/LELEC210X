from pathlib import Path
from typing import List, Tuple

SOUND_DURATION: float = 5.0


def get_cls_from_path(file: Path) -> str:
    """
    Return a sound class from a given path.

    By convention, sound files should be named `some/path/cls_index.format`,
    where format can be any supported audio format, index is some
    unique number for that class,
    and cls is the class name.

    :param file: The file path.
    :return: The class name.
    """
    return file.stem.split("_", maxsplit=1)[0]


class Dataset:
    def __init__(
        self, 
        folder: Path = Path(__file__).parent / "soundfiles", 
        format: str = "wav",
        filter_str: str = None,
    ):
        """
        Initialize a dataset from a given folder, including
        subfolders. Uses :func:`get_cls_from_path` to determine
        the sound class of each file.

        Note: we sort files because directory traversal is
        not consistent accross OSes, and returning different
        file orderings may confuse students :'-).

        :param folder: Where to find the soundfiles.
        :param format: The sound files format, use
            `'*'` to include all formats.
        """
        
        if isinstance(folder, str):
            folder = Path(folder)  # Convert string to Path object

        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Error: The folder '{folder}' does not exist or is not a directory.")

        files = {}
        self.size = 0

        for file in sorted(folder.glob("**/*." + format)):
            if filter_str:
                if filter_str in file.stem:
                    cls = get_cls_from_path(file)
                    files.setdefault(cls, []).append(file)
            else:
                if ("background" not in file.stem or "merged" in file.stem) and "youtube" not in file.stem: #TODO: add potential youtube
                    cls = get_cls_from_path(file)
                    files.setdefault(cls, []).append(file)

        self.files = files
        self.nclass = len(files)
        self.naudio = len(files[list(files.keys())[0]])
        self.size = self.nclass * self.naudio

    def __len__(self) -> int:
        """
        Return the number of sounds in the dataset.
        """
        return self.size

    def __getitem__(self, cls_index: Tuple[str, int]) -> Path:
        """
        Return the file path corresponding the
        the (class name, index) pair.

        :cls_index: Class name and index.
        :return: The file path.
        """
        cls, index = cls_index
        return self.files[cls][index]

    def __getname__(self, cls_index: Tuple[str, int]) -> str:
        """
        Return the name of the sound selected.

        :cls_index: Class name and index.
        :return: The name of the sound.
        """
        cls, index = cls_index
        return self.files[cls][index].stem

    def get_class_files(self, cls_name: str) -> List[Path]:
        """
        Return the list of files of a given class.

        :cls_name: Class name.
        :return: The list of file paths.
        """
        return self.files[cls_name]

    def list_classes(self) -> List[str]:
        """
        Return the list of classes
        in the given dataset.

        :return: The list of classes.
        """
        return list(self.files.keys())
    
