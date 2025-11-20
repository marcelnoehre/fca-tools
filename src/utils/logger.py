from src.utils.constants import RESET

def log(
        msg: str,
        color: str=None
    ) -> None:
    '''
    Print a highlighted message to the console.

    Args:
        msg (str): The message to print.
        color (str, optional): ANSI color code for highlighting. Defaults to None.
    '''
    print(f'{color}{msg}{RESET}')
