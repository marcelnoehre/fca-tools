from src.utils.constants import RESET

def log(
        msg: str,
        color: str=None
    ) -> None:
    '''
    Print a highlighted message to the console.
    '''
    print(f'{color}{msg}{RESET}')