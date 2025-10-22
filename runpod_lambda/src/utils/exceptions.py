
class InvalidPromptError(ValueError):
    def __init__(self, prompt):
        super().__init__(f"Error, prompt '{prompt}' is invalid.")
