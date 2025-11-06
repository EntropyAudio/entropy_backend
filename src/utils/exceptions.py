
class InvalidPromptError(ValueError):
    def __init__(self, prompt):
        super().__init__(f"Error, prompt '{prompt}' is invalid.")

class InvalidRequestError(ValueError):
    def __init__(self, request):
        super().__init__(f"Error, request '{request}' is malformed.")
