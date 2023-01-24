

class UDParseError(Exception):
    def __init__(self, msg):
        self.message = msg
        super().__init__(self.message)


class UDParseConlluError(UDParseError):
    def __init__(self, msg=None):
        self.message = "Bad Conllu Format"
        if msg:
            self.message += ". " + msg

        super().__init__(self.message)

class UDParseLanguageError(UDParseError):
    def __init__(self, msg=None):
        self.message = "Invalid language"
        if msg:
            self.message += ". " + msg

        super().__init__(self.message)
