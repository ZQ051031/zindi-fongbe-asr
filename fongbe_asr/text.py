import unicodedata as ud


def normalize(text):
    """Canonicalize Unicode and case without removing lexical tone marks."""
    out = []
    for char in ud.normalize("NFD", text.casefold().replace("’", "'")):
        kind = ud.category(char)[0]
        if kind in "LN" or char in "'-" or (
            kind == "M" and out and out[-1] != " " and out[-1] not in "'-"
        ):
            out.append(char)
        elif kind != "M" and out and out[-1] != " ":
            out.append(" ")
    return ud.normalize("NFC", "".join(out).strip())


def graphemes(text):
    tokens = []
    for char in normalize(text):
        if ud.category(char).startswith("M"):
            tokens[-1] += char
        else:
            tokens.append(char)
    return tokens


class Vocabulary:
    def __init__(self, units):
        self.units = list(units)
        self.ids = {token: i for i, token in enumerate(self.units) if i != 0}

    @classmethod
    def from_texts(cls, texts):
        return cls(["<blank>", *sorted({unit for text in texts for unit in graphemes(text)})])

    def encode(self, text):
        return [self.ids[unit] for unit in graphemes(text)]

    def render(self, ids):
        return "".join(self.units[i] for i in ids).strip()
