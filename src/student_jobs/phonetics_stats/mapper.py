from src.core.job.mapper import Mapper

_VOWELS = set("aeiouyаеєиіїоуюя")
_IGNORE = set("ь")

def _count_vowels_consonants(token: str) -> tuple[int, int, int]:
    v = c = 0
    for ch in token.lower():
        if not ch.isalpha():
            continue
        if ch in _IGNORE:
            continue
        if ch in _VOWELS:
            v += 1
        else:
            c += 1
    length = v + c
    return v, c, length

class PhoneticsStatsMapper(Mapper):
    def map(self, record, emit):
        text = "".join(ch if ch.isalpha() else " " for ch in str(record))
        for w in text.split():
            v, c, length = _count_vowels_consonants(w)
            if length > 0:
                emit(length, (v, c))
