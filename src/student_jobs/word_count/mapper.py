from src.core.job.mapper import Mapper


class WordCountMapper(Mapper):
    def map(self, record, emit):
        text = "".join(ch if ch.isalpha() else " " for ch in str(record))
        for token in text.split():
            emit(token.lower(), 1)

