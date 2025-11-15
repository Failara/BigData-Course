from src.core.job.mapper import Mapper

class LongWordCountMapper(Mapper):
    def map(self, record, emit):
        text = "".join(ch if ch.isalpha() else " " for ch in str(record))
        total = sum(1 for w in text.split() if len(w) > 5)
        if total:
            emit("long_words_len_gt_5", total)
