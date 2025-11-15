from src.core.job.reducer import Reducer

class PhoneticsStatsReducer(Reducer):
    def reduce(self, key, values, emit):
        total_v = total_c = 0
        for v, c in values:
            total_v += v
            total_c += c
        total = total_v + total_c
        if total == 0:
            return
        p_v = 100.0 * total_v / total
        p_c = 100.0 - p_v
        emit(key, f"vowels={p_v:.2f}%, consonants={p_c:.2f}%")
