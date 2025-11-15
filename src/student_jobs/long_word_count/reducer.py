from src.core.job.reducer import Reducer

class LongWordCountReducer(Reducer):
    def reduce(self, key, values, emit):
        emit(key, sum(values))
