from src.core.job.reducer import Reducer


class WordCountReducer(Reducer):
    def reduce(self, key, values, emit):
        emit(key, sum(values))


