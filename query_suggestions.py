class QuerySuggestions:
    def __init__(self, logfile: str = 'data/queryCount.csv'):
        self.query_log: dict = self._read_log_into_dict(logfile)
    
    def _read_log_into_dict(self, log: str) -> dict:
        out={}
        with open(log,'r') as f:
            data = f.read()
        lines = data.split('\n')
        for line in lines:
            query, count = line.split(',')
            out[query] = int(count)
        return out
    
    def get_candidates(self, query: str = '')-> dict:
        query = query.lower()
        candidates: dict = dict(filter(lambda item: query in item[0], self.query_log.items())) # {'one book': 3, 'one house': 1}
        return candidates
    
    def calc_score(self, candidate_sessions: float, total_sessions: float) -> float:
        return candidate_sessions/total_sessions

    def rank_candidates(self, candidates: dict)-> dict:
        total_sessions = sum(candidates.values())
        score_dict = dict(map(lambda x: (x[0], self.calc_score(x[1], total_sessions)), candidates.items()))
        print('score_dict', score_dict)
        score_dict_sorted = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print('score_dict_sorted', score_dict_sorted)
        return score_dict_sorted

    def run(self, query) -> list:
        rankings: dict = self.rank_candidates(self.get_candidates(query))
        return [item[0] for item in rankings]