import util
import wordsegUtil


class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def start_state(self):
        return 0, wordsegUtil.SENTENCE_BEGIN  # position before which text is reconstructed & previous word

    def is_end(self, state):
        return state[0] == len(self.query)

    def succ_and_cost(self, state):
        #pos: 모음이 삽입된 단어의 갯수, prev_word: 모음 삽입하기 이전 단어 
        # query : 문장 
        pos, prev_word = state
        for i in range(1, len(self.query) - pos + 1):
            word = self.query[pos : pos + i]
            fills = self.possibleFills(word)  # 내가 가지고 있는 모음이 제외된 단어들 중 만들 수 있는 모든 단어의 경우의 수를 반환하는 함수
            for fill in fills:
                next_state = pos + i, fill
                cost = self.bigramCost(prev_word, fill)
                yield fill, next_state, cost  # return action, state, cost



if __name__ == '__main__':
    unigramCost, bigramCost = wordsegUtil.makeLanguageModels('leo-will.txt')
    smoothCost = wordsegUtil.smoothUnigramAndBigram(unigramCost, bigramCost, 0.2)
    possibleFills = wordsegUtil.makeInverseRemovalDictionary('leo-will.txt', 'aeiou')
    problem = JointSegmentationInsertionProblem('mgnllthppl', smoothCost, possibleFills)
    # problem = JointSegmentationInsertionProblem(wordsegUtil.removeAll('whatsup', 'aeiou'), smoothCost, possibleFills)

    # import dynamic_programming_search
    # dps = dynamic_programming_search.DynamicProgrammingSearch(verbose=1)
    # # dps = dynamic_programming_search.DynamicProgrammingSearch(memory_use=False, verbose=1)
    # print(dps.solve(problem))

    import uniform_cost_search
    ucs = uniform_cost_search.UniformCostSearch(verbose=0)
    print(ucs.solve(problem))


# === Other Examples ===
# 
# QUERIES_BOTH = [
#     'stff',
#     'hllthr',
#     'thffcrndprncndrw',
#     'thstffffcrndprncndrwmntdthrhrssndrdn',
#     'whatsup',
#     'ipovercarrierpigeon',
#     'aeronauticalengineering',
#     'themanwiththegoldeneyeball',
#     'lightbulbsneedchange',
#     'internationalplease',
#     'comevisitnaples',
#     'somethingintheway',
#     'itselementarymydearwatson',
#     'itselementarymyqueen',
#     'themanandthewoman',
#     'nghlrdy',
#     'jointmodelingworks',
#     'jointmodelingworkssometimes',
#     'jointmodelingsometimesworks',
#     'rtfclntllgnc',
# ]
