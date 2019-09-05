import utils
import numpy as np
import random

class CardTrick():
    def __init__(self):
        self.ai = utils.RuleBasedAI()
    def sample_win_rate(self, my_cards, table_cards, sample_time=100):
        count_win = 0
        for time in range(sample_time):
            temp_table_cards = np.copy(table_cards)
            temp_my_cards = np.copy(my_cards)
            all_cards = np.concatenate((temp_my_cards, temp_table_cards, np.full(2, 52)))
            appear_flag = [False] * 53
            for i in all_cards:
                appear_flag[i] = True
            for i in range(9):
                if all_cards[i] == 52:
                    while True:
                        rand_num = random.randint(0,51)
                        if appear_flag[rand_num] == False:
                            appear_flag[rand_num] = True
                            break
                    all_cards[i] = rand_num

            if self.ai.score7(all_cards[:7].tolist()) > self.ai.score7(all_cards[2:].tolist()):
                count_win += 1
        return float(count_win)/float(sample_time)
    
    def get_feature(self, my_cards, table_cards):
        #count_same[2,3,4], count_continous[2,3,4,5,6,7], count_suit[spade,heart,diamond,club], win_rate
        def _count_same(cards):
            count = np.zeros(13)
            for card in cards:
                if card != 52:
                    count[card%13] += 1
            same = np.zeros(3)
            for count_num in count[:-1]:
                if count_num >= 2:
                    same[int(count_num)-2] += 1
            return same
        def _count_continous(cards):
            sorted_cards = np.zeros(13)
            for card in np.sort(cards):
                if card != 52:
                    sorted_cards[card%13] += 1
            count_continous = np.zeros(6)
            for continous in range(2,8):
                for start_index in range(0,14-continous):
                    is_conti = True
                    for conti in range(start_index, start_index+continous-1):
                        if sorted_cards[conti]+1 == 0 or sorted_cards[conti+1] == 0:
                            is_conti = False
                            break
                    if is_conti:
                        count_continous[continous-2] += 1
            return count_continous
        def _count_suit(cards):
            suit = np.zeros(4)
            for card in cards:
                if 0 <= card <= 12:
                    suit[0] += 1
                elif 13 <= card <= 25:
                    suit[1] += 1
                elif 26 <= card <= 38:
                    suit[2] += 1
                elif 39 <= card <= 51:
                    suit[3] += 1
            return suit
        all_cards = np.concatenate((my_cards, table_cards))
        win_rate = np.zeros(1)
        for i in range(10):
            win_rate[0] += self.ai.sample_win_rate(my_cards, table_cards)
        win_rate[0] /= float(10*1000)
        ret = np.concatenate((_count_same(all_cards), _count_continous(all_cards), _count_suit(all_cards), win_rate))
        return ret
        



if __name__ == '__main__':
    #test class CardTrick
    ct = CardTrick()
    a = np.array([52,0])
    b = np.array([1,52,3,44,17])
    print(ct.sample_win_rate(a, b))
    print(ct.ai.sample_win_rate(a, b))
    for i in range(1000):
        ct.get_feature(a,b)
        ct.ai.sample_win_rate(a, b)
