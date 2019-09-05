import header
from ctypes import *

class RuleBasedAI():
    def __init__(self):
        #init the rule-based AI
        self.oldman=cdll.LoadLibrary('./oldman.so')
        self.oldman.InitPokerAI()

    def score7(self, cards):
        import numpy as np
        n = np.int64(0)
        for card in cards:
            if (card >= 0 and card < 52) :
                n |= ((np.int64(1)<<(1+(card+12)%13)) << (int(card/13)*16))
        #print(hex(n))
        return self.oldman.score7(int(n>>32), int(n&((1<<32)-1)))

    def sample_win_rate(self, my_cards, table_cards):
        import numpy as np
        my, table = np.int64(0), np.int64(0)
        for card in my_cards:
            if (card >= 0 and card < 52) :
                my |= ((np.int64(1)<<(1+(card+12)%13)) << (int(card/13)*16))
        for card in table_cards:
            if (card >= 0 and card < 52) :
                table |= ((np.int64(1)<<(1+(card+12)%13)) << (int(card/13)*16))
        #print(hex(int(my)),hex(int(table)))
        return self.oldman.goWinRate(int(my>>32), int(my&((1<<32)-1)), int(table>>32), int(table&((1<<32)-1)) )


    def choose_action(self, observation):
        import numpy as np
        myCard=observation[1]
        len=myCard.shape[0]
        n = np.int64(0)
        for i in range(len):
            card=myCard[i]
            if (card >= 0 and card < 52) :
                n |= ((np.int64(1)<<(1+(card+12)%13)) << (int(card/13)*16))
        tableCard=observation[2]
        len=tableCard.shape[0]
        m = np.int64(0)
        for i in range(len):
            card=tableCard[i]
            if (card >= 0 and card < 52) :
                m |= ((np.int64(1)<<(1+(card+12)%13)) << (int(card/13)*16))

        time_step = observation[0]
        newHand = 0;
        if (time_step[0] == 0 and time_step[1] < 2):
            newHand = 1

        mask = observation[4]
        betInfo=observation[5]
        myMoney=betInfo[0]
        myBet=betInfo[2]
        hisBet=betInfo[3]
        pot=betInfo[4]
        maxBet=myMoney+myBet+pot/2

        #print( hex(n), hex(m), myMoney, myBet, hisBet, pot, maxBet )

        bet=self.oldman.goThink(int(n>>32), int(n&((1<<32)-1)), int(m>>32), int(m&((1<<32)-1)), int((myBet*100)/maxBet), int((hisBet*100)/maxBet), int((pot*100)/maxBet), newHand);

        #print( bet )

        if (bet<=0) :
            bet = bet + 1
        else :
            bet = maxBet * bet / 100 - max(myBet,hisBet)
            if ( bet <= 0 or (maxBet - pot/2 - max(myBet,hisBet) ) <= 0):
                bet = 1
            else:
                bet = int(round((bet*10) / (maxBet - pot/2 - max(myBet,hisBet) )) + 1)
            #bet = bet + int((pot*100)/maxBet)
            #bet = round(bet/10) + 1
        if (bet>11) :
            bet = 11
        while mask[bet] == 0:
            bet = (bet-1) % header.total_action
        return bet

class WinRateAI():
    def __init__(self):
        import card
        self.ai = RuleBasedAI()
        self.simple_ai = SimpleAI()
        self.ct = card.CardTrick()
    def choose_action(self, observation):
        valid_mask = observation[4]
        time_step = observation[0]
        win_rate = self.ai.sample_win_rate(observation[1], observation[2])/1000
        if time_step[0] == 0:
            if win_rate > 0.1:
                return 1
            else:
                return 0
        elif time_step[0] == 1:
            if win_rate < 0.4:
                return 0
            if win_rate > 0.9:
                return 11 if valid_mask[11] == 1 else 1
            else:
                return self.simple_ai.return_valid(2, valid_mask)
        else:
            if win_rate > 0.9:
                return 11 if valid_mask[11] == 1 else 1
            elif win_rate > 0.8:
                return self.simple_ai.return_valid(7, valid_mask)
            elif win_rate > 0.7:
                return self.simple_ai.return_valid(3, valid_mask)
            else:
                return 1

class SimpleAI():
    def __init__(self):
        pass
    def choose_action(self, observation):
        import numpy
        valid_mask = observation[4]
        time_step = observation[0]
        if time_step[0] == 0:
            if self.count_same_card(observation[1]) == 1:
                return self.return_valid(5, valid_mask)
            elif sum(observation[1]%13) > 10:
                return self.return_valid(1, valid_mask)
            else:
                return self.return_valid(0, valid_mask)
        elif time_step[0] == 1:
            count = self.count_same_card(numpy.concatenate((observation[1], observation[2]), axis=0))
            if count == 1:
                return self.return_valid(3, valid_mask)
            elif count == 2:
                return self.return_valid(11, valid_mask)
            else:
                return self.return_valid(1, valid_mask)
        else:
            return self.return_valid(1, valid_mask)
    def count_same_card(self, cards):
        length = cards.shape[0]
        count = 0
        for i in range(length-1):
            for j in range(i+1,length):
                if cards[i]%13 == cards[j]%13 and cards[i] != 52 and cards[j] != 52:
                    count += 1

        return count
    def return_valid(self, number, mask):
        while mask[number] == 0:
            number = (number+1) % header.total_action
        return number


'''
import numpy as np
ai = RuleBasedAI()
print(ai.sample_win_rate([0,13],[52, 52, 52, 52, 52]))
print(ai.sample_win_rate([12,38],[52, 52, 52, 52, 52]))
print(ai.sample_win_rate([12,38],[51, 0, 1, 52, 52]))
print(ai.sample_win_rate([1,2],[4, 6, 8, 52, 52]))
print(ai.sample_win_rate([0,12],[11, 10, 9, 52, 52]))
print(ai.sample_win_rate([0,12],[11, 27, 30, 52, 52]))

#cards=[1,14,26,29,40,42,44]
#print(ai.score7(cards)>>27)
obs = [np.array([0,1]), np.array([25,27]), np.array([52, 52, 52, 52, 52]), np.array([1,1]), np.array([1,1,1,1,1,1,1,1,1,1,1,1]), np.array([1000,1000,0,500,0])]
print(ai.choose_action(obs))
obs = [np.array([0,1]), np.array([1,12]), np.array([13, 25, 31, 52, 52]), np.array([1,1]), np.array([1,1,1,1,1,1,1,1,1,1,1,1]), np.array([1000,1000,0,500,0])]
print(ai.choose_action(obs))
obs = [np.array([0,1]), np.array([1,12]), np.array([13, 25, 31, 52, 52]), np.array([1,1]), np.array([1,1,0,0,0,0,0,0,0,0,0,1]), np.array([1000,1000,0,0,500])]
print(ai.choose_action(obs))
obs = [np.array([0,1]), np.array([8,9]), np.array([10, 11, 12, 42, 44]), np.array([1,1]), np.array([1,1,1,1,1,1,1,1,1,1,1,1]), np.array([1000,1000,0,500,0])]
print(ai.choose_action(obs))


import numpy as np
ai = SimpleAI()
obs = [[1,0], np.array([11,13]), np.array([2, 12, 4, 52, 52]), [1,1]]
print(ai.choose_action(obs))
'''
