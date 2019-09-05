import numpy as np
import header

class Player():
    def __init__(self, id, money, cards):
        #init player
        self.id = id
        self.cards = cards
        self.actions = [0] #initial action defined as zero
        self.money = money
        self.bet_amount = 0
        
    def bet(self, cur_bet):
        if self.money < cur_bet:
            raise Exception('Action Error, money insufficient')
        self.bet_amount += cur_bet
        self.money -= cur_bet