import numpy as np
import header
from player import Player
import utils

class TexasHoldem():
    def __init__(self, log=True, save_path='./environment.log'):
        #init environment
        #state: time step + player1's cards *2 + player2's cards *2 + table cards *5 + previous actions
        self.game_number = 0
        self.turn = -1 
        #self.pot_maximum = 1000
        self.small_blind = 20
        self.big_blind = 40
        self.initial_money = 1000
        self.ai = utils.RuleBasedAI() 
        self.log = log
        if(log):
            self.log_file = open(save_path, "w")

    def print_log(self, player_turn=None, action=None, check_or_call=None, winner=None):
        def cards_string(cards):
            color_set = ["Spade", "Heart", "Diamond", "Club"]
            s = ""
            for card in cards:
                if card == 52:
                    s += "%9s " % ("Fold".center(9))
                else:
                    color = int(card / 13)
                    num = int(card % 13) + 1
                    s += "%9s " % ((color_set[color]+str(num)).center(9))
            return s
        
        s = "%51s\n" % ("Game: %d, Time Steps: (%d, %d)" % 
                         (self.game_number, self.time_step[0], self.time_step[1])).center(51)
        s += "%51s\n" % (("Pot: "+str(self.pot)).center(51))
        s += "%51s\n" % ("Table Cards".center(51))
        s += "%51s\n" % ("-"*51)
        s += "%51s\n" % (cards_string(self.observed_table_cards).center(51))
        s += "%51s\n" % ("-"*51)
        s += "%25s|%25s\n" % ("Player1".center(25), "Player2".center(25))
        s += "%25s|" % (("Money: "+str(self.player1.money)).center(25))
        s += "%25s\n" % (("Money: "+str(self.player2.money)).center(25))
        s += "%25s|%25s\n" % ("Cards".center(25), "Cards".center(25))
        s += "%51s\n" % ("-"*51)
        s += "%25s|%25s\n" % (cards_string(self.player1.cards).center(25), 
                              cards_string(self.player2.cards).center(25))
        s += "%51s\n" % ("-"*51)
        if(player_turn == None):
            s += "%51s\n" % ("---Start---".center(51))
        if(player_turn != None):
            if player_turn == header.my_turn:
                player_name = "Player1"
            else:
                player_name = "Player2"
            if action == 0:
                action_name = "Fold"
            elif action == 1:
                if check_or_call:
                    action_name = "Check"
                else:
                    action_name = "Call"
            elif action > 1 and action < 11:
                action_name = "Raise "+str(10*(action-1))+"%"
            else:
                action_name = "All in"
            s += "%51s\n" % (("---"+player_name+" "+action_name+"---").center(51))
        s += "\n"
        if(winner != None):
            if winner == 1:
                info = "Player1 Win!"
            else:
                info = "Player2 Win!"
            s += "%51s\n\n" %(("-"*19+info+"-"*19).center(51))
        print(s, file=self.log_file)
 
    def reset(self):
        #reset a game
        self.game_number += 1
        self.turn *= -1 #1: player1, -1: player2
        self.cur_turn = self.turn
        perm = np.arange(52)
        np.random.shuffle(perm)
        self.player1 = Player(id = 1, money=self.initial_money, cards=perm[:2])
        self.player2 = Player(id = -1, money=self.initial_money, cards=perm[2:4])
        self.table_cards = perm[4:9]
        self.observed_table_cards = np.full(5, 52)
        self.pot = 0
        self.last_raise_amount = self.big_blind
        self.time_step = np.array([0, 0]) #index 0: # of round in one game, index 1: # of action in one round
        #bet_amount only for one turn
        self.valid_action = np.ones(12, dtype=int)
        if self.turn == 1: #next: player1
            actor = self.player2
            opponent = self.player1
        else: #next: player2
            actor = self.player1
            opponent = self.player2
        actor.bet(self.big_blind)
        opponent.bet(self.small_blind)
        actions = np.array((opponent.actions[-1], actor.actions[-1]))
        bet_info = np.array([opponent.money, actor.money, opponent.bet_amount, actor.bet_amount, self.pot])
        observation = [self.time_step, 
                       opponent.cards, 
                       self.observed_table_cards, 
                       actions,
                       self.valid_action,
                       bet_info]
        self.print_log()
        return [observation, self.turn]
        
    def render(self):
        #update environment
        return

    def step(self, player_turn, action):
        #return observation([time_step, player_cards, table_cards*5, previous_action]), reward, done, info
        if(self.cur_turn != player_turn):
            raise Exception('Wrong Turn')
        if self.valid_action[action] == 0:
            raise Exception('ACTION ERROR')
        if player_turn == header.my_turn:
            actor = self.player1
            opponent = self.player2
        else:
            actor = self.player2
            opponent = self.player1

        done = False
        round_ends = False
        winner = None
        reward = 0
        info = None
        check_or_call = None #check:True, call: False, Only valid when action=1
        if action == 0: # FOLD
            round_ends = True
            done = True
            winner = opponent.id
        elif action == 1: #CHECK/CALL
            if opponent.bet_amount > 0: #CALL
                check_or_call = False
                actor.bet(opponent.bet_amount - actor.bet_amount)
            else: #CHECK
                check_or_call = True
            # In each round, after the first acts, this round ends if one check or call.
            if self.time_step[1] > 0: 
                round_ends = True
        elif action > 1 and action < 11: #Raise based on bet rate
            max_raise_amount = np.min([actor.money - (opponent.bet_amount-actor.bet_amount), opponent.money])
            #Based on no-limit Texas hold'em rule, raise_amount must be no less than last_raise_amount 
            diff = max_raise_amount - self.last_raise_amount
            if diff < 0:
                raise Exception('ACTION ERROR')
            #10 choices(actions), linear scale raise amount
            raise_amount = self.last_raise_amount + np.floor((action - 1) * 0.1 * diff)
            actor.bet(opponent.bet_amount + raise_amount - actor.bet_amount)
            self.last_raise_amount = raise_amount
        elif action == 11:
            raise_amount = np.min([actor.money - (opponent.bet_amount-actor.bet_amount), opponent.money])
            actor.bet(opponent.bet_amount + raise_amount - actor.bet_amount)   
        else: 
            raise Exception('ACTION ERROR')
        actor.actions = np.append(actor.actions, action)
        #next: for opponent
        #Check valid action for next players
        valid_raise_amount = opponent.money - (actor.bet_amount - opponent.bet_amount)
        if action == 11:
            self.valid_action = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif valid_raise_amount <= self.last_raise_amount:
            self.valid_action = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        else:
            self.valid_action = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            for i in range(10):
                self.valid_action[i+2] = 0 if int(valid_raise_amount*(i+1)*0.1) == 0 else 1
        actions = np.array((opponent.actions[-1], actor.actions[-1]))
        bet_info = np.array([opponent.money, actor.money, opponent.bet_amount, actor.bet_amount, self.pot])
        observation = [self.time_step, 
                       opponent.cards, 
                       self.observed_table_cards, 
                       actions,
                       self.valid_action,
                       bet_info]
        
        self.time_step[1] += 1
        self.cur_turn *= -1
        if(round_ends):
            self.time_step[0] += 1
            self.time_step[1] = 0
            self.cur_turn = self.turn
            self.pot += actor.bet_amount + opponent.bet_amount
            actor.bet_amount = 0
            opponent.bet_amount = 0
            self.last_raise_amount = self.big_blind
            if(self.time_step[0] >= 4): 
                done = True
            if(actor.money == 0 or opponent.money == 0):
                done = True
        
        if(done):
            if(winner == None):
                #judge the winners
                if self.ai.score7(np.concatenate((self.player1.cards, self.observed_table_cards)).tolist()) > self.ai.score7(np.concatenate((self.player2.cards, self.observed_table_cards)).tolist()):
                    winner = 1
                    info = "Player 1 win!"
                else:
                    winner = -1
                    info = "Player 2 win!"
            if(winner == 1):
                info = "Player 1 win!"
                reward = self.pot + self.player1.money - self.initial_money
            else:
                info = "Player 2 win!"
                reward = self.player1.money - self.initial_money
        
        else:
            if self.time_step[0] > 0 and self.time_step[0] <= 4:
                self.observed_table_cards[:2+self.time_step[0]] = self.table_cards[:2+self.time_step[0]]
        if(self.log):
            self.print_log(player_turn, action, check_or_call, winner)
        return observation, float(reward), done, info
 
