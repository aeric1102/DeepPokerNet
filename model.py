import torch
import header
import numpy as np

class PolicyGradient(torch.nn.Module):
    def __init__(self, value_dim=64, hidden_dim=128, embedding_dim=32, n_card=52):
        super(PolicyGradient, self).__init__()
        self.n_card = header.total_my_card + header.total_table_card
        self.n_action = 2
        self.n_output_action = header.total_action
        self.action_index = np.zeros(self.n_output_action)
        for i in range(self.n_output_action):
            self.action_index[i] = i
        self.time_step = header.total_time_step
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.rnn_output_dim = value_dim
        self.rnn_input_dim = self.time_step + embedding_dim*self.n_card + self.n_action*embedding_dim + self.n_output_action + header.total_bet_info + header.total_card_feature
        self.rnn_main_input_dim = self.rnn_input_dim + self.n_output_action
        self._build_net()
    def _build_net(self):
        self.my_embedding = torch.nn.Embedding(header.total_card_type, self.embedding_dim)
        self.table_embedding = torch.nn.Embedding(header.total_card_type, self.embedding_dim)
        self.action_embedding = torch.nn.Embedding(self.n_output_action, self.embedding_dim)
        self.rnn_main = torch.nn.GRU(
                    input_size=self.rnn_main_input_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=2,
                    batch_first=True
                )
        self.nn_output = torch.nn.Linear(self.hidden_dim, self.n_output_action)
        self.rnn_pred_action = torch.nn.GRU(
                    input_size=self.rnn_input_dim,
                    hidden_size=self.rnn_output_dim,
                    num_layers=2,
                    batch_first=True
                )
        self.nn_pred_action = torch.nn.Linear(self.rnn_output_dim, self.n_output_action)
        self.softmax = torch.nn.functional.softmax
    def choose_action(self, time_step, my_cards, table_cards, actions, action_mask, bet_info, card_info, h_state_rnn_main=None, h_state_rnn_pred=None, pre_train=True, use_cuda=header.use_cuda):
        #time_step=(B,2), my_cards=(B,2), table_cards=(B,5), actions=(B,2)
        #bet_info=(B,5), card_info=(B,14)
        my_cards = self.my_embedding(my_cards).view(1, -1)#(B, 2, embedding_dim)->(B, -1)
        table_cards = self.table_embedding(table_cards).view(1, -1)
        actions = self.action_embedding(actions).view(1, -1)
        rnn_my_pred_input = torch.cat((time_step, my_cards, table_cards, actions, action_mask, bet_info, card_info), dim=1).view(1,1,-1) #(B, 1, self.rnn_input_dim)
        if pre_train == True:
            pred_action_prob = torch.zeros(1, 1, self.n_output_action)
            if use_cuda == True:
                pred_action_prob = pred_action_prob.cuda()
        else:
            pred_value, h_state_rnn_pred = self.rnn_pred_action(rnn_my_pred_input, h_state_rnn_pred)
            pred_action = self.nn_pred_action(pred_value)
            pred_action_prob = self.softmax(pred_action, dim=2)
        rnn_main_input = torch.cat((rnn_my_pred_input, pred_action_prob), dim=2)
        output, h_state_rnn_main = self.rnn_main(rnn_main_input, h_state_rnn_main)
        output = self.nn_output(output).view(self.n_output_action)
        my_action_prob = self.softmax(output, dim=0)
        my_action_prob = (my_action_prob*action_mask).view(-1)
        '''
        byte_tensor_mask = action_mask.view(-1).type(torch.ByteTensor)
        masked_action_index = self.action_index[byte_tensor_mask]
        masked_action = output[byte_tensor_mask]
        masked_action = self.softmax(masked_action, dim=0)
        if masked_action.shape[0] == self.n_output_action:
            my_action_prob = masked_action
        else:
            my_action_prob = action_mask.view(-1)
            temp_masked_index = 0
            for i, _ in enumerate(my_action_prob):
                if my_action_prob[i] == 1:
                    my_action_prob[i] = masked_action[temp_masked_index]
                    temp_masked_index += 1
        my_action = int(masked_action_index[int(torch.max(masked_action, 0)[1])])
        '''
        return my_action_prob, pred_action_prob, (h_state_rnn_main, h_state_rnn_pred)
    
    
class SimplePolicyGradient(torch.nn.Module):
    def __init__(self, value_dim=64, hidden_dim=128, embedding_dim=32, n_card=52):
        super(SimplePolicyGradient, self).__init__()
        self.n_card = header.total_my_card + header.total_table_card
        self.n_action = 2
        self.n_output_action = header.total_action
        self.action_index = np.zeros(self.n_output_action)
        for i in range(self.n_output_action):
            self.action_index[i] = i
        self.time_step = header.total_time_step
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.rnn_output_dim = value_dim
        self.rnn_input_dim = self.time_step + 1
        self.rnn_main_input_dim = self.rnn_input_dim + self.n_output_action
        self._build_net()
    def _build_net(self):
        self.rnn_main = torch.nn.GRU(
                    input_size=self.rnn_main_input_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=2,
                    batch_first=True
                )
        self.nn_output = torch.nn.Linear(self.hidden_dim, self.n_output_action)
        self.rnn_pred_action = torch.nn.GRU(
                    input_size=self.rnn_input_dim,
                    hidden_size=self.rnn_output_dim,
                    num_layers=2,
                    batch_first=True
                )
        self.nn_pred_action = torch.nn.Linear(self.rnn_output_dim, self.n_output_action)
        self.softmax = torch.nn.functional.softmax
    def choose_action(self, time_step, my_cards, table_cards, actions, action_mask, bet_info, card_info, h_state_rnn_main=None, h_state_rnn_pred=None, pre_train=True, use_cuda=header.use_cuda):
        #time_step=(B,2), my_cards=(B,2), table_cards=(B,5), actions=(B,2)
        #bet_info=(B,5), card_info=(B,14)
        card_info = card_info[0,-1].view(1, -1)#only need winrate
        rnn_my_pred_input = torch.cat((time_step, card_info), dim=1).view(1,1,-1) #(B, 1, self.rnn_input_dim)
        if pre_train == True:
            pred_action_prob = torch.autograd.Variable(torch.zeros(1, 1, self.n_output_action))
            if use_cuda == True:
                pred_action_prob = pred_action_prob.cuda()
        else:
            pred_value, h_state_rnn_pred = self.rnn_pred_action(rnn_my_pred_input, h_state_rnn_pred)
            pred_action = self.nn_pred_action(pred_value)
            pred_action_prob = self.softmax(pred_action, dim=2)
        rnn_main_input = torch.cat((rnn_my_pred_input, pred_action_prob), dim=2)
        output, h_state_rnn_main = self.rnn_main(rnn_main_input, h_state_rnn_main)
        output = self.nn_output(output).view(self.n_output_action)
        my_action_prob = self.softmax(output, dim=0)
        my_action_prob = (my_action_prob*action_mask).view(-1)
        #my_action = int(torch.max(my_action_prob, 0)[1])
        '''
        byte_tensor_mask = action_mask.view(-1).type(torch.ByteTensor)
        masked_action_index = self.action_index[byte_tensor_mask]
        masked_action = output[byte_tensor_mask]
        masked_action = self.softmax(masked_action, dim=0)
        if masked_action.shape[0] == self.n_output_action:
            my_action_prob = masked_action
        else:
            my_action_prob = action_mask.view(-1)
            temp_masked_index = 0
            for i, _ in enumerate(my_action_prob):
                if my_action_prob[i] == 1:
                    my_action_prob[i] = masked_action[temp_masked_index]
                    temp_masked_index += 1
        my_action = int(masked_action_index[int(torch.max(masked_action, 0)[1])])
        '''
        return my_action_prob, pred_action_prob, (h_state_rnn_main, h_state_rnn_pred)

