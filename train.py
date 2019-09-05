import torch
import environment
import utils
import model
import argparse
import header
import numpy as np
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import card
import os


def plot_fig(x, y, xlabel='epoch', ylabel='loss', filename='training_jpg'):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Training Progress')
    plt.plot(x, y)
    plt.savefig(filename)
    plt.clf()

def train(args):
    #init environment
    env = environment.TexasHoldem(log=args.save_to_environment, save_path=os.path.abspath(os.path.join(args.save_prefix, 'environment.log')))
    
    #init policy gradient network
    if args.use_pg == True:
        pg = model.PolicyGradient()
    else:
        pg = model.SimplePolicyGradient()
    if args.model_name != 'None':
        pg.load_state_dict(torch.load(args.model_name))
    if args.use_cuda == True:
        pg = pg.cuda()
    #init rule based AI
    rbAI = utils.RuleBasedAI()
    #rbAI_my_sight = utils.RuleBasedAI()
    #rbAI = utils.WinRateAI()
    if args.supervise_train == True:
        rbAI_my_sight = utils.SimpleAI()
    
    ct = card.CardTrick()

    #init optimizer
    optimizer = torch.optim.Adam(pg.parameters(), lr=args.learning_rate)
    
    #loss function
    loss_func = torch.nn.CrossEntropyLoss()
    collect_rewards = []
    collect_loss = []
    collect_epoch = []
    collect_pred_loss = []
    count_continous_fold = 0
    log_save_path = os.path.abspath(os.path.join(args.save_prefix,args.log))
    loss_save_path = os.path.abspath(os.path.join(args.save_prefix,'loss.jpg'))
    reward_save_path = os.path.abspath(os.path.join(args.save_prefix,'reward.jpg'))
    pred_loss_save_path = os.path.abspath(os.path.join(args.save_prefix,'pred_loss.jpg'))
    model_save_path = os.path.abspath(os.path.join(args.save_prefix, 'model.pkl'))

    for epoch in range(args.epoch):
        #print('epoch', epoch)
        count_reward = 0
        count_loss = torch.zeros(1)
        count_pred_loss = torch.zeros(1)
        if args.use_cuda:
            count_loss, count_pred_loss = count_loss.cuda(), count_pred_loss.cuda()
        for game in range(args.game_number):
            #print('game', game)
            #start training
            observation, start_turn = env.reset()
            time_step = observation[0]#time_step[0]=count turn, time_step[1]=whose call
            h_state = [None]*2
            #action_prob = []
            loss = torch.zeros(1).type(torch.float)
            action_loss = torch.zeros(1).type(torch.float)
            if args.use_cuda == True:
                loss, action_loss = loss.cuda(), action_loss.cuda()
            done = False
            my_have_move = False
            my_move_time, my_pred_time = 0, 0
            pred_action_prob = None
            while True:#run until game over
                pre_time_step = np.copy(time_step)
                while pre_time_step[0] == time_step[0]:#while the turn is not over
                    #print(time_step)
                    env.render()
                    if time_step[1] % 2 == (start_turn+1)/2:
                        ai_action = rbAI.choose_action(observation)
                        #print("AI", int(ai_action))
                        observation, reward, done, info = env.step(header.ai_turn, ai_action)
                        if args.supervise_train == False and pred_action_prob is not None:
                            ai_action = torch.tensor(ai_action).view(1).type(torch.long)
                            if args.use_cuda == True:
                                ai_action = ai_action.cuda()
                            if args.pretrain == False:
                                this_pred_loss = loss_func(pred_action_prob.view(1,-1), ai_action)
                                loss += this_pred_loss
                                count_pred_loss += this_pred_loss
                                my_pred_time += 1
                    else:
                        my_have_move = True
                        my_move_time += 1
                        if args.supervise_train == True:
                            correct_action = rbAI_my_sight.choose_action(observation)
                            correct_action = torch.tensor(correct_action).type(torch.long).view(1)
                            if args.use_cuda == True:
                                correct_action = correct_action.cuda()
                        input_time_step = Variable(torch.FloatTensor(observation[0]).view(1,2))
                        my_cards = Variable(torch.LongTensor(observation[1]).view(1, 2))
                        table_cards = Variable(torch.LongTensor(observation[2]).view(1,5))
                        actions = Variable(torch.LongTensor(observation[3]).view(1,2))
                        action_mask = Variable(torch.FloatTensor(observation[4]).view(1,12))
                        bet_info = Variable(torch.FloatTensor(observation[5]).view(1, header.total_bet_info))
                        card_info = Variable(torch.FloatTensor(ct.get_feature(observation[1], observation[2])).view(1, header.total_card_feature))
                        if args.use_cuda == True:
                            input_time_step, my_cards, table_cards, actions, action_mask, bet_info, card_info = input_time_step.cuda(), my_cards.cuda(), table_cards.cuda(), actions.cuda(), action_mask.cuda(), bet_info.cuda(), card_info.cuda()

                        my_action_prob, pred_action_prob, h_state = pg.choose_action(input_time_step, my_cards, table_cards, actions, action_mask, bet_info, card_info, *h_state, pre_train=args.pretrain, use_cuda=args.use_cuda)
                        
                        if args.supervise_train == False:
                            m = torch.distributions.Multinomial(probs=my_action_prob)
                            my_action = m.sample()
                            ret_action = int(torch.max(my_action, 0)[1])
                            if ret_action == 0:#punish continous fold and high winrate fold
                                punish_win_rate_coef = torch.exp(torch.max(torch.tensor([0.0, card_info[0,13]-0.5])))
                                punish_coef = torch.max(torch.tensor([(count_continous_fold/10) * punish_win_rate_coef, punish_win_rate_coef]))
                                count_continous_fold += 1
                            else:
                                punish_coef = torch.tensor([1.0])
                                count_continous_fold = 0
                            punish_coef = punish_coef.cuda() if args.use_cuda else punish_coef
                            action_loss += -m.log_prob(my_action)*punish_coef
                        #print("MY", my_action_prob)
                        #print("MY", int(correct_action))
                        ret_action = int(correct_action) if args.supervise_train == True else ret_action
                        observation, reward, done, info = env.step(header.my_turn, ret_action)
                        #print(my_action_prob, correct_action)
                        if args.supervise_train == True:
                            loss += loss_func(my_action_prob.view(1, -1), correct_action)
                    time_step = np.copy(observation[0])#update time_step from observation
                    #print(time_step)
                if done == True:
                    break
            if my_have_move == True:
                optimizer.zero_grad()
                if args.supervise_train == True:
                    loss = loss / my_move_time
                else:
                    loss = loss / my_pred_time if my_pred_time != 0 else loss
                    count_pred_loss = count_pred_loss / my_pred_time if my_pred_time != 0 else count_pred_loss
                    action_loss = action_loss * reward
                    loss += action_loss
                loss.backward()
                optimizer.step()
                count_loss += loss
            count_reward += reward
        with open(log_save_path, 'a') as f:
            print('%d, %.5f, %.5f, %.5f' % (epoch+int(args.epoch_prefix), count_loss/args.game_number, count_reward/args.game_number, count_pred_loss), file=f)

        print('epoch %d loss: %.3f' % (epoch+int(args.epoch_prefix), count_loss)) 
        print('total rewards after %d rounds: %d' % (args.game_number, count_reward))
        collect_loss.append(count_loss/args.game_number)
        collect_rewards.append(count_reward/args.game_number)
        collect_epoch.append(epoch+int(args.epoch_prefix))
        collect_pred_loss.append(count_pred_loss)
        plot_fig(collect_epoch, collect_loss, ylabel='avg loss', filename=loss_save_path)
        plot_fig(collect_epoch, collect_rewards, ylabel='avg reward', filename=reward_save_path)
        plot_fig(collect_epoch, collect_pred_loss, ylabel='avg pred loss', filename=pred_loss_save_path)
        torch.save(pg.state_dict(), model_save_path)


if __name__ == '__main__':
    def str2bool(v):
        if v == 'True':
            return True
        else:
            return False
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('--model_name', help='Load model parameters', default='model.pkl', type=str)
    parser.add_argument('-gn', '--game_number', help='Total games to train with', default=100, type=int)
    parser.add_argument('--pretrain', help='pretrain model with rnn_my_value without rnn_pred_value', default=False, type=str2bool)
    parser.add_argument('--supervise_train', help='Train parameters with supervision of rule-based AI', default=False, type=str2bool)
    parser.add_argument('-e', '--epoch', help='total epoch for training supervisive learning', default=1000, type=int)
    parser.add_argument('--use_cuda', help='Whether to use cuda', default=header.use_cuda, type=str2bool)
    parser.add_argument('--log', help='Write record to the log document', default='train.log', type=str)
    parser.add_argument('--epoch_prefix', help='Epoches that has trained', default=0, type=int)
    parser.add_argument('--use_pg', help='Which model to use? PolicyGradient or SimplePolicyGradient', default=True, type=str2bool)
    #parser.add_argument('--plot_filename', help='prefix name of figure to plot', default='training_', type=str)
    parser.add_argument('--save_prefix', help='Create a directory and save all the file there.', default='.', type=str)
    parser.add_argument('--save_to_environment', help='save environment file or not', default=True, type=str2bool)
    args = parser.parse_args()
    
    train(args)
