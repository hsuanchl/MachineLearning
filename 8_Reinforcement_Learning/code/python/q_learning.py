#install pyglet to render
#comment out render() if want to train faster
# example: python3 q_learning.py tile weight.out returns.out 10 200 0.05 0.99 0.01
from environment import MountainCar
import sys
import numpy as np

def main(args):
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])
    return mode,weight_out,returns_out,episodes,max_iterations,epsilon,gamma,learning_rate

def pick_action(state, w, b, epsilon):
    #epsilon greedy action selection
    if np.random.random() < 1 - epsilon:
        action = np.argmax(calc_q(state,w,b))
    else:
        action = np.random.randint(low=0,high=3)
    return action

def calc_q(state,w,b):
    sparse_product = 0
    for key,val in state.items():
        sparse_product += val * w[key]
    q = sparse_product + b
    return q

def dic_to_vec(dic):
    # convert state from dictionary to vector
    vec = np.zeros((env.state_space,1))
    for key,val in dic.items():
        vec[key] = val
    return vec

def update_weights(w,b,action,state,next_state,reward,learning_rate):
    #gradients
    g_state = dic_to_vec(state).T
    g_b = 1
    #q values
    q = calc_q(state,w[:,action],b)
    max_q_next = np.max(calc_q(next_state,w,b))
    #update
    w[:,action] -= learning_rate * (q - (reward + gamma * max_q_next)) * g_state.squeeze()
    b -= learning_rate * (q - (reward + gamma * max_q_next))* g_b
    return w,b
    
def output_files(weight_out,w,b,returns_out,total_reward_list):
    file = open(weight_out,"w")
    file.write(str(b))
    for x in w.flatten():
        file.write("\n" + str(x))
    file.close()

    file = open(returns_out,"w")
    for x in total_reward_list:
        file.write(str(x)+"\n")
    file.close()

if __name__ == "__main__":
    mode,weight_out,returns_out,episodes,max_iterations,epsilon,gamma,learning_rate = main(sys.argv)
    env = MountainCar(mode)
    #initialize weights and bias to 0
    w = np.zeros((env.state_space,env.action_space))
    b = 0

    total_reward_list = []
    for i in range(episodes):
        state = env.reset()
        total_reward = 0
        for i in range(max_iterations):
            env.render()
            action = pick_action(state, w, b, epsilon)
            next_state,reward,done = env.step(action)
            total_reward += reward
            w,b = update_weights(w,b,action,state,next_state,reward,learning_rate)
            if done:
                break
            state = next_state
        print("total_reward",total_reward)
        total_reward_list.append(total_reward)
    
    output_files(weight_out,w,b,returns_out,total_reward_list)
    env.close()
    

    #plot
    if True:
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 22})
        moving_avg_list = []
        for i in range(25,episodes):
            moving_avg = np.sum(total_reward_list[i-25:i])/25
            moving_avg_list.append(moving_avg)
            
        plt.plot(range(episodes),total_reward_list,label = "return")
        plt.plot(range(25,episodes), moving_avg_list,label = "moving average of return", color = 'red')

        # plt.title("Raw Features: max iterations=200, epsilon=0.05, gamma=0.999,learning rate=0.001")
        plt.title("Tile Features: max iterations=200, epsilon=0.05, gamma=0.99,learning rate=0.00005")
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("Return")
        plt.show()
