from k_armed_bandit import BanditEnv
import numpy as np
from scipy.special import softmax
import gym
import matplotlib.pyplot as plt

"""
Paramter study of various bandit algorithms for the nonstationary 10-armed
bandit problem (Exercise 2.11, page 44). The data points represent the averages
of the last 100000 steps from 200000 steps runs.
"""

class Agent:
    def __init__(
            self,
            exploration_prob: float,
            ucb_confidence: float = 0.0,
            init_val_estimate: float = 0.0,
            step_size: str = None,
            estimation_method: str = "normal"):
        self.exploration_prob = exploration_prob
        self.ucb_confidence = ucb_confidence
        self.init_val_estimate = init_val_estimate
        self.step_size = step_size
        self.estimation_method = estimation_method

    def train(
            self,
            environment: gym.Env,
            data_type: str,
            num_episodes: int = 2000,
            num_steps: int = 100000
            ) -> np.array:
        output_data = np.empty([num_episodes, num_steps])

        # episodes
        for i_episode in range(num_episodes):
            #print("- Episode {}".format(i_episode))
            observation = env.reset()
            val_estimates = np.full(environment.bandits, self.init_val_estimate)
            iterations = np.full(environment.bandits, 1)
            if self.estimation_method == "gradient":
                avg_reward = 0.0
                rewards = np.zeros(environment.bandits)
                preferences = val_estimates

            # steps
            for t_step in range(num_steps):
                #print("- Episode {}: Step {}".format(i_episode, t_step))
                env.render()

                # explore or exploit
                if environment.np_random.uniform() < self.exploration_prob:
                    action = environment.action_space.sample()
                else:
                    action = np.argmax(
                        [val_estimates[action] + self.ucb_confidence * np.sqrt(np.log(t_step + 1) / iterations[action])
                            for action in range(environment.bandits)])

                # perform action
                observation, reward, done, info = env.step(action)
                iterations[action] += 1

                #print("  > Action {} (iteration {})".format(
                #    action, iterations[action]))
                #print("    Value estimate: {:.3f}".format(
                #    val_estimates[action]), end="")

                # set step size
                if self.step_size is None:  # sample-average
                    self.step_size = 1 / iterations[action]

                # update estimates
                if self.estimation_method == "gradient":
                    avg_reward += (reward - rewards[action]) / len(rewards)
                    rewards[action] = reward
                    action_brob = softmax(preferences)
                    is_selected, is_selected[action] = np.zeros(environment.bandits), 1

                    preferences += (
                        self.step_size * (reward - avg_reward) * (is_selected - action_brob))
                else:
                    val_estimates[action] += (
                        self.step_size * (reward - val_estimates[action]))

                #print(" -> {:.3f}".format(val_estimates[action]))
                #print("    Reward: {:.3f}".format(reward))

                # store data
                if data_type == "scores":
                    output_data[i_episode, t_step] = reward
                elif data_type == "best_found":
                    best_action = np.argmax(
                        [column[0] for column in env.r_dist])
                    output_data[i_episode, t_step] = (action == best_action)

                if done:
                    print("Episode finished after {} timesteps".format(t_step + 1))
                    break

            ## print info
            #np.set_printoptions(precision=2)
            #print("  Estimates: {}".format(val_estimates))
            #print("  Iterations: {}".format(iterations))
            #if data_type == "scores":
            #    print("  Score: {:.2f}".format(
            #        output_data[i_episode][-1]))
            #elif data_type == "best_found":
            #    print("  Found best ({}): {}".format(
            #        best_action, output_data[i_episode][-1]))

        return np.average(output_data, axis=0)


if __name__ == "__main__":
    """
    10 armed bandit mentioned on page 33 of Sutton and Barto's Reinforcement Learning: An Introduction
    - Actions always pay out
    - Mean of payout is pulled from a normal distribution (0, 1) (called q*(a))
      and takes random walks (adding an increment from a normal distribution (0, 0.01))
    - Actual reward is drawn from a normal distribution (q*(a), 1)
    """
    env = BanditEnv(bandits=10)
    env.p_dist = np.full(env.bandits, 1)
    env.walk_dist = np.repeat([[0, 0.01]], env.bandits, axis=0)
    #env.walk_dist = None
    env.set_r_dist(
        np.array([[env.np_random.normal(0, 1), 1] for i in range(10)]))

    num_episodes = 200
    num_steps = 1000
    n = 6

    """
    # epsilon-greedy
    print("-- epsilon-greedy:")
    greedy_scores = np.empty(n)
    epsilon = np.empty(n)

    for i in range(n):
        epsilon[i] = 2 ** -(i + 2)
        print("epsilon({}): {}".format(i, epsilon[i]), end=", ")
        greedy_agent = Agent(
            exploration_prob=epsilon[i],
            step_size=0.1)

        greedy_train = greedy_agent.train(env, "scores", num_episodes, num_steps)
        greedy_scores[i] = np.average(greedy_train[-int(num_steps / 2):])
        print("score: {}".format(greedy_scores[i]))

    plt.plot(epsilon, greedy_scores, color="red", label="ε-greedy")

    # greedy with optimistic initialization
    print("-- greedy with optimistic initialization:")
    optimist_scores = np.empty(n)
    Q0 = np.empty(n)

    for i in range(n):
        Q0[i] = 2 ** -(i - 2)
        print("Q0({}): {}".format(i, Q0[i]), end=", ")
        optimist_agent = Agent(
            init_val_estimate=Q0[i],
            exploration_prob=0.1,
            step_size=0.1)

        optimist_train = optimist_agent.train(env, "scores", num_episodes, num_steps)
        optimist_scores[i] = np.average(optimist_train[-int(num_steps / 2):])
        print("score: {}".format(optimist_scores[i]))

    plt.plot(Q0, optimist_scores, color="black",
             label="greedy with optimistic initialization α=0.1")

    # UCB
    print("-- upper confidence bound:")
    ucb_scores = np.empty(n)
    c = np.empty(n)

    for i in range(n):
        c[i] = 2 ** -(i - 3)
        print("c({}): {}".format(i, c[i]), end=", ")
        ucb_agent = Agent(
            ucb_confidence=c[i],
            exploration_prob=0.1,
            step_size=0.1)

        ucb_train = ucb_agent.train(env, "scores", num_episodes, num_steps)
        ucb_scores[i] = np.average(ucb_train[-int(num_steps / 2):])
        print("score: {}".format(ucb_scores[i]))

    plt.plot(c, ucb_scores, color="blue", label="UCB")
    """

    # gradient bandit
    print("-- upper confidence bound:")
    gradient_scores = np.empty(n)
    a = np.empty(n)

    for i in range(n):
        a[i] = 2 ** -(i - 1)
        print("a({}): {}".format(i, a[i]), end=", ")
        gradient_agent = Agent(
            estimation_method="gradient",
            step_size=a[i],
            exploration_prob=0.1)

        gradient_train = gradient_agent.train(env, "scores", num_episodes, num_steps)
        gradient_scores[i] = np.average(gradient_train[-int(num_steps / 2):])
        print("score: {}".format(gradient_scores[i]))

    plt.plot(a, gradient_scores, color="green", label="gradient bandit")

    # plots
    plt.xlim([1 / 2 ** 7, 2 ** 2])
    #plt.ylim([1, 2.5])
    plt.xscale("log", base=2)
    plt.legend()
    plt.show()

    env.close()
