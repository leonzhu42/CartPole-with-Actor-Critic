import random
import tensorflow as tf
import numpy as np
import gym

actor_network_verbose = False
update_gradients_verbose = False

actor_learning_rate = 0.0001
critic_learning_rate = 0.00008 # Slightly less than actor_learning_rate

n_episode = 1000
max_step = 1000
gamma = 0.99

observation_placeholder = tf.placeholder(tf.float32, shape=[None, 4])
W1_actor = tf.get_variable('W1_actor', shape=[4, 64])
b1_actor = tf.get_variable('b1_actor', shape=[64])
W2_actor = tf.get_variable('W2_actor', shape=[64, 64])
b2_actor = tf.get_variable('b2_actor', shape=[64])
W3_actor = tf.get_variable('W3_actor', shape=[64, 2])
b3_actor = tf.get_variable('b3_actor', shape=[2])
hidden1_actor = tf.nn.relu(tf.matmul(observation_placeholder, W1_actor) + b1_actor)
hidden2_actor = tf.nn.relu(tf.matmul(hidden1_actor, W2_actor) + b2_actor)
scores = tf.nn.relu(tf.matmul(hidden2_actor, W3_actor) + b3_actor)
probs = tf.nn.softmax(scores)

discount_placeholder = tf.placeholder(tf.float32)
gain_placeholder = tf.placeholder(tf.float32)
action_placeholder = tf.placeholder(tf.float32, shape=[None, 2])
log_probs = tf.log(probs) * action_placeholder
grads_actor = tf.gradients(log_probs, [W1_actor, b1_actor, W2_actor, b2_actor, W3_actor, b3_actor])
update_W1_actor = W1_actor.assign_add(actor_learning_rate * discount_placeholder * gain_placeholder * grads_actor[0])
update_b1_actor = b1_actor.assign_add(actor_learning_rate * discount_placeholder * gain_placeholder * grads_actor[1])
update_W2_actor = W2_actor.assign_add(actor_learning_rate * discount_placeholder * gain_placeholder * grads_actor[2])
update_b2_actor = b2_actor.assign_add(actor_learning_rate * discount_placeholder * gain_placeholder * grads_actor[3])
update_W3_actor = W3_actor.assign_add(actor_learning_rate * discount_placeholder * gain_placeholder * grads_actor[4])
update_b3_actor = b3_actor.assign_add(actor_learning_rate * discount_placeholder * gain_placeholder * grads_actor[5])

W1_critic = tf.get_variable('W1_critic', shape=[4, 64])
b1_critic = tf.get_variable('b1_critic', shape=[64])
W2_critic = tf.get_variable('W2_critic', shape=[64, 64])
b2_critic = tf.get_variable('b2_critic', shape=[64])
W3_critic = tf.get_variable('W3_critic', shape=[64, 1])
b3_critic = tf.get_variable('b3_critic', shape=[1])
hidden1_critic = tf.nn.relu(tf.matmul(observation_placeholder, W1_critic) + b1_critic)
hidden2_critic = tf.nn.relu(tf.matmul(hidden1_critic, W2_critic) + b2_critic)
values = tf.matmul(hidden2_critic, W3_critic) + b3_critic

grads_critic = tf.gradients(values, [W1_critic, b1_critic, W2_critic, b2_critic, W3_critic, b3_critic])
update_W1_critic = W1_critic.assign_add(critic_learning_rate * gain_placeholder * grads_critic[0])
update_b1_critic = b1_critic.assign_add(critic_learning_rate * gain_placeholder * grads_critic[1])
update_W2_critic = W2_critic.assign_add(critic_learning_rate * gain_placeholder * grads_critic[2])
update_b2_critic = b2_critic.assign_add(critic_learning_rate * gain_placeholder * grads_critic[3])
update_W3_critic = W3_critic.assign_add(critic_learning_rate * gain_placeholder * grads_critic[4])
update_b3_critic = b3_critic.assign_add(critic_learning_rate * gain_placeholder * grads_critic[5])

def actor_network(observation, session, verbose=False):
    probabilities = session.run(probs, feed_dict={observation_placeholder: [observation]})
    if verbose:
        print(probabilities)
    if random.random() <= probabilities[0][0]:
        return 0
    else:
        return 1

def update_actor_gradients(discount, gain, observation, action, session, verbose=False):
    if action == 0:
        mask = [[1, 0]]
    else:
        mask = [[0, 1]]
    result = session.run([probs, log_probs, grads_actor, update_W1_actor, update_b1_actor, update_W2_actor, update_b2_actor, update_W3_actor, update_b3_actor], feed_dict={observation_placeholder: [observation], discount_placeholder: discount, gain_placeholder: gain, action_placeholder: mask})
    if verbose:
        print(result[2]) # print the gradients w.r.t. weights and biases

def update_critic_gradients(gain, observation, session, verbose=False):
    result = session.run([grads_critic, update_W1_critic, update_b1_critic, update_W2_critic, update_b2_critic, update_W3_critic, update_b3_critic], feed_dict={observation_placeholder: [observation], gain_placeholder: gain})
    if verbose:
        print(result[0])

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    env = gym.make('CartPole-v1')

    # Train
    print('Training...')
    for episode in range(n_episode):
        observation = env.reset()
        reward = 0
        done = False
        total_reward = 0.0
        memory = []
        discount = 1.0
        step = 0
        while not done and step < max_step:
            action = actor_network(observation, sess, verbose=actor_network_verbose)
            next_observation, next_reward, next_done, _ = env.step(action)
            if next_done:
                next_reward = -100

            gain_with_critic = next_reward - sess.run(values, feed_dict={observation_placeholder: [observation]})[0][0]
            if not next_done:
                gain_with_critic += gamma * sess.run(values, feed_dict={observation_placeholder: [next_observation]})[0][0]
            update_actor_gradients(discount=discount,gain=gain_with_critic, observation=observation,action=action, session=sess,verbose=update_gradients_verbose)
            update_critic_gradients(gain=gain_with_critic,observation=observation, session=sess,verbose=update_gradients_verbose)
            discount *= gamma

            total_reward += next_reward
            observation, reward, done = next_observation, next_reward, next_done
            step += 1
        print('Episode {0} Reward: {1}'.format(episode, total_reward))

    print('Training done!')
    print()

    # Test
    input('Press Enter to start testing!')
    print('Testing...')
    for episode in range(5):
        observation = env.reset()
        reward = 0
        done = False
        total_reward = 0.0
        while not done:
            env.render()
            action = actor_network(observation, sess)
            next_observation, next_reward, next_done, _ = env.step(action)
            total_reward += next_reward
            observation, reward, done = next_observation, next_reward, next_done
        print('Reward:', total_reward)

    env.close()