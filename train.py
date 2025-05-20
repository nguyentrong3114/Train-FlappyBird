from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

# Thiết lập seed cho tính ổn định
torch.manual_seed(42)
np.random.seed(42)

# Khởi tạo môi trường và agent
env = FlappyBirdEnv(render_enabled=False)
agent = DQNAgent()

if os.path.exists("dqn_model.pth"):
    agent.load("dqn_model.pth")
    print("✅ Loaded saved model.")

# Theo dõi metrics
scores = []
avg_scores = []
best_score = float('-inf')
episode_rewards = []
episode_lengths = []

num_episodes = 1000  # Tăng số episode
eval_frequency = 50  # Đánh giá mô hình mỗi 50 episode

def evaluate_agent(env, agent, num_eval_episodes=5):
    eval_scores = []
    for _ in range(num_eval_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)  # Không sử dụng epsilon trong evaluation
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
        eval_scores.append(total_reward)
    return np.mean(eval_scores)

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if steps % 100 == 0:  # Render mỗi 100 steps để tiết kiệm tài nguyên
            env.render()

    # Cập nhật epsilon
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    
    # Lưu metrics
    scores.append(total_reward)
    episode_lengths.append(steps)
    avg_score = np.mean(scores[-100:])
    avg_scores.append(avg_score)
    
    print(f"Episode {episode} - Score: {total_reward:.2f} - Steps: {steps} - Avg100: {avg_score:.2f} - Epsilon: {agent.epsilon:.3f}")

    # Đánh giá và lưu mô hình tốt nhất
    if episode % eval_frequency == 0:
        eval_score = evaluate_agent(env, agent)
        print(f"🔄 Evaluation Score: {eval_score:.2f}")
        
        if eval_score > best_score:
            best_score = eval_score
            agent.save("best_model.pth")
            print(f"🏆 New best model saved! Score: {best_score:.2f}")
    
    # Lưu checkpoint định kỳ
    if episode % 50 == 0:
        agent.save(f"checkpoint_model_{episode}.pth")
        print("💾 Checkpoint saved.")

# Vẽ đồ thị kết quả
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(scores, label='Score', alpha=0.6)
plt.plot(avg_scores, label='Average Score', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Training Scores")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Episode Lengths")

plt.subplot(2, 2, 3)
plt.hist(scores, bins=50)
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Score Distribution")

plt.tight_layout()
plt.savefig("training_result.png")
plt.close()

# Lưu mô hình cuối cùng
agent.save("final_model.pth")
print("✅ Training completed!")
