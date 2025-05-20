from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import time

# Khởi tạo môi trường và agent
env = FlappyBirdEnv(render_enabled=True)  # Bật render để xem
agent = DQNAgent()

# Load mô hình tốt nhất
agent.load("best_model.pth")
print("✅ Đã load mô hình tốt nhất")

# Test mô hình
num_episodes = 5
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    print(f"\nEpisode {episode + 1}:")
    while not done:
        action = agent.act(state)  # Không sử dụng epsilon trong test
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1
        time.sleep(0.01)  # Thêm delay nhỏ để dễ xem
    
    print(f"Score: {total_reward:.2f} - Steps: {steps}")

print("\n✅ Test hoàn tất!") 