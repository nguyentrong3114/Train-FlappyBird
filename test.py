from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import time
import numpy as np
import matplotlib.pyplot as plt

def test_model(model_path, num_episodes=10, render=True):
    # Khởi tạo môi trường và agent
    env = FlappyBirdEnv(render_enabled=render)
    agent = DQNAgent()
    
    # Load mô hình
    agent.load(model_path)
    print(f"✅ Đã load mô hình từ {model_path}")
    
    # Test mô hình
    scores = []
    steps_list = []
    
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
            if render:
                time.sleep(0.01)  # Thêm delay nhỏ để dễ xem
        
        scores.append(total_reward)
        steps_list.append(steps)
        print(f"Score: {total_reward:.2f} - Steps: {steps}")
    
    # Tính toán thống kê
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    avg_steps = np.mean(steps_list)
    
    print("\n📊 Thống kê:")
    print(f"Điểm trung bình: {avg_score:.2f}")
    print(f"Điểm cao nhất: {max_score:.2f}")
    print(f"Số bước trung bình: {avg_steps:.2f}")
    
    # Vẽ đồ thị
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores, 'b-', label='Score')
    plt.axhline(y=avg_score, color='r', linestyle='--', label='Trung bình')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Kết quả test')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(steps_list, 'g-', label='Steps')
    plt.axhline(y=avg_steps, color='r', linestyle='--', label='Trung bình')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Số bước mỗi episode')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()
    
    return scores, steps_list

if __name__ == "__main__":
    # Test mô hình tốt nhất
    print("🧪 Testing best model...")
    test_model("best_model.pth", num_episodes=10)
    
    # Test mô hình checkpoint cuối cùng
    print("\n🧪 Testing latest checkpoint...")
    test_model("checkpoint_model_750.pth", num_episodes=10) 