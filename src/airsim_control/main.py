import argparse
import time

from agents.random_walker import RandomWalker
from agents.vision_flyer import VisionFlyer
from client.drone_client import DroneClient

# Control loop delay in seconds
CONTROL_LOOP_DELAY = 0.1


def start_recording(client: DroneClient) -> bool:
    """Start recording flight data.

    Args:
        client: Drone client instance.

    Returns:
        True if recording started successfully, False otherwise.
    """
    try:
        client.client.startRecording()
        print("录制已开始")
        return True
    except AttributeError:
        print("警告：当前 AirSim 版本可能不支持 startRecording 方法")
        return False


def stop_recording(client: DroneClient) -> bool:
    """Stop recording flight data.

    Args:
        client: Drone client instance.

    Returns:
        True if recording stopped successfully, False otherwise.
    """
    try:
        client.client.stopRecording()
        print("录制已停止")
        return True
    except AttributeError:
        return False


def main() -> None:
    """Main entry point for drone control."""
    parser = argparse.ArgumentParser(description="AirSim Drone Controller")
    parser.add_argument('--interval', default=1, type=int, help='Control interval in seconds')
    parser.add_argument('--move-type', default='velocity', type=str, help='Movement type')
    parser.add_argument('--save-path', default='./images', type=str, help='Image save path')
    parser.add_argument('--record', default=False, action='store_true', help='Enable recording')
    parser.add_argument('--duration', default=30, type=int, help='Run duration in seconds')
    config = parser.parse_args()

    client = DroneClient(config.interval, root_path=config.save_path)

    # Control recording functionality
    if config.record:
        print("录制功能已启用，开始记录飞行数据")
        start_recording(client)
    else:
        print("录制功能已禁用，减少内存和磁盘使用")

    agent = RandomWalker(client, config.move_type, (-0.5, 0.5))

    # Run for specified duration
    print(f"开始运行，持续 {config.duration} 秒...")
    start_time = time.time()
    while time.time() - start_time < config.duration:
        agent.act()
        time.sleep(CONTROL_LOOP_DELAY)

    # Stop recording if enabled
    if config.record:
        stop_recording(client)

    print("运行完成")
    client.destroy()


if __name__ == '__main__':
    main()