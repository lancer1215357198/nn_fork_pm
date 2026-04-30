import time

try:
    import keyboard
except ImportError:
    # Provide a mock keyboard implementation for testing
    class MockKeyboard:
        def is_pressed(self, key: str) -> bool:
            return False

    keyboard = MockKeyboard()

from agents.agent import Agent

# Default speed constants
DEFAULT_SPEED = 5
DEFAULT_TURN_SPEED = 2
DEFAULT_HEIGHT_ADJUST_SPEED = 3

# Control key mappings
KEY_FORWARD = 'w'
KEY_BACKWARD = 's'
KEY_LEFT = 'a'
KEY_RIGHT = 'd'
KEY_UP = 'q'
KEY_DOWN = 'e'
KEY_STOP = 'space'
KEY_EXIT = 'esc'


class KeyboardController(Agent):
    """Keyboard-controlled drone agent.

    Args:
        client: AirSim client instance.
        move_type: Movement type for the drone.
    """

    def __init__(self, client, move_type: str) -> None:
        super(KeyboardController, self).__init__(client, move_type)
        self.client.start()
        self.speed = DEFAULT_SPEED
        self.turn_speed = DEFAULT_TURN_SPEED
        self.height_adjust_speed = DEFAULT_HEIGHT_ADJUST_SPEED
        self.print_controls()

    def print_controls(self) -> None:
        """Print keyboard control instructions."""
        print("\n控制说明:")
        print("键盘控制:")
        print("W: 向前移动")
        print("S: 向后移动")
        print("A: 向左转向")
        print("D: 向右转向")
        print("Q: 向上移动")
        print("E: 向下移动")
        print("空格键: 停止移动")
        print("ESC: 退出控制\n")

    def get_state(self) -> dict:
        """Get current drone state.

        Returns:
            Dictionary with position and velocity information.
        """
        state = self.client.get_state()
        position = state.kinematics_estimated.position
        return {
            'position': (position.x_val, position.y_val, position.z_val),
            'velocity': state.kinematics_estimated.linear_velocity
        }

    def handle_keyboard_input(self) -> tuple[float, float, float]:
        """Process keyboard input and return velocity components.

        Returns:
            Tuple of (vx, vy, vz) velocity components.
        """
        vx, vy, vz = 0.0, 0.0, 0.0

        # Forward/backward movement
        if keyboard.is_pressed(KEY_FORWARD):
            vx = self.speed
        elif keyboard.is_pressed(KEY_BACKWARD):
            vx = -self.speed

        # Left/right turning
        if keyboard.is_pressed(KEY_LEFT):
            vy = -self.turn_speed
        elif keyboard.is_pressed(KEY_RIGHT):
            vy = self.turn_speed

        # Up/down movement
        if keyboard.is_pressed(KEY_UP):
            vz = self.height_adjust_speed
        elif keyboard.is_pressed(KEY_DOWN):
            vz = -self.height_adjust_speed

        # Stop movement
        if keyboard.is_pressed(KEY_STOP):
            vx, vy, vz = 0.0, 0.0, 0.0

        return vx, vy, vz

    def act(self) -> bool:
        """Execute one control step.

        Returns:
            True to continue, False to exit.
        """
        vx, vy, vz = self.handle_keyboard_input()
        self.client.move(self.move_type, vx, vy, vz)

        # Check for exit key
        if keyboard.is_pressed(KEY_EXIT):
            print("退出控制")
            return False

        return True

    def run(self, loop_cnt: int = 100) -> None:
        """Run the keyboard control loop.

        Args:
            loop_cnt: Maximum number of iterations.
        """
        for _ in range(loop_cnt):
            if not self.act():
                break
            time.sleep(0.1)
