import torch
import matplotlib.pyplot as plt

# Parameters
k = 0.1
Lfc = 2.0
Kp = 1.0
dt = 0.1
WB = 2.9  

show_animation = True


class State:
    """
    Vehicle state class.
    車両の状態を管理するクラス
    """
    def __init__(self, x=torch.Tensor([0.0]), y=torch.Tensor([0.0]), yaw=torch.Tensor([0.0]), v=torch.Tensor([0.0])):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * torch.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * torch.sin(self.yaw))
    
    def update(self, a, delta):
        self.x += self.v * torch.cos(self.yaw) * dt
        self.y += self.v * torch.sin(self.yaw) * dt
        self.yaw += self.v / WB * torch.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * torch.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * torch.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return torch.sqrt(dx**2 + dy**2)


class States:
    """
    Store states.
    描画のために状態を保存するクラス
    """
    def __init__(self):
        self.x = torch.Tensor([])
        self.y = torch.Tensor([])
        self.yaw = torch.Tensor([])
        self.v = torch.Tensor([])
        self.t = torch.Tensor([])

    def append(self, t: torch.Tensor, state: State):
        self.x = torch.hstack([self.x, state.x])
        self.y = torch.hstack([self.y, state.y])
        self.yaw = torch.hstack([self.yaw, state.yaw])
        self.v = torch.hstack([self.v, state.v])
        self.t = torch.hstack([self.t, t])



class TargertCourse:
    """
    Target course class.
    this class is used to search the target index.
    """
    def __init__(self, cx, cy):
        """
        Args:
            cx (torch.Tensor): x points of the course
            cy (torch.Tensor): y points of the course
        """
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        """
        Search nearest target index.
        Args:
            state (State): current state of the vehicle
        Returns:
            ind (int): target index
            Lf (torch.Tensor): look ahead distance
        """
        # * doing this at only first time
        if self.old_nearest_point_index is None:
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = torch.sqrt(torch.tensor(dx)**2 + torch.tensor(dy)**2)
            ind = torch.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])

            while True:
                distance_next_index = state.calc_distance(self.cx[ind + 1], self.cy[ind + 1])

                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc

        # * search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break
            ind += 1
        
        return ind, Lf
    

def proportional_control(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """
    Proportional control: 比例制御
    Args:
        target (torch.Tensor): target speed 
        current (torch.Tensor): current speed
    Returns:
        a (torch.Tensor): acceleration
    """
    a = Kp * (target - current)
    return a


def pure_pursuit_control(state: State, trajectory: TargertCourse, pind: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure pursuit control
    Args:
        state (State): state of the vehicle
        trajectory (TargertCourse): target trajectory
        pind (int): previous index
    Returns:
        delta (torch.Tensor): steering angle
        ind (torch.Tensor): target index
    """
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else: # for the last point
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = torch.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw

    delta = torch.atan2(2.0 * WB * torch.sin(alpha) / Lf, torch.Tensor([1.0]))

    return delta, ind

def main():
    # target course
    cx = torch.arange(0, 50, 0.5)
    cy = torch.sin(cx / 5.0) * cx / 2.0

    target_speed = 10.0 / 3.6  # [m/s]

    T = 100.0  # max simulation time

    # initial state
    state = State(x=torch.Tensor([-0.0]),
                  y=torch.Tensor([-3.0]), 
                  v=torch.Tensor([0.0]), 
                  yaw=torch.Tensor([0.0]))

    lastIndex = len(cx) - 1
    time = torch.Tensor([0.0])
    states = States()
    states.append(time, state)
    target_course = TargertCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)

    while T >= time and lastIndex > target_ind:
        # calc control input
        ai = proportional_control(target_speed, state.v)
        di, target_ind = pure_pursuit_control(state, target_course, target_ind)

        state.update(ai, di)

        time += dt
        states.append(time, state)

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx.numpy(), cy.numpy(), "-r", label="course")
            plt.plot(states.x.numpy(), states.y.numpy(), "-b", label="trajectory")
            plt.plot(cx[target_ind].item(), cy[target_ind].item(), "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v.item() * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert lastIndex >= target_ind, "Cannot reach goal"

    if show_animation:
        print(states.x.numpy())
        plt.cla()
        plt.plot(cx.numpy(), cy.numpy(), ".r", label="course")
        plt.plot(states.x.numpy(), states.y.numpy(), "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t.numpy(), [iv * 3.6 for iv in states.v.numpy()], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()