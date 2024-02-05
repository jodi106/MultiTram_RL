# TODO: Add while loop to check_dist_to_0 in act fucntions to allow more than 3 action
class DeterministicAgent:
    def __init__(self, env, max_speed=None):
        self.env = env
        self.max_acceleration = 1
        self.max_deceleration = 1
        self.maxSpeed = max_speed

    def check_dist_to_0(self,position, speed, action, dist_target, time_steps = 1, max_deceleration = 1):
        acceleration = action -1
        new_position = position + max(0, 0.5 * acceleration * time_steps**2 + time_steps * speed)
        new_speed =  max(0,acceleration*time_steps+speed)
        distance_to_0 = -(new_speed ** 2) / (2 * -max_deceleration)
        distance_to_target = position-new_position+dist_target

        return distance_to_0 <= distance_to_target

    def check_dist_to_0_MT(self,position, speed, action, dist_target, ind, time_steps = 1, max_deceleration = 1):
        acceleration = action -1
        if ind % 2 == 0:
            new_position = position + max(0, 0.5 * acceleration * time_steps**2 + time_steps * speed)
        else:
            new_position = position - max(0, 0.5 * acceleration * time_steps**2 + time_steps * speed)
        
        new_speed =  max(0,acceleration*time_steps+speed)
        distance_to_0 = (new_speed ** 2) / (2 * max_deceleration)
        if ind % 2 == 0:
            distance_to_target = position-new_position+dist_target
        else:
            distance_to_target = new_position-position+dist_target

        return distance_to_0 <= distance_to_target
    
    def act(self):
        speed, _, _, _, _, _, _, _, _, _, _, dist_target, _, _, _, _, _, _ = self.env.state
        position = self.env.position
        max_speed = self.env.speed_limits[int(round(position))]
        if self.maxSpeed is not None:
            max_speed=self.maxSpeed

        # Initial action is to accelerate to max speed
        action = 2  # Max acceleration

        if speed >= max_speed:
            action = 1  # Maintain speed

        if not self.check_dist_to_0(position, speed, action, dist_target):
            action-=1

        if not self.check_dist_to_0(position, speed, action, dist_target):
            action-=1

        if dist_target == 0:
            action = 2

        return action
    
    def act_MT(self):
        position = self.env.positions.copy()
        state_unscaled = self.env.get_state()
        speed, dist_target, max_speed = [], [], []
        [speed.append(state_unscaled[x*18]) for x in range(self.env.n_vehicles)]
        [dist_target.append(state_unscaled[x*18+11]) for x in range(self.env.n_vehicles)]
        if self.maxSpeed is not None:
            max_speed=[self.maxSpeed]*self.env.n_vehicles        
        else:
            for x in range(self.env.n_vehicles):

                if x % 2 == 0:
                    max_speed.append(self.env.speed_limits[min(int(round(position[x])), len(self.env.speed_limits) - 1)])
                else:
                    max_speed.append(self.env.reverse_speed_limits[max(int(round(position[x])), 0)])

        # Initial action is to accelerate to max speed
        action = [2]*self.env.n_vehicles  # Max acceleration

        for x in range(self.env.n_vehicles):
            if speed[x] == max_speed[x]:
                action[x] = 1  # Maintain speed

            if speed[x] > max_speed[x]:
                action[x] = 0  # Maintain speed

            else:
                if not self.check_dist_to_0_MT(position[x], speed[x], action[x], dist_target[x], ind = x):
                    action[x]-=1

                if not self.check_dist_to_0_MT(position[x], speed[x], action[x], dist_target[x], ind = x):
                    action[x]-=1

                if dist_target[x] == 0:
                    action[x] = 2

        return action    