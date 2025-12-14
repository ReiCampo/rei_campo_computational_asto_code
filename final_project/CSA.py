import numpy as np

class ObservingSatellite:
    
    def __init__(self,
                 name,
                 position,
                 velocity):
        
        self.name = name
        
        self.position = np.array(position, dtype = float)
        self.velocity = np.array(velocity, dtype = float)    
        
        self.position_history = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.time_history = [0.0]
        
    def line_of_sight(self, target_position):
        los = target_position - self.position
        norm_los = los / np.linalg.norm(los)
        return norm_los
        
    def update_state_vector(self,
                            updated_position,
                            updated_velocity,
                            time):
        
        self.position = np.array(updated_position, dtype = float)
        self.velocity = np.array(updated_velocity, dtype = float)
        
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.time_history.append(time)
        
    
    def distance_to(self, target_position):
        """Calculate distance from satellite to a target position."""
        return np.linalg.norm(self.position - np.array(target_position))
    
    def get_current_state(self):
        return {"X Position: ": self.position[0],
                "Y Position: ": self.position[1],
                "Z Position: ": self.position[2],
                "X Velocity: ": self.velocity[0],
                "Y Velocity: ": self.velocity[1],
                "Z Velocity: ": self.velocity[2]}
       
       
        
    def __repr__(self):
        return(f"ObservingSatellite('{self.name}', "
               f"pos = {self.position}, vel = {self.velocity}")

 
class TargetObject:
    
    def __init__(self,
                 name,
                 position,
                 velocity):
        
        self.name = name
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        
        # Store history
        self.position_history = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.time_history = [0.0]
        
    def update_state(self, new_position, new_velocity, time):

        self.position = np.array(new_position, dtype=float)
        self.velocity = np.array(new_velocity, dtype=float)
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.time_history.append(time)
        
    def get_current_state(self):

        return np.concatenate([self.position, self.velocity])
    
    def __repr__(self):

        return (f"TargetObject('{self.name}', "
                f"pos={self.position}, vel={self.velocity})")
        
        
        
class SynodicFrame:
      
##-----------------------------------------------------------------------------
##  First creating an initializing function that holds values pertaining to   -
##  the Earth and Moon                                                        -
##-----------------------------------------------------------------------------
    
    def __init__(self):
        
        ###  I am going to use the gravitational parameters, mu, of the masses in the     
        ###  system (so G * mass of object) because it is easier to keep track of those   
        ###  things and shows you how strong each mass's gravitational influence is       

        self.mu_earth = 398600.4418  # km^3/s^2, may change this to Lundar Distance (LU) later
        self.mu_moon = 4902.8000     # km^3/s^2
        
        # Average distance of moon from the Earth
        self.earth_moon_distance = 384400  # km
        
        # Calculating mass parameter of Moon relative to Earth. This will help
        # later on when setting up initial positions of the system.
        self.mu = self.mu_moon / (self.mu_earth + self.mu_moon)
        
        # Angular velocity of Earth-Moon system (rad/s), the equation comes from
        # one of Kepler's laws adapted a bit: 
        # omega ^ 2 = (total mass of system) / (distance between masses) ^ 3
        self.omega = np.sqrt((self.mu_earth + self.mu_moon) / 
                            self.earth_moon_distance**3)
        
        # Period of Earth-Moon system
        self.period = 2 * np.pi / self.omega
        
        # Characteristic units, will be used later for normalizing/denormalizing:
        self.len_characteristic = self.earth_moon_distance
        self.time_characteristic = 1 / self.omega
        self.velo_characteristic = self.len_characteristic/ self.time_characteristic
    
    def normalize_initial_conditions(self, positions, velocities):
        
        normalized_positions = positions / self.len_characteristic
        normalized_velocities = velocities / self.velo_characteristic
        
        return normalized_positions, normalized_velocities
    
    def denormalize_state_vector(self, normalized_positions, normalized_velocities):

        positions = normalized_positions * self.len_characteristic
        velocities = normalized_velocities * self.velo_characteristic
        return positions, velocities
    
    def rotate_frame(self, theta):
        
        cosine = np.cos(theta)
        sine = np.sin(theta)
        
        rotation_matrix = np.array([ [cosine, -sine,  0],
                                     [sine,   cosine, 0],
                                     [0,      0,      1] ])
    
        return rotation_matrix
    
    def inertial_to_synodic(self, r_inertial, velo_inertial, time):
        
        theta = self.omega * time
        
        # Choosing negative theta here because the Moon is going around the 
        # Earth counterclockwise
        R = self.rotate_frame(-theta)
        
        r_synodic = R @ r_inertial
        
        omega_vector = np.array([0, 0, self.omega])
        velo_synodic = R @ velo_inertial - np.cross(omega_vector, r_synodic)
        
        return r_synodic, velo_synodic
    
    def synodic_to_inertial(self, r_synodic, velo_synodic, time):
        theta = self.omega * time
        
        R = self.rotate_frame(theta)
        
        r_inertial = R @ r_synodic
        
        omega_vector = np.array([0, 0, self.omega])
        velo_inertial = R @ velo_synodic + np.cross(omega_vector, r_inertial)
        
        return r_inertial, velo_inertial

    
    def get_lagrange_points(self, normalized = False):

        # L1, L2, L3 are on the x-axis (approximate solutions)
        # These would need numerical refinement for exact values
        
        # Approximate L1 (between Earth and Moon)
        L1_x = 1 - self.mu - (self.mu/3)**(1/3)
        
        # Approximate L2 (beyond Moon)
        L2_x = 1 - self.mu + (self.mu/3)**(1/3)
        
        # Approximate L3 (beyond Earth)
        L3_x = -1 - self.mu - (5*self.mu/12) - (13*self.mu**2/48)
        
        # L4 and L5 form equilateral triangles
        L4 = np.array([0.5 - self.mu, np.sqrt(3)/2, 0])
        L5 = np.array([0.5 - self.mu, -np.sqrt(3)/2, 0])
        
        L_points = {'L1': np.array([L1_x, 0, 0]),
                    'L2': np.array([L2_x, 0, 0]),
                    'L3': np.array([L3_x, 0, 0]),
                    'L4': L4,
                    'L5': L5}
        
        
        if normalized == False:
            for i in L_points:
                L_points[i] = L_points[i] * self.len_characteristic
            
        
        return L_points
        
    
        
    def get_earth_position(self, normalized = False):
        pos = np.array([-self.mu, 0, 0])
            
        if normalized == True:
            pos = pos
        else:
            pos = pos * self.len_characteristic
        
        return pos
        
            
    def get_moon_position(self, normalized = False):
        pos = np.array([1 - self.mu, 0, 0])
        
        if normalized == True:
            pos = pos
        else:
            pos = pos * self.len_characteristic
        
        return pos
