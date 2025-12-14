
###############################################################################
###############################################################################
###                                                                         ###
###  COMPUTATIONAL METHODS FINAL PROJECT: CISLUNAR SITATUATIONAL AWARENESS  ###
###                            SATELLITE TRACKER                            ###
###                                                                         ###
###############################################################################
###############################################################################


##----------------------------------------------------------------
##                  Importing Necessary Packages                 -
##----------------------------------------------------------------

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import ImageMagickWriter, writers
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import copy

# sys.path.insert(0, "/Users/RachelCampo/Desktop/CUNY Classes/Fall 2025 Computational Astro/rei_campo_computational_asto_code/final_project")
# from CSA import ObservingSatellite, TargetObject, SynodicFrame

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


def calculate_parallax(sat1, 
                       sat2, 
                       real_target_position,
                       angular_noise = 0.01,
                       n_samples = 10):
    """
    This function calculates the parallax of the target object while including
    Gaussian noise in the angular calculation. This error is added because it is
    one of the main sources of deviations in when predicting trajectories. 
    
    Inputs:
        sat1: ObservingSatellite class
            An observing satellite that can either be in GEO, or at an
            Earth-Moon Lagrange point.
            
        sat2: ObservingSatellite class
            A secondary observing satellite that can be in GEO or at an
            Earth-Moon Lagrange point. Two satellites are needed to calculate
            parallax.
            
        real_target_position: array with shape (3,)
            The actual position of the target, in kilometers.
            
        angular_noise: float
            The standard deviation of angular measurement error in degrees.
            0.01 is a typical deviation based on today's satellites' measurment
            precision.
        
        n_samples: int
            The number of samples taken from the Gaussian uncertainty. These
            samples will then be used later to calculate alternate trajectories
            of the target object. The sampling is Monte Carlo.
            
    
    Outputs:
        average_estimate array with shape (3,)
        
        
    """
    # Creating a function that will be used multiple times when calculating the
    # true and sampled positions:
    
    def parallax_helper(sat1,
                        sat2,
                        los1,
                        los2):
        
    # Here is where I begin to use a lot of linear algebra to set up the system
    # so I can calculate parallax later. Because both satellites have varying
    # distances to the target object, the equation that I'm going to solve is
    # satellite 1 position + distance 1 to target * line of sight vector 1 =~ satellite 2 position + distance 2 to target * line of sight vector 2
    # or if you rearrange, you get:
    # satellite position 2 - satellite position 1 = (distance 1 to target * line of sight vector 1) - (distance 2 to target * line of sight vector 2)
    # Because this is a system of equations with three variables and two
    # unknowns, we are going to have to use least squares calculations to find
    # distance 2 to target and distance 1 to target.
    
        distance_between_satellites = sat2.position - sat1.position
    
    # If we factor out the line of sight vectors from the right hand side of 
    # the equation, we will get this matrix below:
        los_matrix = np.column_stack([los1, -los2])
    
    # So now, with the way we have this set up, we have the equation:
    # los_matrix @ distance_matrix = distance between satellites. To solve for
    # this system, we can use np.linalg.lstsq(). This function will find an 
    # approximate solution to the equation.
        target_distances, residuals, rank, s = np.linalg.lstsq(los_matrix, 
                                                           distance_between_satellites, 
                                                           rcond = None)
    
    # Extracting estimated distances for each satellite from previous 
    # calculation:
        satellite1_dist_estimate = target_distances[0]
        satellite2_dist_estimate = target_distances[1]
    
    # Calculate target position from both satellites
        target_from_sat1 = sat1.position + satellite1_dist_estimate * los1
        target_from_sat2 = sat2.position + satellite2_dist_estimate * los2

    # Take average as best estimate
        target_position = (target_from_sat1 + target_from_sat2) / 2
        
        return target_position


    def add_angular_error(los_vector,
                          angular_std):
        
        los_unit = los_vector / np.linalg.norm(los_vector)
        angular_std_rad = np.radians(angular_std)
        
        if abs(los_unit[2]) < 0.9:
            perp1 = np.cross(los_unit, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(los_unit, np.array([1, 0, 0]))
            
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(los_unit, perp1)
        perp2 = np.cross(los_unit, perp1) / np.linalg.norm(perp2)
        
        # Adding in the Gaussian noise:
        noise1 = np.random.normal(0, angular_std_rad)
        noise2 = np.random.normal(0, angular_std_rad)
        
        los_with_noise = los_unit + (noise1 * perp1) + (noise2 * perp2)
        los_with_noise = los_with_noise / np.linalg.norm(los_with_noise)
        
        return los_with_noise
        
    # First, calculate the true estimate with no noise:

    sat1_true_los = sat1.line_of_sight(real_target_position)
    sat2_true_los = sat2.line_of_sight(real_target_position)
    
    true_estimate = parallax_helper(sat1,
                                    sat2,
                                    sat1_true_los,
                                    sat2_true_los)
    
    # Now calculate the estimated trajectories:
    
    sampled_estimates = np.zeros((n_samples, 3))
    
    for i in range(n_samples):
        
        sat1_los_noise = add_angular_error(sat1_true_los, angular_noise)
        sat2_los_noise = add_angular_error(sat2_true_los, angular_noise)
        
        sampled_estimates[i] = parallax_helper(sat1,
                                               sat2,
                                               sat1_los_noise,
                                               sat2_los_noise)
        
    average_estimate = np.mean(sampled_estimates, axis = 0)
    uncertainty = np.std(np.linalg.norm(sampled_estimates - real_target_position, axis = 1))
    
    
    return average_estimate, true_estimate, sampled_estimates, uncertainty



def propagation_function(target,
                         dt,
                         n_steps,
                         synodic_frame,
                         method,
                         n_samples,
                         calc_uncertainties = True,
                         position_uncertainty = 100,
                         velocity_uncertainty = 0.01):
    
    state_vector_norm, _ = synodic_frame.normalize_initial_conditions(target.position,
                                                                      target.velocity)
    
    state_vector_norm = np.concatenate([state_vector_norm,
                                        target.velocity / synodic_frame.velo_characteristic])
    
    def cr3bp_equations_of_motion(mu, state_vector):
    
        x_position, y_position, z_position, x_velocity, y_velocity, z_velocity = state_vector
    
        r1 = np.sqrt((x_position + mu)**2 + y_position**2 + z_position**2)
        r2 = np.sqrt((x_position - (1 - mu))**2 + y_position**2 + z_position**2)
    
        x_accel = (2*y_velocity + x_position - (1 - mu) * (x_position + mu)/r1**3 - 
              mu * (x_position - (1 - mu)) / r2**3)
        y_accel = (-2 * x_velocity + y_position - (1 - mu) * y_position / r1**3 - mu * y_position / r2**3)
        z_accel = (-(1 - mu) * z_position / r1**3 - mu * z_position / r2**3)
    
        return np.array([x_velocity, y_velocity, z_velocity, x_accel, y_accel, z_accel])
    
    def leapfrog_verlet(h,
                    state_vector,
                    mu):
    
        # Calculating the current velocities and accelerations:
        x, y, z, vx, vy, vz = state_vector
        
        derivatives = cr3bp_equations_of_motion(mu, state_vector)
        ax, ay, az = derivatives[3:6]
        
    # Finding the velocity a half step forward:
        vx_half = vx + (ax * (h / 2))
        
        vy_half = vy + (ay * (h / 2))
    
        vz_half = vz + (az * (h / 2))
        
        # Now finding the new positions using the half velocities from above:
        new_x_position = x + (h * vx_half)
        
        new_y_position = y + (h * vy_half)
    
        new_z_position = z + (h * vz_half)
    
        new_state = [new_x_position, new_y_position, new_z_position, vx_half, vy_half, vz_half]
        
        vx_new, vy_new, vz_new, ax_new, ay_new, az_new = cr3bp_equations_of_motion(mu,
                                                                               new_state)
        
        # Finding the new velocities with the new accelerations and positions:
        new_x_velo = vx_half + (ax_new * h / 2)
        
        new_y_velo = vy_half + (ay_new * h / 2)
    
        new_z_velo = vz_half + (az_new * h / 2)
    
        return np.array([new_x_position, new_y_position, new_z_position, new_x_velo, new_y_velo, new_z_velo])
    
    def runge_kutta(h,
                state_vector,
                mu):
            # Calculate all four terms then add them up later:
        
        first_term = h * cr3bp_equations_of_motion(mu,
                                               state_vector)
        
        second_term = h * cr3bp_equations_of_motion(mu,
                                                state_vector + 0.5 * first_term)
        
        third_term = h * cr3bp_equations_of_motion(mu,
                                               state_vector + 0.5 * second_term)
        
        fourth_term = h * cr3bp_equations_of_motion(mu,
                                                state_vector + third_term)
    
        # Adding up the terms:
        rk_solution = (1 / 6) * (first_term + 2 * second_term + 2 * third_term + fourth_term)
        
        
        return state_vector + rk_solution
    
    
    for step in range(1, n_steps +1):
        if method == "rk4":
            state_vector_norm = runge_kutta(dt, state_vector_norm, synodic_frame.mu)
            positions, velocities = synodic_frame.denormalize_state_vector(state_vector_norm[:3], state_vector_norm[3:])
            time = step * dt * synodic_frame.time_characteristic
            target.update_state(positions, velocities, time)
        
        else:
            state_vector_norm = leapfrog_verlet(dt, state_vector_norm, synodic_frame.mu)
            positions, velocities = synodic_frame.denormalize_state_vector(state_vector_norm[:3], state_vector_norm[3:])
            time = step * dt * synodic_frame.time_characteristic
            target.update_state(positions, velocities, time)
    
    if calc_uncertainties == False:
        return None
    
    # Create an empty list that will later be appended to with the 
    # uncertainties:
    uncertainty_targets_list = []
    
    # Now, grab the first element of the position and velocity history, since 
    # that is the initial conditions and make a copy of them. This is to prevent
    # any mishaps by accidentally writing over the position and velocity 
    # history:
    initial_pos = target.position_history[0].copy()
    initial_vel = target.velocity_history[0].copy()
    
    for i in range(n_samples):
        
        # First, I need to create perturbed initial conditions. By using
        # random.randn(3), this ensures that a Gaussian is picked for all 3 
        # coordinates, xyz:
        initial_perturbed_pos = initial_pos + np.random.randn(3) * position_uncertainty
        initial_perturbed_vel = initial_vel + np.random.randn(3) * velocity_uncertainty
        
        unc_target = TargetObject(name = "Uncertainty " + str(i + 1),
                                        position = initial_perturbed_pos,
                                        velocity = initial_perturbed_vel)
        
        uncertain_target_norm, _ = synodic_frame.normalize_initial_conditions(initial_perturbed_pos,
                                                                              initial_perturbed_vel)
        uncertain_target_norm = np.concatenate([uncertain_target_norm,
                                                initial_perturbed_vel / synodic_frame.velo_characteristic])
        
        # Now propagate for every uncertainty:
        for step in range(1, n_steps + 1):
            if method == "rk4":
                uncertain_target_norm = runge_kutta(dt, uncertain_target_norm, synodic_frame.mu)
            else:
                uncertain_target_norm = leapfrog_verlet(dt, uncertain_target_norm, synodic_frame.mu)
            
            pos, vel = synodic_frame.denormalize_state_vector(uncertain_target_norm[:3],
                                                              uncertain_target_norm[3:])
            
            time = step * dt * synodic_frame.time_characteristic
            unc_target.update_state(pos, vel, time)
            
        uncertainty_targets_list.append(unc_target)
        
    return uncertainty_targets_list
        
    

# Creating access affects:
# I will be using vector calculus to determine if a satellite's line of sight to 
# the target object is blocked by the Earth or Moon. I used ClaudeAI to help me 
# get started with implementing the vector math and much of my code here is 
# influenced by its suggested code, however it has changed to fit my needs a bit 
# better than its overly specific example.

def check_satellite_visibility(satellites, 
                               e_pos, 
                               m_pos, 
                               t_pos,
                               e_rad = 6371,
                               m_rad = 1737.5,
                               satellite_positions = None):
    
    def access_effects(sattelite_position, 
                   target_position, 
                   blocker_radius, 
                   blocker_center):
    
        # First, I will start by getting the distance between the observing
        # satellite and the target object:
        sat_target_dist = target_position - sattelite_position
    
        # Now I will be getting the distance from the satellite's position to the 
        # center of the blocking object (Earth or Moon)
        sat_block_dist = sattelite_position - blocker_center
    
        # So, this is the area where a lot of background math is happening: The
        # point of this code is to draw a line between the satellite and target 
        # object. If that line is crossed by a blocker, the satellite is blind. 
        # To calculate if this line is crossed by a blocker, you first have to 
        # parametrize the line:
        #
        # P(t) = satellite_position + t x sat_target_dist
        #
        # t acts as the point you are at along the line and is between 0 and 1. 0 
        # means that you are currently at the satellite and 1 means you're at the
        # target. So we want to search in between the 0 and 1 to see if a blocker
        # has crossed this line!
        # 
        # To determine if a blocker crosses P(t), this is where the sat_block_dist
        # comes into play. We are going to use the vector created from the
        # sat_block_dist and use the blocker_radius to see if P(t) intersects
        # anywhere within or at the radius.
        #
        # | P(t) - blocker_center |^2 = blocker_radius^2
        #
        # | (satellite_position + t x sat_target_dist) - blocker_center|^2 = blocker_radius^2
        #
        # | sat_block_dist + t x sat_target_dist|^2 = blocker_radius^2
        #
        # You can use that |v|^2 = v dot v and simplify down to get a quadratic
        # equation:
        #
        # (sat_target_dist * sat_target_dist)t^2 + 2t(sat_block_dist * sat_target_dist) + ((sat_block_dist * sat_block_dist) - blocker_radius^2) = 0
        # All of this above is why I can declare a, b, and c values:
    
        a = np.dot(sat_target_dist, sat_target_dist)
        b = 2 * np.dot(sat_block_dist, sat_target_dist)
        c = np.dot(sat_block_dist, sat_block_dist) - blocker_radius**2
    
        # We can use the discriminant to determine if the line of sight is blocked
        # (so discrim > 0), just touching (discrim = 0, this would still be
        # considered blocked later on), or not blocked (discrim < 0)
        discrim = b**2 - (4 * a * c)
    
        # If discrim is greater or equal to zero, let's handle that:
        if discrim >= 0:

            sqrt_discrim = np.sqrt(discrim)
            
            # Plug into the quadratic equation:
            t1 = (-b - sqrt_discrim) / (2 * a)
            t2 = (-b + sqrt_discrim) / (2 * a)
            
            # Check if either intersection point is between satellite and target
            if (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1):
                return True
    
        # If not, then let's return False to the funciton
        return False


    def check_blockers(sat_pos, earth_pos, moon_pos, target_pos,
                        earth_rad = 6371, 
                        moon_rad = 1737.5):
    
        earth_blocking = access_effects(sattelite_position = sat_pos,
                                    target_position = target_pos,
                                    blocker_radius = earth_rad,
                                    blocker_center = earth_pos)
    
        if earth_blocking == True:
            return False, "Earth"
    
    
        moon_blocking = access_effects(sattelite_position = sat_pos,
                                   target_position = target_pos,
                                   blocker_radius = moon_rad,
                                   blocker_center = moon_pos)
    
        if moon_blocking == True:
            return False, "Moon"
    
        return True, None
    
    visibility_status = {}
    visibility_satellites = []
    
    for i in satellites:
        
        if satellite_positions != None:
            s_pos = satellite_positions[i.name]
        else:
            s_pos = i.position
        
        visibility, blocker = check_blockers(sat_pos = s_pos,
                                             target_pos = t_pos,
                                             earth_pos = e_pos,
                                             moon_pos = m_pos,
                                             earth_rad = e_rad,
                                             moon_rad = m_rad)
        
        visibility_status[i.name] = {"visible": visibility,
                                     "blocking object": blocker}
        
        if visibility == True:
            visibility_satellites.append(i)
        
    return visibility_status, visibility_satellites
        
            
# Creating the plotting function since I was thinking about potentially plotting
# many still images (basically one image for each time step) and creating a gif
# of them. Within the plotting code, I want to update the line of sight of the
# observing satellites if they do or don't see the target object

def plotting_code(satellites, 
                  target_estimated,
                  target_true, 
                  earth_position,
                  moon_position,
                  title,
                  save_gif = True,
                  uncertainty_targets = None,
                  blind_index = None):
    
    if uncertainty_targets is not None:
        print(f"uncertainty_targets type: {type(uncertainty_targets)}")
        print(f"First element type: {type(uncertainty_targets[0])}")
        print(f"Has position_history: {hasattr(uncertainty_targets[0], 'position_history')}")
    
    ### Setting up number of frames, trajectories, 3D spheres, and initializing
    ### the figure:
   
# ======================================================================== #
    sf = SynodicFrame()
    
    num_frames = len(target_estimated.position_history)
    
    estimated_trajectory = np.array(target_estimated.position_history[:num_frames])
    true_trajectory = np.array(target_true.position_history[:num_frames])
    
    # I am going to add in the uncertainty trajectories to use later:
    uncertainty_trajectories = []
    if uncertainty_targets != None:
        for unc_target in uncertainty_targets:
            unc_traj = np.array(unc_target.position_history[:num_frames])
            uncertainty_trajectories.append(unc_traj)
        
    
    # Just like with target_trajectory above, I am going to precalculate the 
    # trajectory of the satellite objects before plotting. I'm doing this to
    # avoid looping through each frame and calculating the position at each
    # time step. First, I'm going to set up a dictonary that will contain all
    # of the satellite trajectories, then I will append to that dictionary:
    
    satellite_trajectories = {}
    
    # Calculating trajectories:
    
    for sat in satellites:
        
        # Right now, I'm only going to check to see if a satellite is in GEO.
        # If so, then I will compute its trajectory and update it in the
        # plotting code. I'm keeping the Lagrange satellites stationary for now.
        
        if "GEO" in sat.name:
            # Calculating GEO trajectory for all frames, starting with an empty
            # array, then calculating the period of GEO (it's the same as 
            # Earth's rotational period, approximately 24 hours):
            
            geo_trajectory = np.zeros((num_frames, 3))
            geo_period = 2 * np.pi / np.sqrt(sf.mu_earth / (42164**3))
            
            # Now I have to calculate how far the GEO satellite is from the
            # center of Earth. I am assuming circular orbits here, so the radius
            # should be approximately a little over 42k kilometers:

            initial_offset = sat.position - earth_position # This should give us our radius in 3D
            
            # Calculating the norm like this (with the [:2] at the end) since
            # we only care about the x and y coordinates since GEO always 
            # remains at the equator! Its GeoSTATIONARY not GeoSYNCHRONOUS. 
            # Remember, Geosynchronous orbit is tilted, whereas Geostationary
            # always sits at the equator:
            radius = np.linalg.norm(initial_offset[:2])
            
            # I'm going to assume the initial angle is at 0
            initial_angle = np.arctan2(initial_offset[1], initial_offset[0])
            
            # Calculating the position as the orbit goes around Earth:
            for i in range(num_frames):
                # Gets time at current frame.
                time = target_estimated.time_history[i] if i < len(target_estimated.time_history) else 0
                # Calculates the new angle that the satellite is at:
                angle = initial_angle + (time / geo_period) * 2 * np.pi
                
                # Now calculating the x, y, z positions based on the things we
                # calculated above. Switches from polar to cartesian
                geo_trajectory[i, 0] = earth_position[0] + radius * np.cos(angle)
                geo_trajectory[i, 1] = earth_position[1] + radius * np.sin(angle)
                geo_trajectory[i, 2] = earth_position[2]
            
            satellite_trajectories[sat.name] = geo_trajectory
        else:
            # I want the L4 satellite to stay stationary for now. Maybe will 
            # add in motion later...
            # However, this tile() function creates the same values over and
            # over again based on the previous array you put in there,
            # in my case, it's sat.position
            satellite_trajectories[sat.name] = np.tile(sat.position, (num_frames, 1))
    
    # Plot Earth
    earth_radius = 6371
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v)) + earth_position[0]
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v)) + earth_position[1]
    z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + earth_position[2]
    
    # Plot Moon
    moon_radius = 1737.5
    x_moon = moon_radius * np.outer(np.cos(u), np.sin(v)) + moon_position[0]
    y_moon = moon_radius * np.outer(np.sin(u), np.sin(v)) + moon_position[1]
    z_moon = moon_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + moon_position[2]
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
# ======================================================================== #
    
    ### Now creating a dictionary that will be updated throughout the plotting
    ### function:
    
    plot_elements = {
        'earth': None,
        'moon': None,
        'true_point': None,
        'true_trail': None,
        'estimated_point': None,
        'estimated_trail': None,
        'uncertainty_points': [],
        'uncertainty_trails': [],
        'blind_marker': None,
        'satellites': {},
        'los_lines': {},
        'sat_labels': {},
        'time_text': None,
        'visibility_text': None
    }
    
# ======================================================================== #


    ### Now I will be creating an initializer function that sets up the static
    ### portions of my plot like the Earth, Moon, plot limits, aspect ratios,
    ### and the legend
    
    def initializer():
        
        plot_elements['earth'] = ax.plot_surface(x_earth, y_earth, z_earth, 
                                                 color='blue', 
                                                 alpha=0.4,
                                                 linewidth = 0, 
                                                 antialiased = True)
        
        plot_elements['moon'] = ax.plot_surface(x_moon, y_moon, z_moon, 
                                                color='gray', 
                                                alpha=0.4, 
                                                linewidth = 0, 
                                                antialiased = True)
        
        x_range = np.max(np.abs(estimated_trajectory[:, 0]))
        y_range = np.max(np.abs(estimated_trajectory[:, 1]))
        z_range = np.max(np.abs(estimated_trajectory[:, 2]))
        
        earth_dist = np.linalg.norm(earth_position)
        moon_dist = np.linalg.norm(moon_position)
        
        geo_range = 42164 + np.linalg.norm(earth_position)
        max_range = max(x_range, y_range, z_range, moon_dist, earth_dist, geo_range) * 1.2
        # max_range = max(
        # np.max(np.abs(target_trajectory)),
        # np.linalg.norm(moon_position)
        # ) * 1.2
    
        ax.set_xlim([-max_range/4, max_range])
        ax.set_ylim([-max_range/2, max_range/2])
        ax.set_zlim([-max_range/4, max_range/4])
    
        # Set aspect ratio for better viewing
        ax.set_box_aspect([1, 1, 0.5])
        
        ax.set_xlabel("X Direction (In Kilometers)", fontsize = 10)
        ax.set_ylabel("Y Direction (In Kilometers)", fontsize = 10)
        ax.set_zlabel("Z Direction (In Kilometers)", fontsize = 11)
        ax.set_title(title, fontsize = 14, fontweight = "bold")
        
        legend_objects =    [Line2D([0], [0], color = 'green', linewidth = 3, 
                             label = "Visible"),
                            Line2D([0], [0], color = 'red', linewidth = 3, 
                             label = "Blocked"),
                            Line2D([0], [0], marker = "*", color = "black",
                                   markerfacecolor = "red", markersize = 12,
                                   label = "True Trajectory"),
                            Line2D([0], [0], marker = "*", color = "black", 
                             markerfacecolor = "purple", markersize = 12,
                             label = "Estimated Trajectory"),
                            Line2D([0], [0], marker = "^", color = "black", 
                             markerfacecolor = "green", markersize = 10,
                             label = "Satellite")]
        
        if len(uncertainty_trajectories) > 0:
            legend_objects.append(Line2D([0], [0], 
                                         color = "orange", 
                                         linewidth = 2,
                                         alpha = 0.5,
                                         label = f"Uncertainty Samples"))
    
        ax.legend(handles = legend_objects, loc = "upper right", fontsize = 10)
        
        return []
    
# ======================================================================== #
    
    ### This next function does the heavy lifting of this section of the
    ### plotting code. It updates the animation of every frame. This function
    ### is the reason why the target object and satellites animate on the plot.
    
    def update(frame):
        """Update animation for each frame."""
        if frame < 5 or frame == blind_index:
            print(f"Frame {frame}: time = {target_estimated.time_history[frame] if frame < len(target_estimated.time_history) else 'N/A'}")
        # Clear previous dynamic elements. The reason I have to do this is
        # because matplotlib doesn't already do this. If I didn't clear out the
        # previous dynamic elements, you would see every target point plotted
        # throughout the animation. In past animatioon code that I wrote, that
        # was okay! For this, however, I do not want that for this plot.
        
        # Clearing out previous red star marker (the estimated trajectory):
        if plot_elements['estimated_point'] is not None:
            plot_elements['estimated_point'].remove()
        if plot_elements['estimated_trail'] is not None:
            plot_elements['estimated_trail'].remove()
        
        # Clearing out the previous purple star marker, the true trajectory:
        if plot_elements['true_point'] is not None:
            plot_elements['true_point'].remove()
        if plot_elements['true_trail'] is not None:
            plot_elements['true_trail'].remove()
            
        # Now time to get rid of the previous points and trails for the 
        # uncertainty targets, the orange circles:
        for point in plot_elements['uncertainty_points']:
            point.remove()
        plot_elements["uncertainty_points"] = []
        
        # Now clear out the orange trails from before:
        for trail in plot_elements["uncertainty_trails"]:
            trail.remove()
        plot_elements["uncertainty_trails"] = []
        
        # Clear out the previous satellite location, line of sight location,
        # and their labels (so we can see when the satellite is blocked or not,
        # this allows for the label to clear out what it was before and give
        # the correct updated line of sight):
        for key in ['satellites', 'los_lines', 'sat_labels']:
            for elem in plot_elements[key].values():
                if isinstance(elem, list):
                    for e in elem:
                        e.remove()
                else:
                    elem.remove()
            plot_elements[key] = {}
        
        # Remove the text that tells us what frame we are at and what physical
        # time it is, and if the satellites can see at that given time step:
        if plot_elements['time_text'] is not None:
            plot_elements['time_text'].remove()
        if plot_elements['visibility_text'] is not None:
            plot_elements['visibility_text'].remove()
        
        # Get current target position
        current_estimated_pos = estimated_trajectory[frame]
        current_true_pos = true_trajectory[frame]
        
        # Get current GEO satellite position
        current_sat_positions = {}
        for i in satellites:
            current_sat_positions[i.name] = satellite_trajectories[i.name][frame]
        
        # Now time to plot the uncertainty trajectories:
        if len(uncertainty_trajectories) > 0 and blind_index != None:
            
            # The code will only start to plot the uncertainties after the first
            # instance when the constellation has gone blind:
            if frame >= blind_index:
                for unc_traj in uncertainty_trajectories:
                    # First to plot the trail:
                    if frame > blind_index:
                        trail = unc_traj[blind_index:frame + 1]
                        trail_line = ax.plot(
                            trail[:, 0], trail[:, 1], trail[:, 2],
                            "orange", linewidth = 0.5, alpha = 0.3
                        )[0]
                        plot_elements["uncertainty_trails"].append(trail_line)
                    
                    current_unc_pos = unc_traj[frame]
                    point = ax.scatter(*current_unc_pos, c = "orange", s = 50,
                                    marker = "o", alpha = 0.5, zorder = 4)
                    plot_elements["uncertainty_points"].append(point)
        
        # This code will create an X where the blind point happens, good for
        # visualizing exactly where the constellation goes blind!            
        if blind_index != None and frame >= blind_index:
            blind_pos = true_trajectory[blind_index]
            plot_elements["blind_marker"] = ax.scatter(
                *blind_pos, c = "yellow", s = 50, marker = "X",
                edgecolors = "red", linewidths = 1, zorder = 10
            )
        
        # Plotting the true trajectory of the target object with its point and 
        # trails:
        if frame > 0:
            trail = true_trajectory[:frame + 1]
            plot_elements["true_trail"] = ax.plot(trail[:, 0], 
                                                  trail[:, 1],
                                                  trail[:, 2],
                                                  "red",
                                                  linewidth = 2,
                                                  alpha = 0.6,
                                                  zorder = 3)[0]
            
        plot_elements["true_point"] = ax.scatter(*current_true_pos,
                                                 c = "red",
                                                 s = 250,
                                                 marker = "*",
                                                 edgecolors = "black",
                                                 linewidth = 2.5,
                                                 zorder = 6)
        
        
        
        # elif plot_elements["blind_marker"] != None and frame != blind_index:
        #     plot_elements["blind_marker"].remove()
        #     #plot_elements["blind_marker"] = None
        
        # Plot target trail (past positions)
        if frame > 0:
            trail = estimated_trajectory[:frame + 1]
            plot_elements['estimated_trail'] = ax.plot(
                trail[:, 0], trail[:, 1], trail[:, 2],
                'purple', linewidth=1, alpha=0.3
            )[0]
        
        # Plot current target position
        plot_elements['estimated_point'] = ax.scatter(
            *current_estimated_pos, c='purple', s=200, marker='*',
            edgecolors='black', linewidths=2, zorder=5
        )
        
        
        # Check visibility for each satellite
        vis_status, visible_sats = check_satellite_visibility(satellites, 
                                                              earth_position, 
                                                              moon_position, 
                                                              current_true_pos,
                                                              satellite_positions = current_sat_positions)
        
        # Plot each satellite and its line of sight
        for sat in satellites:
            status = vis_status[sat.name]
            is_visible = status['visible']
            
            current_sat_pos = current_sat_positions[sat.name]
            
            # Choose colors based on visibility
            if is_visible:
                sat_color = 'green'
                los_color = 'green'
                linestyle = '-'
                linewidth = 2.5
            else:
                sat_color = 'orange'
                los_color = 'red'
                linestyle = '--'
                linewidth = 2
            
            # Plot satellite
            plot_elements['satellites'][sat.name] = ax.scatter(
                *current_sat_pos, c=sat_color, s=150, marker='^',
                edgecolors='black', linewidths=1.5, zorder=5
            )
            
            # Plot line of sight
            plot_elements['los_lines'][sat.name] = ax.plot(
                [current_sat_pos[0], current_estimated_pos[0]],
                [current_sat_pos[1], current_estimated_pos[1]],
                [current_sat_pos[2], current_estimated_pos[2]],
                color=los_color, linestyle=linestyle, 
                linewidth=linewidth, alpha=0.7, zorder=3
            )
            
            # Label satellite with status
            label_text = sat.name
            if not is_visible:
                label_text += f"\n(blocked by {status['blocking object']})"
            
            plot_elements['sat_labels'][sat.name] = ax.text(
                current_sat_pos[0], current_sat_pos[1], current_sat_pos[2] + 20000,
                label_text, fontsize=9, ha='center', zorder=6,
                bbox=dict(boxstyle='round', facecolor=sat_color, 
                         alpha=0.7, edgecolor='black')
            )
        
        # Need to add in divergence information to time text:
        if frame > 0:
            divergence = np.linalg.norm(current_true_pos - current_estimated_pos)
        else:
            divergence = 0
        
        # Add time information
        time_elapsed = target_estimated.time_history[frame] if frame < len(target_estimated.time_history) else 0
        time_days = time_elapsed / 86400  # Convert seconds to days
        
        if blind_index != None:
            if frame >= blind_index:
                status = "BLIND PROPAGATION"
                blind_time = (time_elapsed - target_estimated.time_history[blind_index]) / 86400
                time_info = (f'Time: {time_days:.2f} days\n'
                             f"Frame: {frame + 1}/{num_frames}\n"
                             f"{status}\n"
                             f"Divergence: {divergence:.2f} km")
            else:
                time_info = (f"Time: {time_days:.2f} days\n"
                             f"Frame: {frame + 1}/{num_frames}\n"
                             f"TRACKING"
                             f"Divergence: {divergence:.2f} km")
        else:
            time_info = f"Time {time_days:.2f} days\nFrame: {frame + 1}/{num_frames}"
        
        plot_elements['time_text'] = ax.text2D(
            0.02, 0.98, time_info,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        # Add visibility summary
        n_visible = len(visible_sats)
        tracking_status = "✓ TRACKING" if n_visible >= 2 else "✗ BLIND"
        status_color = 'lightgreen' if n_visible >= 2 else 'lightcoral'
        
        vis_summary = f'{tracking_status}\n{n_visible}/{len(satellites)} satellites visible'
        plot_elements['visibility_text'] = ax.text2D(
            0.02, 0.02,
            vis_summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor=status_color, 
                     alpha=0.8, edgecolor='black', linewidth=2)
        )
        
        return []
    
    anim = FuncAnimation(fig, update, init_func=initializer, frames=num_frames,
                        interval= 50, blit=False, repeat=True)
    
    if save_gif == True:
        desktop = os.path.expanduser("~/Desktop/")
        gif_filename = str(title) + ".gif"
        full_path = os.path.join(desktop, gif_filename)
    
        print(f"Saving animation to: {full_path}")
    
        try:
            writer = ImageMagickWriter(fps=20)
            anim.save(full_path, writer=writer)
            print("✓ Animation saved successfully!")
        except Exception as e:
            print(f"✗ Error: {e}")
        
    plt.show()
    
    return fig, anim


def test_tracking_system(title,
                         satellites_in_system,
                         time_step = 0.01,
                         time = 1000,
                         integrator = "rk4",
                         target_name = "Lyapunov Orbit"):
    """
    Complete test of tracking system with classes.
    """
    
    # Creating helper functions to make the code cleaner:
    
    def get_satellite_positions_at_time(satellites, 
                                        time, 
                                        earth_position, 
                                        synodic_frame):
        """
        Calculate satellite positions at a specific time. Right now as the code
        stands, this really only matters for a GEO satellite since that is the
        only one that is moving right now. The Lagrange satellites are
        stationary (which is not realistic!) but maybe I will make it more
        realistic later.
        
        Parameters
        ----------
        satellites : list
            List of ObservingSatellite objects
        time : float
            Time in seconds
        earth_position : ndarray
            Earth position
        synodic_frame : SynodicFrame
            Reference frame
        
        Returns
        -------
        positions : dict
            {satellite_name: position_array}
        """
        positions = {}
        
        for sat in satellites:
            if "GEO" in sat.name:
                # Calculate GEO position at this time
                geo_radius = 42164  # km
                geo_period = 2 * np.pi / np.sqrt(synodic_frame.mu_earth / geo_radius**3)
                
                # Get radius from initial position
                initial_offset = sat.position - earth_position
                radius = np.linalg.norm(initial_offset[:2])
                
                # Calculate angle at this time
                initial_angle = 0  # Assume starts at 0
                angle = initial_angle + (time / geo_period) * 2 * np.pi
                
                # Position in equatorial plane
                positions[sat.name] = np.array([
                    earth_position[0] + radius * np.cos(angle),
                    earth_position[1] + radius * np.sin(angle),
                    earth_position[2]
                ])
            else:
                # Lagrange satellites stay stationary in synodic frame
                positions[sat.name] = sat.position.copy()
        
        return positions
    
    
    def create_satellite_constellation(satellite_names,
                                       synodic_frame, 
                                       earth_position,
                                       target_name = "Lyapunov Target"):
        
        satellites = []
        geo_tracker = 0
        
        for i, names in enumerate(satellite_names):
            
            if names == "GEO":
                geo_radius = 42164

                num_geo_total = satellite_names.count("GEO")
                starting_angle = (geo_tracker * 2 * np.pi) / num_geo_total
                # GEO satellite
                geo_pos = earth_position + np.array([geo_radius * np.cos(starting_angle),
                                                     geo_radius * np.sin(starting_angle),
                                                     0])
                
                geo_angular_vel = np.sqrt(synodic_frame.mu_earth / geo_radius**3)
                omega_diff = geo_angular_vel - synodic_frame.omega
                
                geo_vel = np.array([-geo_radius * omega_diff * np.sin(starting_angle),
                                    geo_radius * omega_diff * np.cos(starting_angle),
                                    0])
                
                sat = ObservingSatellite(f"GEO-{i}", geo_pos, geo_vel)
                satellites.append(sat)
                geo_tracker += 1
            
            elif names == "L4":
                L_points = synodic_frame.get_lagrange_points()
                sat = ObservingSatellite("L4-Observer", L_points["L4"], [0, 0, 0])
                satellites.append(sat)
            
            elif names == "L5":
                L_points = synodic_frame.get_lagrange_points()
                sat = ObservingSatellite("L5-Observer", L_points["L5"], [0, 0, 0])
                satellites.append(sat)
    
        return satellites


    def create_lyapunov_target(synodic_frame,
                                object_name):
        """
        Create target on L2 Lyapunov orbit.
    
        REPLACES all the manual target setup code!
        """
        L2_x_norm = 1 - synodic_frame.mu + (synodic_frame.mu/3)**(1/3)
        Az = 0.02
    
        target_pos_norm = np.array([L2_x_norm, 0, Az])
        target_vel_norm = np.array([0, 0.15 * Az, 0])
    
        target_pos, target_vel = synodic_frame.denormalize_state_vector(
            target_pos_norm, target_vel_norm
        )
    
        return TargetObject(object_name, target_pos, target_vel)


    def find_blind_time(target, satellites, earth_position, moon_position, 
                   synodic_frame):
        """
        Find when satellites go blind (< 2 visible).
    
        Uses get_satellite_positions_at_time() helper!
        """
    
        num_frames = len(target.position_history)
    
        for frame in range(num_frames):
            target_pos = np.array(target.position_history[frame])
            time = target.time_history[frame]
        
            # Use helper instead of manual calculation!
            sat_positions = get_satellite_positions_at_time(
                satellites, time, earth_position, synodic_frame
            )
        
            # Check visibility
            vis_status, visible_sats = check_satellite_visibility(
                satellites, earth_position, moon_position, target_pos,
                satellite_positions=sat_positions
            )
        
            if len(visible_sats) < 2:
                return frame
    
        return None
    
    # Initialize synodic frame
    sf = SynodicFrame()
    earth_pos = sf.get_earth_position()
    moon_pos = sf.get_moon_position()
    
    satellites = create_satellite_constellation(satellites_in_system,
                                                sf,
                                                earth_pos)
    
    target = create_lyapunov_target(sf, target_name)
    
    # Now let's propogate the trajectories!
    
    # Starting with propogating the TRUE trajectory of the target object. This 
    # object will be represented by the purple star in the simulation:
    target_true = copy.deepcopy(target)
    propagation_function(target_true,
                         time_step,
                         time,
                         sf,
                         integrator,
                         n_samples = 0,
                         calc_uncertainties = False)
    
    # Now let's find where the constellation went blind:
    blind_index = find_blind_time(target_true,
                                  satellites,
                                  earth_pos,
                                  moon_pos,
                                  sf)
    
    # This calculates the time (in days) when the constellation goes blind!
    # This is the moment when our estimated trajectory should deviate quite
    # a lot from the purple star. Remember, our estimation is going to be
    # represented by the red star!
    blind_time = target_true.time_history[blind_index] / 86400
    
    # Now let's go ahead and track up until the blind point:
    target_estimated = copy.deepcopy(target)
    propagation_function(target_estimated, 
                         time_step,
                         blind_index,
                         sf,
                         integrator,
                         n_samples = 0,
                         calc_uncertainties = False)
    
    # We have to calculate the trajectory up to the point where the 
    # constellation goes blind:
    estimated_pos_at_blind = np.array(target_estimated.position_history[-1])
    estimated_vel = np.array(target_estimated.velocity_history[-1])
    
    sat_positions = get_satellite_positions_at_time(satellites,
                                                    target_estimated.time_history[-1],
                                                    earth_pos,
                                                    sf)
    
    # Now we are going to measure with parallax the average estimates
    mean_estimate, _, _, uncertainty = calculate_parallax(satellites[0],
                                                          satellites[1],
                                                          estimated_pos_at_blind)
    error = np.linalg.norm(mean_estimate - estimated_pos_at_blind)
    
    print(f"\nLast measurement:")
    print(f"  Error: {error:.3f} km")
    print(f"  Uncertainty: {uncertainty:.3f} km")
    
    # Now we will go forward and propagate blindly:
    target_blind = TargetObject("Blind", mean_estimate, estimated_vel)
    remaining_steps_in_sim = time - blind_index
    
    uncertainty_targets = propagation_function(target_blind,
                                               time_step,
                                               remaining_steps_in_sim,
                                               sf,
                                               integrator,
                                               n_samples = 10, 
                                               calc_uncertainties = True)
    
    # Now we have to stich together the original trajectory in the beginning
    # with the blind estimate:
    
    # But first, we have to make time continuous so there are no skips in the 
    # animation:
    time_offset = target_estimated.time_history[blind_index - 1]
    
    for i in range(1, len(target_blind.position_history)):
        target_estimated.position_history.append(target_blind.position_history[i])
        target_estimated.velocity_history.append(target_blind.velocity_history[i])
        
        # Now account for the time offsent and make it continuous:
        target_estimated.time_history.append(target_blind.time_history[i] + time_offset)
        
    # Now we have to pad the uncertainty trajectories! If we don't pad this out,
    # our plot will not work correctly. In fact, nothing will plot at all. This
    # is due to not having indicies aligned! The following dummy variables are
    # not going to be plotted, but they're just there to keep indicies the same:
    if uncertainty_targets != None:
        for k in uncertainty_targets:
            
            # Creating dummy variables for blind_index points:
            dummy_pos = target_estimated.position_history[0]
            dummy_vel = target_estimated.velocity_history[0]
            
            for i in range(blind_index):
                k.position_history.insert(0, dummy_pos)
                k.velocity_history.insert(0, dummy_vel)
                k.time_history.insert(0, target_estimated.time_history[i])
                
        
                
    # ========== RESULTS ==========
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    final_true = np.array(target_true.position_history[-1])
    final_est = np.array(target_estimated.position_history[-1])
    final_error = np.linalg.norm(final_true - final_est)
    
    blind_duration = (target_true.time_history[-1] - 
                     target_true.time_history[blind_index]) / 86400
    
    print(f"Blind duration: {blind_duration:.2f} days")
    print(f"Final divergence: {final_error:.1f} km")
    
    
    print("\n" + "="*70)
    print("DIAGNOSTIC INFORMATION")
    print("="*70)

    print(f"\nTarget estimated:")
    print(f"  Length: {len(target_estimated.position_history)}")
    print(f"  First pos: {target_estimated.position_history[0]}")
    print(f"  Last pos: {target_estimated.position_history[-1]}")

    print(f"\nUncertainty targets:")
    if uncertainty_targets:
        print(f"  Number: {len(uncertainty_targets)}")
        print(f"  First target length: {len(uncertainty_targets[0].position_history)}")
        print(f"  First position: {uncertainty_targets[0].position_history[0]}")

    print(f"\nBlind index: {blind_index}")
    print(f"Earth pos: {earth_pos}")
    print(f"Moon pos: {moon_pos}")

    # Check if positions are valid
    all_positions = np.array(target_estimated.position_history)
    print(f"\nPosition ranges:")
    print(f"  X: [{all_positions[:, 0].min():.1f}, {all_positions[:, 0].max():.1f}]")
    print(f"  Y: [{all_positions[:, 1].min():.1f}, {all_positions[:, 1].max():.1f}]")
    print(f"  Z: [{all_positions[:, 2].min():.1f}, {all_positions[:, 2].max():.1f}]")
    
    # ========== PLOT ==========
    fig_anim, anim = plotting_code(
        satellites=satellites,
        target_estimated=target_estimated,
        target_true=target_true,
        earth_position=earth_pos,
        moon_position=moon_pos,
        title = title,
        save_gif=False,
        uncertainty_targets=uncertainty_targets,
        blind_index = blind_index
    )
    



###########################################################################
###########################################################################
###                                                                     ###
###                            FOR LIVE DEMO                            ###
###                                                                     ###
###########################################################################
###########################################################################


###  Let's run a couple of experiments here. Which satellite architecture will   
###  yield us the most accurate results, i.e. which satellite constellation      
###  will be more accurate when tracking the target object?                      


##----------------------------------------------------------------
##          Test 1: Two Satellites in GEO Using Verlet           -
##----------------------------------------------------------------                

if __name__ == "__main__":
    
    test_tracking_system(title = "Tracking a Target Object At L2 Lyapunov Orbit with 2 GEO Observers Using Verlet Integrator",
                         integrator = "Verlet",
                         satellites_in_system = ["GEO", "GEO"])



##---------------------------------------------------------------
##            Test 2: Two Satellites in GEO Using RK4           -
##---------------------------------------------------------------                     

if __name__ == "__main__":
    
    test_tracking_system(title = "Tracking a Target Object At L2 Lyapunov Orbit with 2 GEO Observers Using Runge-Kutta Integrator",
                         integrator = "rk4",
                         satellites_in_system = ["GEO", "GEO"])
    

###  Conclusions: Using Two Satellites in GEO STINKS! This is due to a few         
###  major causes: calculating parallax with smaller angles leads to larger        
###  error, and choosing the numerical integrator can greatly vary your results!   



##----------------------------------------------------------------
##      Test 3: One Satellite in GEO, One at L4 Using Verlet     -
##----------------------------------------------------------------

if __name__ == "__main__":
    
    test_tracking_system(title = "Tracking a Target Object At L2 Lyapunov Orbit with 1 GEO and 1 L4 Observer Using Leapfrog-Verlet Integrator",
                         integrator = "verlet",
                         satellites_in_system = ["L4", "GEO"])
    
    
##---------------------------------------------------------------
##  Test 4: One Satellite in GEo, One at L4 Using Runge-Kutta   -
##---------------------------------------------------------------

if __name__ == "__main__":
    
    test_tracking_system(title = "Tracking a Target Object At L2 Lyapunov Orbit with 1 GEO and 1 L4 Observer Using Runge-Kutta Integrator",
                         integrator = "rk4",
                         satellites_in_system = ["L4", "GEO"])   
    
 
###  Conclusions: We are definitely getting better results, as expected since   
###  our angle for parallax is larger, but still the error is very large. It    
###  looks like Runge-Kutta is always going to be worse in these kinds of       
###  scenarios, so lets stick with Verlet for the rest of the experiment        


##-------------------------------------------------------------------
##  Test 5: One Satellite at L4, One Satellite at L5 Using Verlet   -
##-------------------------------------------------------------------

if __name__ == "__main__":
    
    test_tracking_system(title = "Tracking a Target Object At L2 Lyapunov Orbit with 1 L4 and 1 L5 Observer Using Leapfrog-Verlet Integrator",
                         integrator = "verlet",
                         satellites_in_system = ["L4", "L5"])  
    

###  Conclusions: Doing better, but still didn't gain as much in accuracy as I   
###  thought I would. I am noticing, however, that the time at which you         
###  observe the target is going to be critical, especially as the target        
###  approaches closer to a celestial body. These close-encounters can be        
###  highly sensitive, so perhaps in a more fleshed out model, you can adjust    
###  the satellites to always observe at a certain time, rather than all the     
###  time.                                                                       


##-----------------------------------------------------------------------------
##  Test 6: 3 Satellites Spread Across Lagrange Points and GEO using Verlet   -
##-----------------------------------------------------------------------------

if __name__ == "__main__":
    
    test_tracking_system(title = "Tracking a Target Object At L2 Lyapunov Orbit with 1 L4, 1 L5, and 1 GEO Observer Using Leapfrog-Verlet Integrator",
                         integrator = "verlet",
                         satellites_in_system = ["L4", "L5", "GEO"])  


###  Conclusion: Even with three satellites, the improvement in accuracy is not   
###  much better than two satellites in L4 and L5. With this low-level            
###  simulation, perhaps the best situational awareness architecture is two       
###  satellites in L4 and L5? Need to improve the model to see if this is true!   


##---------------------------------------------------------------
##                        Just for fun:                         -
##---------------------------------------------------------------

if __name__ == "__main__":
    
    test_tracking_system(title = "Tracking a Target Object At L2 Lyapunov Orbit with MAXIMUM POWER",
                         integrator = "verlet",
                         satellites_in_system = ["L4", "L5", "GEO", "GEO", "GEO"]) 