import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost


def estimate(particles, particles_w):
    mean = np.dot(particles.T, particles_w)
    mean = mean.reshape(-1,)

    return mean

def observe(particles, frame, H, W, hist_bin, hist_target, sigma_observe):
    height_frame, width_frame, _ = frame.shape

    # Compute color histograms for all particles
    hist_particles = np.zeros((len(particles), hist_bin * 3))
    for i, particle in enumerate(particles):
        x_min = max(0, int(particle[0] - 0.5 * W))
        y_min = max(0, int(particle[1] - 0.5 * H))
        x_max = min(width_frame, int(particle[0] + 0.5 * W))
        y_max = min(height_frame, int(particle[1] + 0.5 * H))

        hist_particles[i, :] = color_histogram(x_min, y_min, x_max, y_max, frame, hist_bin)

    # Calculate weights
    particles_w = np.zeros(len(particles))
    for i in range(len(particles)):
        chi2_distance = chi2_cost(hist_target, hist_particles[i, :])
        particles_w[i] = (1 / (np.sqrt(2 * np.pi) * sigma_observe)) * np.exp(-0.5 * chi2_distance / sigma_observe**2)

    # Normalization
    particles_w = particles_w / np.sum(particles_w)
    return particles_w


def propagate(particles, h, w, params):
    n_particles, dim = particles.shape

    if params['model'] == 0:  # (no motion)
        A = np.eye(2)
        particles = particles + np.random.randn(n_particles, dim) * params['sigma_position']
    else:  # constant velocity
        A = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        noise_position = np.random.randn(2, n_particles) * params['sigma_position']
        noise_velocity = np.random.randn(2, n_particles) * params['sigma_velocity']
        noise = np.vstack((noise_position, noise_velocity))
        particles = np.dot(A, particles.T) + noise
        particles = particles.T

    # Make sure that the particles lie inside the frame
    particles[:, :2] = np.clip(particles[:, :2], 0, [w-1, h-1])

    return particles

def resample(particles, particles_w):   
    num_particles = len(particles)

    # Normalize weights to make them probabilities
    if np.sum(particles_w) != 0:    
        particles_w = particles_w / np.sum(particles_w)
    else:
        particles_w = np.ones(num_particles) / num_particles
    
    # Resampling indices based on the weights
    resample_indices = np.random.choice(np.arange(num_particles), size=num_particles, p=particles_w)

    # Create new particles and weights based on resampling indices
    new_particles = particles[resample_indices]
    new_weights = particles_w[resample_indices]
    new_weights = np.array(new_weights) / np.sum(new_weights) # Normalize weights to make them probabilities

    return new_particles, new_weights
