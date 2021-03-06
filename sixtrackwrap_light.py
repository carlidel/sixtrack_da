from numba import jit, njit, prange
import numpy as np
# import sixtracklib as st
from conversion import convert_norm_to_physical, convert_physical_to_norm
import warnings
import os
import time
import pickle
from scipy.special import erf


@njit(parallel=True)
def accumulate_and_return(r, alpha, th1, th2, n_sectors):
    """Executes a binning of the radiuses over the th1-th2 phase space.

    Parameters
    ----------
    r : ndarray
        shape = (alpha_sampling, n_iterations)
    alpha : ndarray
        shape = (alpha_sampling, n_iterations)
    th1 : ndarray
        shape = (alpha_sampling, n_iterations)
    th2 : ndarray
        shape = (alpha_sampling, n_iterations)
    n_sectors : unsigned int
        binning fineness

    Returns
    -------
    tuple of stuff
        count matrices (alpha_sampling, n_sectors, n_sectors), average matrices (alpha_sampling, n_sectors, n_sectors), results (alpha_sampling).
        Average matrices are pure (no power is performed)
        Result is already fully processed (powered and unpowered properly)
    """
    tmp_1 = ((th1 + np.pi) / (np.pi * 2)) * n_sectors
    tmp_2 = ((th2 + np.pi) / (np.pi * 2)) * n_sectors

    i_1 = np.empty(tmp_1.shape, dtype=np.int32)
    i_2 = np.empty(tmp_2.shape, dtype=np.int32)

    for i in prange(i_1.shape[0]):
        for j in range(i_1.shape[1]):
            i_1[i, j] = int(tmp_1[i, j])
            i_2[i, j] = int(tmp_2[i, j])

    result = np.empty(r.shape[0])
    matrices = np.empty((r.shape[0], n_sectors, n_sectors))
    count = np.zeros((r.shape[0], n_sectors, n_sectors), dtype=np.int32)

    for j in prange(r.shape[0]):
        matrix = np.zeros((n_sectors, n_sectors)) * np.nan

        for k in range(r.shape[1]):
            if count[j, i_1[j, k], i_2[j, k]] == 0:
                matrix[i_1[j, k], i_2[j, k]] = r[j, k]
            else:
                matrix[i_1[j, k], i_2[j, k]] = (
                    (matrix[i_1[j, k], i_2[j, k]] * count[j, i_1[j, k],
                                                          i_2[j, k]] + r[j, k]) / (count[j, i_1[j, k], i_2[j, k]] + 1)
                )
            count[j, i_1[j, k], i_2[j, k]] += 1

        for a in range(matrix.shape[0]):
            for b in range(matrix.shape[1]):
                if matrix[a, b] == 0:
                    matrix[a, b] = np.nan

        result[j] = np.power(np.nanmean(np.power(matrix, 4)), 1/4)
        matrices[j, :, :] = matrix

    return count, matrices, result


def recursive_accumulation(count, matrices):
    """Execute a recursive accumulation of the binning performed on the th1 th2 phase space (fineness must be a power of 2!!!)

    Parameters
    ----------
    count : ndarray
        count matrix (provided by accumulate_and_return)
    matrices : ndarray
        average matrix (provided by accumulate_and_return)

    Returns
    -------
    tuple
        (count matrices, average matrices, result list, validity list)
        Average matrices are pure.
        Results are the radiuses elevated with power 4!
    """
    n_sectors = count.shape[1]
    c = []
    m = []
    r = []
    count = count.copy()
    matrices = matrices.copy()
    validity = []
    c.append(count.copy())
    m.append(matrices.copy())
    r.append(np.nanmean(np.power(matrices, 4), axis=(1, 2)))
    validity.append(np.logical_not(
        np.any(np.isnan(matrices), axis=(1, 2))))
    while n_sectors >= 2 and n_sectors % 2 == 0:
        matrices *= count
        count = np.nansum(count.reshape(
            (count.shape[0], n_sectors//2, 2, n_sectors//2, 2)), axis=(2, 4))
        matrices = np.nansum(matrices.reshape(
            (matrices.shape[0], n_sectors//2, 2, n_sectors//2, 2)), axis=(2, 4)) / count
        result = np.nanmean(np.power(matrices, 4), axis=(1, 2))
        c.append(count.copy())
        m.append(matrices.copy())
        r.append(result.copy())
        validity.append(np.logical_not(
            np.any(np.isnan(matrices), axis=(1, 2))))
        n_sectors = n_sectors // 2
    return c, m, r, np.asarray(validity, dtype=np.bool)


# def track_particles(x, px, y, py, n_turns, opencl=True):
#     """Wrap Sixtracklib and track the particles requested
    
#     Parameters
#     ----------
#     x : ndarray
#         initial conditions
#     px : ndarray
#         initial conditions
#     y : ndarray
#         initial conditions
#     py : ndarray
#         initial conditions
#     n_turns : unsigned int
#         number of turns to perform
#     opencl : bool (optional)
#         use opencl backend (default: True)
    
#     Returns
#     -------
#     particles object
#         Sixtracklib particles object
#     """
#     assert len(x) == len(px)
#     assert len(x) == len(py)
#     assert len(x) == len(y)

#     particles = st.Particles.from_ref(
#         num_particles=len(x), p0c=6.5e12)

#     particles.x += x
#     particles.px += px
#     particles.y += y
#     particles.py += py

#     lattice = st.Elements.fromfile(os.path.join(
#         os.path.dirname(__file__), 'data/beam_elements.bin'))
#     if opencl:
#         cl_job = st.TrackJob(lattice, particles, device="opencl:0.0")
#     else:
#         cl_job = st.TrackJob(lattice, particles)

#     status = cl_job.track_until(n_turns)
#     cl_job.collect_particles()
#     return particles


# def full_track_particles(radiuses, alpha, theta1, theta2, n_turns, opencl=True):
#     """Complete tracking of particles for the given number of turns
    
#     Parameters
#     ----------
#     radiuses : ndarray
#         initial conditions
#     alpha : ndarray
#         initial conditions
#     theta1 : ndarrayq
#         initial conditions
#     theta2 : ndarray
#         initial conditions
#     n_turns : unsigned int
#         number of turns to perform
#     opencl : bool (optional)
#         use opencl backend (default: True)
    
#     Returns
#     -------
#     tuple
#         (r, alpha, theta1, theta2), shape = (initial conditios, n turns)
#     """
#     x, px, y, py = polar_to_cartesian(radiuses, alpha, theta1, theta2)
#     x, px, y, py = convert_norm_to_physical(x, px, y, py)

#     particles = st.Particles.from_ref(
#         num_particles=len(x), p0c=6.5e12)

#     particles.x += x
#     particles.px += px
#     particles.y += y
#     particles.py += py

#     lattice = st.Elements.fromfile(os.path.join(
#         os.path.dirname(__file__), 'data/beam_elements.bin'))
#     if opencl:
#         cl_job = st.TrackJob(lattice, particles, device="opencl:0.0")
#     else:
#         cl_job = st.TrackJob(lattice, particles)

#     data_r = np.empty((len(x), n_turns))
#     data_a = np.empty((len(x), n_turns))
#     data_th1 = np.empty((len(x), n_turns))
#     data_th2 = np.empty((len(x), n_turns))

#     for i in range(n_turns):
#         status = cl_job.track_until(i)
#         cl_job.collect_particles()
#         # print(particles.at_turn)
#         t_x, t_px, t_y, t_py = convert_physical_to_norm(
#             particles.x, particles.px, particles.y, particles.py)
#         data_r[:, i], data_a[:, i], data_th1[:, i], data_th2[:,
#                                                              i] = cartesian_to_polar(t_x, t_px, t_y, t_py)

#     return data_r, data_a, data_th1, data_th2


@njit
def polar_to_cartesian(radius, alpha, theta1, theta2):
    """Convert polar coordinates to cartesian coordinates
    
    Parameters
    ----------
    radius : ndarray
        ipse dixit
    alpha : ndarray
        ipse dixit
    theta1 : ndarray
        ipse dixit
    theta2 : ndarray
        ipse dixit
    
    Returns
    -------
    tuple of ndarrays
        x, px, y, py
    """
    x = radius * np.cos(alpha) * np.cos(theta1)
    px = radius * np.cos(alpha) * np.sin(theta1)
    y = radius * np.sin(alpha) * np.cos(theta2)
    py = radius * np.sin(alpha) * np.sin(theta2)
    return x, px, y, py


@njit
def cartesian_to_polar(x, px, y, py):
    """Convert cartesian coordinates to polar coordinates
    
    Parameters
    ----------
    x : ndarray
        ipse dixit
    px : ndarray
        ipse dixit
    y : ndarray
        ipse dixit
    py : ndarray
        ipse dixit
    
    Returns
    -------
    tuple of ndarrays
        r, alpha, theta1, theta2
    """
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) +
                np.power(px, 2) + np.power(py, 2))
    theta1 = np.arctan2(px, x)
    theta2 = np.arctan2(py, y)
    alpha = np.arctan2(np.sqrt(y * y + py * py),
                       np.sqrt(x * x + px * px))
    return r, alpha, theta1, theta2


class radial_provider(object):
    """Base class for managing coordinate system on radiuses"""

    def __init__(self, alpha, theta1, theta2, dr, starting_step):
        """Init radial provider class
        
        Parameters
        ----------
        object : self
            base class for managing coordinate system on radiuses
        alpha : ndarray
            angles to consider
        theta1 : ndarray
            angles to consider
        theta2 : ndarray
            angles to consider
        dr : float
            radial step
        starting_step : unsiged int
            starting step point
        """
        assert starting_step >= 0
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.dr = dr
        self.starting_step = starting_step

        self.count = 0
        self.active = True

    def get_positions(self, n_pos):
        """Get coordinates and update count
        
        Parameters
        ----------
        n_pos : unsigned int
            number of coordinates to gather
        
        Returns
        -------
        tuple of ndarrays
            (radius, alpha, theta1, theta2)
        """
        assert n_pos > 0
        assert self.active
        a = np.ones(n_pos) * self.alpha
        th1 = np.ones(n_pos) * self.theta1
        th2 = np.ones(n_pos) * self.theta2
        r = np.linspace(self.starting_step + self.count, self.starting_step +
                        self.count + n_pos, n_pos, endpoint=False) * self.dr

        self.count += n_pos

        return r, a, th1, th2

    def peek_positions(self, n_pos):
        """Get coordinates WITHOUT updating the count
        
        Parameters
        ----------
        n_pos : unsigned int
            number of coordinates to gather
        
        Returns
        -------
        tuple of ndarrays
            (radius, alpha, theta1, theta2)
        """
        assert n_pos > 0
        assert self.active
        a = np.ones(n_pos) * self.alpha
        th1 = np.ones(n_pos) * self.theta1
        th2 = np.ones(n_pos) * self.theta2
        r = np.linspace(self.starting_step + self.count, self.starting_step +
                        self.count + n_pos, n_pos, endpoint=False) * self.dr

        return r, a, th1, th2

    def reset(self):
        """Reset the count and the status
        """
        self.active = True
        self.count = 0


class radial_scanner(object):
    def __init__(self, alpha, theta1, theta2, dr, starting_step=1):
        """Init a radial scanner object
        
        Parameters
        ----------
        object : radial scanner
            wrapper for doing a proper radial scanning
        alpha : ndarray
            angles to consider
        theta1 : ndarray
            angles to consider
        theta2 : ndarray
            angles to consider
        dr : float
            radial step
        starting_step : int, optional
            starting step, by default 1
        """
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.dr = dr
        self.starting_step = starting_step

        self.radiuses = [radial_provider(
            alpha[i], theta1[i], theta2[i], dr, starting_step) for i in range(len(alpha))]
        self.steps = [np.array([]) for i in range(len(alpha))]
        self.n_elements = len(alpha)
        self.weights = [np.array([]) for i in range(len(alpha))]
        self.min_time = np.nan
        self.max_time = np.nan

    # def scan(self, max_turns, min_turns, batch_size=int(10e4), opencl=True):
    #     """Perform a radial scanning
        
    #     Parameters
    #     ----------
    #     max_turns : unsigned int
    #         max number of turns to perform
    #     min_turns : unsigned int
    #         minimum number of turns to perform
    #     batch_size : unsigned int, optional
    #         batch size for parallel computing (OpenCL support), by default 10e4
    #     opencl : bool, optional
    #         use opencl backend, by default True
        
    #     Returns
    #     -------
    #     ndarray
    #         step informations (array of arrays)
    #     """
    #     self.max_time = max_turns
    #     self.min_time = min_turns
    #     total_time = 0
    #     time_per_iter = np.nan
    #     turns_to_do = max_turns
    #     while True:
    #         n_active = 0
    #         for radius in self.radiuses:
    #             if radius.active:
    #                 n_active += 1

    #         if n_active == 0:
    #             print(
    #                 "TOTAL ELAPSED TIME IN SECONDS: {:.2f}".format(total_time))
    #             maximum = 0
    #             for i in range(len(self.steps)):
    #                 maximum = max(maximum, len(self.steps[i]))
    #             temp = np.zeros((len(self.steps), maximum))
    #             for i in range(len(self.steps)):
    #                 temp[i, : len(self.steps[i])] = self.steps[i]
    #             self.steps = temp
    #             return self.steps

    #         if batch_size % n_active == 0:
    #             sample_size = int(batch_size // n_active)
    #         else:
    #             sample_size = int(batch_size // n_active) + 1

    #         print("Active radiuses:", n_active, "/", len(self.radiuses))
    #         print("Sample size per active radius:", sample_size)
    #         print("Expected execution time for step: {:.2f}".format(
    #               sample_size * n_active * time_per_iter * turns_to_do))

    #         r = []
    #         a = []
    #         th1 = []
    #         th2 = []

    #         for radius in self.radiuses:
    #             if radius.active:
    #                 t_r, t_a, t_th1, t_th2 = radius.get_positions(sample_size)
    #                 r.append(t_r)
    #                 a.append(t_a)
    #                 th1.append(t_th1)
    #                 th2.append(t_th2)

    #         r = np.asarray(r).flatten()
    #         a = np.asarray(a).flatten()
    #         th1 = np.asarray(th1).flatten()
    #         th2 = np.asarray(th2).flatten()

    #         x, px, y, py = polar_to_cartesian(r, a, th1, th2)
    #         x, px, y, py = convert_norm_to_physical(x, px, y, py)

    #         start = time.time()

    #         particles = track_particles(
    #             x, px, y, py, turns_to_do, opencl=opencl)

    #         end = time.time()
    #         time_per_iter = (end - start) / \
    #             (sample_size * n_active * turns_to_do)
    #         print("Elapsed time for whole iteration: {:.2f}".format(
    #             end - start))
    #         print("Time per single iteration: {}".format(time_per_iter))
    #         total_time += (end - start)
    #         turns = particles.at_turn

    #         i = 0
    #         turns_to_do = 0

    #         for j, radius in enumerate(self.radiuses):
    #             if radius.active:
    #                 r_turns = turns[i * sample_size: (i + 1) * sample_size]
    #                 if min(r_turns) < min_turns:
    #                     radius.active = False
    #                 self.steps[j] = np.concatenate((self.steps[j], r_turns))
    #                 turns_to_do = max(turns_to_do, min(r_turns))
    #                 i += 1
    #         print("r:", np.max(r), ". Turns to do:",
    #               turns_to_do, ". Min found:", np.min(turns))

    @staticmethod
    @njit
    def static_extract_DA(n_elements, sample_list, steps, dr, starting_step):
        values = np.empty((n_elements, len(sample_list)))
        for i in prange(n_elements):
            for j, sample in enumerate(sample_list):
                values[i, j] = np.argmin(steps[i] >= sample) - 1
                if values[i, j] < 0:
                    values[i, j] = np.nan
                else:
                    values[i, j] = (
                        values[i, j] + starting_step) * dr
        return values


    def extract_DA(self, sample_list):
        """Gather DA radial data from the step data
        
        Parameters
        ----------
        sample_list : ndarray
            values to consider
        
        Returns
        -------
        ndarray
            radial values (n_elements, sample_list)
        """
        return self.static_extract_DA(self.n_elements, sample_list, self.steps, self.dr, self.starting_step)

    def save_values(self, f, label="SixTrack LHC no bb"):
        self.label = label
        data_dict = {
            "label": label,
            "alpha": self.alpha,
            "theta1": self.theta1,
            "theta2": self.theta2,
            "dr": self.dr,
            "starting_step": self.starting_step,
            "values": self.steps,
            "weights": self.weights,
            "max_turns": self.max_time,
            "min_turns": self.min_time
        }
        with open(f, 'wb') as destination:
            pickle.dump(data_dict, destination, protocol=4)

    @classmethod
    def load_values(cls, f):
        with open(f, 'rb') as destination:
            data_dict = pickle.load(destination)

        instance = cls(
            data_dict["alpha"],
            data_dict["theta1"],
            data_dict["theta2"],
            data_dict["dr"],
            data_dict["starting_step"],
        )
        instance.steps = data_dict["values"]
        instance.weights = data_dict["weights"]
        instance.label = data_dict["label"]
        instance.max_time = data_dict["max_turns"]
        instance.min_time = data_dict["min_turns"]
        
        return instance

    def assign_weights(self, f=lambda r, a, th1, th2: r):
        """Assign weights to the various radial samples computed (not-so-intuitive to setup, beware...).

        Parameters
        ----------
        f : lambda, optional
            the lambda to assign the weights with, by default returns r
            this lambda has to take as arguments
            r : float
                the radius
            a : float
                the alpha angle
            th1 : float
                the theta1 angle
            th2 : float
                the theta2 angle
        """
        self.weights = np.zeros_like(self.steps)

        for i in range(self.weights.shape[0]):
            self.weights[i] = np.array([
                f(
                    self.dr * (j + self.starting_step),
                    self.alpha[i],
                    self.theta1[i],
                    self.theta2[i]
                ) for j in range(self.weights.shape[1])
            ])
            self.weights[i][1:] = np.diff(self.weights[i])

    def compute_loss(self, sample_list, cutting_point=-1.0, normalization=True):
        """Compute the loss based on a boolean masking of the various timing values.

        Parameters
        ----------
        sample_list : ndarray
            list of times to use as samples
        cutting_point : float, optional
            radius to set-up as cutting point for normalization purposes, by default -1.0
        normalization : boolean, optional
            execute normalization? By default True

        Returns
        -------
        ndarray
            the values list measured (last element is the cutting point value 1.0 used for renormalization of the other results.)
        """
        if cutting_point > self.starting_step:
            cutting_point = int((cutting_point / self.dr)
                                ) - self.starting_step + 1
            assert cutting_point < self.weights.shape[1]
        values = np.empty(len(sample_list) + 1)
        values[-1] = sum([np.sum(weight[:cutting_point])
                          for weight in self.weights])
        for i, sample in enumerate(sample_list):
            values[i] = np.sum(self.weights * (self.steps >= sample))
        if normalization:
            values /= values[-1]
        return values

    def compute_loss_cut(self, cutting_point):
        """Compute the loss based on a simple DA cut.

        Parameters
        ----------
        cutting_point : float
            radius to set-up as cutting point

        Returns
        -------
        float
            the (not-normalized) value
        """
        cutting_point = int((cutting_point / self.dr)
                            ) - self.starting_step + 1
        assert cutting_point < self.weights.shape[1]
        return sum([np.sum(weight[:cutting_point])
                    for weight in self.weights])


def assign_symmetric_gaussian(sigma=1.0):
    def f(r, a, th1, th2):
        return (
            - np.exp(- ((r / sigma) ** 2) / 2) * (r ** 2 + 2 * sigma ** 2) 
            + 2 * sigma ** 2
        )
    return f


def assign_uniform_distribution():
    def f(r, a, th1, th2):
        return (
            r ** 4 / 4
        )
    return f


def assign_generic_gaussian(sigma_x, sigma_px, sigma_y, sigma_py):
    def f(r, a, th1, th2):
        x, px, y, py = polar_to_cartesian(r, a, th1, th2)
        x /= sigma_x
        px /= sigma_px
        y /= sigma_y
        py /= sigma_py
        r, a, th1, th2 = cartesian_to_polar(x, px, y, py)
        return (
            - np.exp(- ((r) ** 2) / 2) * (r ** 2 + 2)
            + 2
        )
    return f
